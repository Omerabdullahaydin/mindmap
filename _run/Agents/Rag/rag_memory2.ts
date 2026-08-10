//rag memory for mindmap
//
// ADVANCED RAG NOTU: Kullanıcının elindeki Azure kaynağında bir EMBEDDING
// deployment'ı yok (sadece bir chat/Responses modeli var) — bu yüzden
// "anlamsal" (semantic/vektör) benzerlik aramasını Azure ÜZERİNDEN
// YAPAMIYORUZ. Bunun temeli, naif kelime sayımından çok daha güçlü bir
// "lexical" (kelime tabanlı) sıralama algoritması olan BM25 (bkz. bm25.ts).
// EK OLARAK, yerelde çalışan Ollama üzerinden (nomic-embed-text modeliyle,
// bkz. Utils/ollamaEmbeddingClient.ts) gerçek embedding de hesaplanıyor ve
// bellek-içi bir vektör indeksinde (bkz. vectorIndex.ts) tutuluyor — BM25'in
// kaçırdığı eş anlamlı/parafraz eşleşmelerini yakalamak için. Ollama o an
// çalışmıyorsa bu katman sessizce atlanır, pipeline sadece BM25 ile devam
// eder (bkz. getEmbeddingRetriever). Sorgu genişletme (query expansion,
// bkz. queryExpansion.ts) ve LLM tabanlı reranking (bkz. rag_tool_citation.ts)
// ile birlikte, bunlar klasik "Advanced RAG" hattının (pre-retrieval ->
// retrieval -> post-retrieval) hibrit bir versiyonunu oluşturuyor.
import { PDFLoader } from "@langchain/community/document_loaders/fs/pdf";
import type { Document } from "@langchain/core/documents";
import { readdir } from "fs/promises";
import path from "path";
import 'dotenv/config';
import { BM25Index } from "./bm25.js";
import { VectorIndex } from "./vectorIndex.js";
import { OllamaEmbeddingClient } from "../Utils/ollamaEmbeddingClient.js";

// tokenize: Bir metni küçük harfe çevirip sadece harf/rakam dizilerine
// (kelimelere) ayırır. "\p{L}\p{N}" Unicode harf/rakam sınıfını kullanır,
// böylece Türkçe karakterler (ç, ğ, ı, ö, ş, ü) doğru şekilde kelime
// sınırlarına dahil olur. 2 karakterden kısa kelimeler (bağlaçlar vb. gibi
// anlam taşımayan kısa kelimeler) elenir. Saf/stateless bir fonksiyon
// olduğu için class'a zorlanmadan burada fonksiyon olarak kalıyor.
export function tokenize(text: string): string[] {
  return (text.toLowerCase().match(/[\p{L}\p{N}]+/gu) || []).filter(w => w.length > 2);
}

// extractPageNumber: Bir paragraf metninin başındaki "[Sayfa N]" etiketinden
// (bkz. reconstructParagraphs) sayfa numarasını çıkarır. Bu proje genelinde
// (rag_tool_citation.ts, mindmap_citation_v2.ts) aynı regex birden fazla
// yerde tekrarlanıyordu — etiket formatının "sahibi" bu dosya olduğu için
// tek, paylaşılan bir yer burada.
export function extractPageNumber(pageContent: string): number {
  const match = pageContent.match(/\[Sayfa (\d+)\]/);
  return match ? parseInt(match[1]) || 0 : 0;
}

// BM25Retriever: MindmapRagStore'un topladığı TÜM paragrafları BM25
// indeksine yükler. invoke(query, k) çağrıldığında, sorguya göre en yüksek
// BM25 skoruna sahip "k" paragrafı döndürür. rag_tool_citation.ts'nin
// beklediği "retriever.invoke(query, k) -> Document[]" arayüzüyle
// (LangChain'in VectorStoreRetriever'ıyla aynı şekilde) uyumludur.
class BM25Retriever {
  private index: BM25Index<Document>;

  constructor(docs: Document[]) {
    this.index = new BM25Index(docs, d => d.pageContent, tokenize);
  }

  async invoke(query: string, k: number): Promise<Document[]> {
    return this.index.search(query, k).map(r => r.item);
  }
}

// EmbeddingRetriever: rag_tool_citation.ts'nin beklediği aynı
// "invoke(query, k) -> Document[]" arayüzünü, BM25 yerine yerel Ollama
// embedding'i + cosine similarity (bkz. vectorIndex.ts) ile sağlar. BM25'in
// yakalayamadığı eş anlamlı/parafraz eşleşmelerini yakalamak için BM25'e EK
// olarak (onun yerine değil) kullanılıyor — bkz. rag_tool_citation.ts'teki
// RRF fusion.
class EmbeddingRetriever {
  private index: VectorIndex<Document>;

  constructor(docs: Document[], vectors: number[][], private embeddingClient: OllamaEmbeddingClient) {
    this.index = new VectorIndex(docs, vectors);
  }

  async invoke(query: string, k: number): Promise<Document[]> {
    const queryVector = await this.embeddingClient.embedOne(query);
    return this.index.search(queryVector, k).map(r => r.item);
  }
}

// ============================================================================
// PARAGRAF YÖNETİCİSİ (module-private, saf fonksiyonlar)
// ----------------------------------------------------------------------------
// reconstructParagraphs: PDF'ten çıkan ham metin genelde satır satır
// KIRILMIŞ olur (PDF'in kendi sayfa/satır düzeninden dolayı, gerçek
// paragraflar bozulmuş gibi görünür). Bu fonksiyon, satırları tekrar
// mantıklı paragraflar haline GERİ BİRLEŞTİRİR ve her paragrafı ayrı bir
// "Document" (LangChain'in standart metin-parçası formatı) olarak döndürür.
//
// Kurallar (satır satır, yukarıdan aşağıya kontrol edilir):
//   - Satır boşsa -> o ana kadar biriken paragrafı kaydet, sıfırla
//   - Satır "- ", "• ", "1. " gibi bir madde işaretiyle başlıyorsa -> yeni paragraf başlat
//   - Satır kısa (<50 karakter) ve büyük harfle başlıyorsa -> muhtemelen bir
//     alt başlık, AYRI bir paragraf (type: "heading") olarak kaydet
//   - Satır tire (-) ile bitip bir sonraki satır küçük harfle başlıyorsa ->
//     kelime PDF'te "tire ile bölünmüş" demektir (örn. "günlerinden bi-" +
//     "rinde" = "birinde"), tireyi kaldırıp birleştir
//   - Bunların hiçbiri değilse -> normal bir devam satırı, mevcut paragrafa ekle
//
// Her paragrafın metnine başına "[Sayfa N]" etiketi ekleniyor (bkz.
// extractPageNumber, bu etiketi geri okuyan paylaşılan yardımcı).
//
// Bu fonksiyon (ve loadPDFs) hem girdisini parametre olarak alıp hem de
// paylaşılan bir state'e dokunmadan yeni bir değer döndürdüğü için (saf),
// MindmapRagStore'un bir metoduna çevrilmiyor — module-private fonksiyon
// olarak kalması "this" plumbing'i eklemeden aynı işi görüyor.
// ============================================================================
function reconstructParagraphs(rawText: string, source: string, pageNumber: number): Document[] {
  const paragraphs: Document<any>[] = [];
  const lines = rawText.split('\n');

  let currentParagraph = ""; // Şu an inşa edilmekte olan paragraf metni
  let paragraphIndex = 0;    // Bu sayfadaki paragrafların sıra numarası (0, 1, 2, ...)

  // flush: o ana kadar biriken paragrafı (varsa) paragraphs listesine ekler
  // ve sıfırlar. Aşağıdaki 3 farklı "yeni paragraf başlat" durumunda
  // (boş satır / madde işareti / kısa başlık) tekrarlanan aynı push+reset
  // bloğunun tek yeri.
  const flush = (type: "paragraph" | "heading" = "paragraph", text: string = currentParagraph) => {
    if (text.trim()) {
      paragraphs.push({
        pageContent: `[Sayfa ${pageNumber}] ${text.trim()}`,
        metadata: {
          source,
          page: pageNumber,
          paragraph_id: paragraphIndex++,
          type
        }
      });
    }
  };

  for (let i = 0; i < lines.length; i++) {
    const line = lines[i];
    const trimmedLine = line.trim();
    const nextLine = lines[i + 1]?.trim() || ""; // Bir sonraki satıra bakmak için (tire birleştirme kuralı için lazım)

    // Boş satır = paragraf sonu
    if (trimmedLine === "") {
      flush();
      currentParagraph = "";
      continue;
    }

    // Madde işareti = yeni paragraf
    if (/^[-•*]\s/.test(trimmedLine) || /^\d+[.)]\s/.test(trimmedLine)) {
      flush();
      currentParagraph = trimmedLine;
      continue;
    }

    // Çok kısa satır (başlık olabilir) = yeni paragraf
    if (trimmedLine.length < 50 && trimmedLine.length > 0 && /^[A-ZÇĞİÖŞÜ]/.test(trimmedLine)) {
      flush();
      currentParagraph = "";
      flush("heading", trimmedLine);
      continue;
    }

    // Tire ile kırılmış kelime birleştir: "meme-\nliler" → "memeliler"
    if (trimmedLine.endsWith("-") && nextLine && /^[a-zçğıöşü]/.test(nextLine)) {
      currentParagraph += trimmedLine.slice(0, -1); // Tire'yi çıkar
      continue;
    }

    // Normal satır = birleştir (nokta olsa bile devam et, sadece boş satırda kes)
    currentParagraph += (currentParagraph ? " " : "") + trimmedLine;
  }

  // Son paragrafı ekle
  flush();

  return paragraphs;
}

// loadPDFs: Verilen dosya yollarındaki (filePaths) TÜM PDF'leri PARALEL
// olarak açar (birbirinden bağımsız dosya I/O'ları), her birinin her
// sayfasını reconstructParagraphs ile paragraflara böler ve hepsini TEK bir
// düz listede toplar.
//
// Bir PDF yüklenirken hata olursa (ör. bozuk dosya), try/catch sayesinde
// SADECE o dosya atlanır (boş dizi döner) — diğer PDF'ler yine de işlenir.
async function loadPDFs(filePaths: string[]): Promise<Document[]> {
  const perFileParagraphs = await Promise.all(filePaths.map(async filePath => {
    try {
      const loader = new PDFLoader(filePath);
      const pages = await loader.load(); // pages: her biri bir sayfanın { pageContent, metadata } objesi

      const paragraphs: Document[] = [];
      for (const page of pages) {
        // PDF Loader'ın sayfa numarasını sakladığı alan sürüme göre değişebiliyor
        // (loc.pageNumber / page / pageNumber). Hangisi doluysa onu kullan.
        let pageNumber = 0;
        if (page.metadata?.loc?.pageNumber !== undefined) {
          pageNumber = page.metadata.loc.pageNumber;
        } else if (page.metadata?.page !== undefined) {
          pageNumber = page.metadata.page;
        } else if (page.metadata?.pageNumber !== undefined) {
          pageNumber = page.metadata.pageNumber;
        }

        paragraphs.push(...reconstructParagraphs(page.pageContent, filePath, pageNumber));
      }
      return paragraphs;

    } catch (error: any) {
      console.error(`✗ ${filePath} yüklenemedi:`, error.message);
      return [];
    }
  }));

  return perFileParagraphs.flat();
}

// ============================================================================
// MindmapRagStore
// ----------------------------------------------------------------------------
// PDF paragraflarını ve embedding vektörlerini tutan, BM25/embedding
// retriever'ları üreten sınıf. ESKİDEN bu state modül seviyesinde
// (let localDocuments/localEmbeddings) tutuluyordu — bu, aynı process
// içinde iki pipeline çalışması üst üste binerse (ör. bir sunucuda eşzamanlı
// istek), birinin diğerinin dokümanlarını sessizce ezmesine yol açabilecek
// bir concurrency riskiydi. Artık her MindmapRagStore instance'ı kendi
// documents/embeddings alanlarını taşıyor — her pipeline çalıştırması kendi
// store instance'ını oluşturduğu sürece (bkz. MindmapPipeline), bu risk
// ortadan kalkıyor.
// ============================================================================
export class MindmapRagStore {
  private documents: Document[] | null = null;
  private embeddings: number[][] | null = null; // documents ile aynı sırada, aynı uzunlukta

  constructor(private embeddingClient: OllamaEmbeddingClient = new OllamaEmbeddingClient()) {}

  // ingest: documentsDir klasöründeki TÜM .pdf dosyalarını okur, paragraflara
  // ayırır, this.documents'i doldurur; ardından (opsiyonel) her paragrafın
  // embedding'ini yerel Ollama'dan hesaplayıp this.embeddings'i doldurur.
  // Ollama o an çalışmıyorsa/model kurulu değilse, sadece uyarı verip
  // embeddings'i null bırakır — pipeline BM25 ile (embedding'siz) çalışmaya
  // devam eder, çökmez (bkz. getEmbeddingRetriever ve rag_tool_citation.ts'teki
  // try/catch).
  async ingest(documentsDir: string = "./documents"): Promise<Document[]> {
    console.log("PDF INGEST (Mindmap - BM25 + yerel embedding hibrit arama)");

    // ESKİDEN tek bir sabit dosya adı vardı: "./documents/FransaSolidarizm.pdf".
    // Kullanıcı kendi PDF'ini yükleyebilsin diye, artık documentsDir
    // klasöründeki TÜM .pdf dosyaları otomatik olarak bulunup işleniyor.
    let pdfPaths: string[] = [];
    try {
      pdfPaths = (await readdir(documentsDir))
        .filter(f => f.toLowerCase().endsWith(".pdf"))
        .map(f => path.join(documentsDir, f));
    } catch (error: any) {
      console.warn(`"${documentsDir}" klasörü okunamadı:`, error.message);
    }

    if (pdfPaths.length === 0) {
      console.warn(`"${documentsDir}" klasöründe PDF bulunamadı.`);
      return [];
    }

    console.log(`İşlenecek PDF(ler): ${pdfPaths.join(", ")}`);

    // PDF'leri yükle ve paragraflara ayır
    const paragraphs = await loadPDFs(pdfPaths);

    if (paragraphs.length === 0) {
      console.warn("Hiçbir PDF paragrafı yüklenemedi!");
      return [];
    }

    console.log(`${paragraphs.length} paragraf çıkarıldı, belleğe ekleniyor...`);
    this.documents = paragraphs;
    console.log(`✓ ${paragraphs.length} paragraf belleğe eklendi.`);

    // ---- Embedding hesapla (yerel Ollama üzerinden) ----
    // Batch'ler birbirinden bağımsız olduğu için PARALEL gönderiliyor.
    try {
      console.log("Embedding'ler hesaplanıyor (yerel Ollama, nomic-embed-text)...");
      const EMBED_BATCH_SIZE = 64; // Tek istekte gönderilecek paragraf sayısı

      const batches: string[][] = [];
      for (let i = 0; i < paragraphs.length; i += EMBED_BATCH_SIZE) {
        batches.push(paragraphs.slice(i, i + EMBED_BATCH_SIZE).map(d => d.pageContent));
      }

      // Promise.all sonuçları, promise'lerin verildiği SIRAYLA döner (bitiş
      // sırasıyla değil), bu yüzden .flat() sonrası embeddings[i] hâlâ
      // paragraphs[i]'ye karşılık geliyor.
      const batchResults = await Promise.all(batches.map(batch => this.embeddingClient.embed(batch)));
      const embeddings = batchResults.flat();

      this.embeddings = embeddings;
      console.log(`✓ ${embeddings.length} embedding hesaplandı.`);
    } catch (error: any) {
      console.warn(
        "⚠️ Embedding hesaplanamadı (Ollama çalışmıyor olabilir), sadece BM25 ile devam edilecek:",
        error.message
      );
      this.embeddings = null;
    }

    return paragraphs;
  }

  // getBM25Retriever: ingest() henüz çalışmadıysa (documents boşsa) fırlatır.
  getBM25Retriever(): BM25Retriever {
    if (!this.documents) {
      throw new Error(
        "Paragraf listesi henüz oluşturulmadı. getBM25Retriever() çağrılmadan önce " +
        "ingest() çalışıp PDF'leri işlemiş olmalı."
      );
    }
    return new BM25Retriever(this.documents);
  }

  // getEmbeddingRetriever: embeddings hesaplanamadıysa (ör. Ollama o an
  // çalışmıyorsa) fırlatır; çağıran taraf (rag_tool_citation.ts) bunu
  // yakalayıp sadece BM25 ile devam edebilir (embedding, BM25'e EK bir
  // katman, zorunlu bağımlılık değil). this.embeddingClient'ı (ingest()'in
  // embedding hesaplarken kullandığı AYNI istemciyi) yeniden kullanır —
  // gereksiz ikinci bir Ollama istemcisi kurmaz.
  getEmbeddingRetriever(): EmbeddingRetriever {
    if (!this.documents || !this.embeddings) {
      throw new Error(
        "Embedding listesi henüz oluşturulmadı (Ollama kullanılamamış olabilir). " +
        "getEmbeddingRetriever() çağrılmadan önce ingest() başarıyla çalışmış olmalı."
      );
    }
    return new EmbeddingRetriever(this.documents, this.embeddings, this.embeddingClient);
  }
}
