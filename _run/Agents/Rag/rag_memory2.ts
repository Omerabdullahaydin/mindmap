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
// eder (bkz. makeMindmapEmbeddingRetriever). Sorgu genişletme (query
// expansion, bkz. queryExpansion.ts) ve LLM tabanlı reranking (bkz.
// rag_tool_citation.ts) ile birlikte, bunlar klasik "Advanced RAG" hattının
// (pre-retrieval -> retrieval -> post-retrieval) hibrit bir versiyonunu
// oluşturuyor.
import { PDFLoader } from "@langchain/community/document_loaders/fs/pdf";
import type { Document } from "@langchain/core/documents";
import { readdirSync } from "fs";
import path from "path";
import 'dotenv/config';
import { BM25Index } from "./bm25.js";
import { VectorIndex } from "./vectorIndex.js";
import { OllamaEmbeddingClient } from "../Utils/ollamaEmbeddingClient.js";

// tokenize: Bir metni küçük harfe çevirip sadece harf/rakam dizilerine
// (kelimelere) ayırır. "\p{L}\p{N}" Unicode harf/rakam sınıfını kullanır,
// böylece Türkçe karakterler (ç, ğ, ı, ö, ş, ü) doğru şekilde kelime
// sınırlarına dahil olur. 2 karakterden kısa kelimeler (bağlaçlar vb. gibi
// anlam taşımayan kısa kelimeler) elenir.
export function tokenize(text: string): string[] {
  return (text.toLowerCase().match(/[\p{L}\p{N}]+/gu) || []).filter(w => w.length > 2);
}

// BM25Retriever: docIngestNodeMindmap'in topladığı TÜM paragrafları BM25
// indeksine yükler. invoke(query, k) çağrıldığında, sorguya göre en yüksek
// BM25 skoruna sahip "k" paragrafı döndürür. rag_tool_citation.ts'nin
// beklediği "retriever.invoke(query) -> Document[]" arayüzüyle (LangChain'in
// VectorStoreRetriever'ıyla aynı şekilde) uyumludur; k'yi çağrı başına
// (multi-query/fusion için farklı sorgularla) özelleştirebilmek için opsiyonel
// ikinci parametre olarak alır.
class BM25Retriever {
  private index: BM25Index<Document>;

  constructor(docs: Document[], private defaultK: number) {
    this.index = new BM25Index(docs, d => d.pageContent, tokenize);
  }

  async invoke(query: string, k?: number): Promise<Document[]> {
    return this.index.search(query, k ?? this.defaultK).map(r => r.item);
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
  private embeddingClient: OllamaEmbeddingClient;

  constructor(docs: Document[], vectors: number[][], private defaultK: number) {
    this.index = new VectorIndex(docs, vectors);
    this.embeddingClient = new OllamaEmbeddingClient();
  }

  async invoke(query: string, k?: number): Promise<Document[]> {
    const queryVector = await this.embeddingClient.embedOne(query);
    return this.index.search(queryVector, k ?? this.defaultK).map(r => r.item);
  }
}

// Azure AI Search'ün yerini alan, bellekte tutulan basit paragraf listesi
// (ve buna paralel embedding vektörleri). docIngestNodeMindmap tarafından
// doldurulur, makeMindmapRetriever/makeMindmapEmbeddingRetriever tarafından
// okunur. Modül seviyesinde (dosya genelinde) tutulduğu için, aynı Node.js
// süreci içindeki her iki fonksiyon da aynı listeyi paylaşır.
let localDocuments: Document[] | null = null;
let localEmbeddings: number[][] | null = null; // localDocuments ile aynı sırada, aynı uzunlukta

//retriever oluştur - mindmap için daha fazla sonuç
export function makeMindmapRetriever(): BM25Retriever {
  if (!localDocuments) {
    throw new Error(
      "Paragraf listesi henüz oluşturulmadı. makeMindmapRetriever() çağrılmadan önce " +
      "docIngestNodeMindmap() çalışıp PDF'leri işlemiş olmalı."
    );
  }
  return new BM25Retriever(localDocuments, 8); // Advanced RAG: LLM'in reranking yapabilmesi için daha GENİŞ bir aday havuzu (3 değil, 8)
}

// embedding retriever oluştur - localEmbeddings hesaplanamadıysa (ör. Ollama
// o an çalışmıyorsa) fırlatır; çağıran taraf (rag_tool_citation.ts) bunu
// yakalayıp sadece BM25 ile devam edebilir (embedding, BM25'e EK bir katman,
// zorunlu bağımlılık değil).
export function makeMindmapEmbeddingRetriever(): EmbeddingRetriever {
  if (!localDocuments || !localEmbeddings) {
    throw new Error(
      "Embedding listesi henüz oluşturulmadı (Ollama kullanılamamış olabilir). " +
      "makeMindmapEmbeddingRetriever() çağrılmadan önce docIngestNodeMindmap() " +
      "başarıyla çalışmış olmalı."
    );
  }
  return new EmbeddingRetriever(localDocuments, localEmbeddings, 8);
}


// ============================================================================
// PARAGRAF YÖNETİCİSİ
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
// Her paragrafın metnine başına "[Sayfa N]" etiketi ekleniyor — bu etiket,
// projenin başka yerlerinde (mapNode, expandItemsNode) sayfa numarasını
// metinden geri çıkarmak için kullanılıyor (regex ile: /\[Sayfa (\d+)\]/).
// ============================================================================
function reconstructParagraphs(rawText: string, source: string, pageNumber: number): Document[] {
  const paragraphs: Document<any>[] = [];
  const lines = rawText.split('\n');

  let currentParagraph = ""; // Şu an inşa edilmekte olan paragraf metni
  let paragraphIndex = 0;    // Bu sayfadaki paragrafların sıra numarası (0, 1, 2, ...)

  for (let i = 0; i < lines.length; i++) {
    const line = lines[i];
    const trimmedLine = line.trim();
    const nextLine = lines[i + 1]?.trim() || ""; // Bir sonraki satıra bakmak için (tire birleştirme kuralı için lazım)

    // Boş satır = paragraf sonu
    if (trimmedLine === "") {
      if (currentParagraph.trim()) {
        paragraphs.push({
          pageContent: `[Sayfa ${pageNumber}] ${currentParagraph.trim()}`,
          metadata: {
            source,
            page: pageNumber,
            paragraph_id: paragraphIndex++,
            type: "paragraph"
          }
        });
        currentParagraph = "";
      }
      continue;
    }

    // Madde işareti = yeni paragraf
    if (/^[-•*]\s/.test(trimmedLine) || /^\d+[.)]\s/.test(trimmedLine)) {
      if (currentParagraph.trim()) {
        paragraphs.push({
          pageContent: `[Sayfa ${pageNumber}] ${currentParagraph.trim()}`,
          metadata: {
            source,
            page: pageNumber,
            paragraph_id: paragraphIndex++,
            type: "paragraph"
          }
        });
        currentParagraph = "";
      }
      currentParagraph = trimmedLine;
      continue;
    }

    // Çok kısa satır (başlık olabilir) = yeni paragraf
    if (trimmedLine.length < 50 && trimmedLine.length > 0 && /^[A-ZÇĞİÖŞÜ]/.test(trimmedLine)) {
      if (currentParagraph.trim()) {
        paragraphs.push({
          pageContent: `[Sayfa ${pageNumber}] ${currentParagraph.trim()}`,
          metadata: {
            source,
            page: pageNumber,
            paragraph_id: paragraphIndex++,
            type: "paragraph"
          }
        });
        currentParagraph = "";
      }
      paragraphs.push({
        pageContent: `[Sayfa ${pageNumber}] ${trimmedLine}`,
        metadata: {
          source,
          page: pageNumber,
          paragraph_id: paragraphIndex++,
          type: "heading"
        }
      });
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
  if (currentParagraph.trim()) {
    paragraphs.push({
      pageContent: `[Sayfa ${pageNumber}] ${currentParagraph.trim()}`,
      metadata: {
        source,
        page: pageNumber,
        paragraph_id: paragraphIndex++,
        type: "paragraph"
      }
    });
  }

  return paragraphs;
}

// loadPDFs: Verilen dosya yollarındaki (filePaths) TÜM PDF'leri sırayla açar,
// her birinin her sayfasını reconstructParagraphs ile paragraflara böler ve
// hepsini TEK bir düz listede (allParagraphs) toplar.
//
// Bir PDF yüklenirken hata olursa (ör. bozuk dosya), try/catch sayesinde
// SADECE o dosya atlanır — diğer PDF'ler yine de işlenmeye devam eder.
async function loadPDFs(filePaths: string[]): Promise<Document[]> {
  const allParagraphs: Document[] = [];

  for (const filePath of filePaths) {
    try {
      const loader = new PDFLoader(filePath);
      const pages = await loader.load(); // pages: her biri bir sayfanın { pageContent, metadata } objesi

      // Her sayfayı paragraflara ayır
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

        const paragraphs = reconstructParagraphs(page.pageContent, filePath, pageNumber);
        allParagraphs.push(...paragraphs);
      }

    } catch (error: any) {
      console.error(`✗ ${filePath} yüklenemedi:`, error.message);
    }
  }
  return allParagraphs;
}

// Doküman ingest node - Mindmap için
export async function docIngestNodeMindmap(state: any) {
  console.log("PDF INGEST NODE (Mindmap - BM25 + yerel embedding hibrit arama)");

  // ESKİDEN tek bir sabit dosya adı vardı: "./documents/FransaSolidarizm.pdf".
  // Kullanıcı kendi PDF'ini yükleyebilsin diye, artık documents/ klasöründeki
  // TÜM .pdf dosyaları otomatik olarak bulunup işleniyor.
  const documentsDir = "./documents";
  let pdfPaths: string[] = [];
  try {
    pdfPaths = readdirSync(documentsDir)
      .filter(f => f.toLowerCase().endsWith(".pdf"))
      .map(f => path.join(documentsDir, f));
  } catch (error: any) {
    console.warn(`"${documentsDir}" klasörü okunamadı:`, error.message);
  }

  if (pdfPaths.length === 0) {
    console.warn(`"${documentsDir}" klasöründe PDF bulunamadı.`);
    return state;
  }

  console.log(`İşlenecek PDF(ler): ${pdfPaths.join(", ")}`);

  // PDF'leri yükle ve paragraflara ayır
  const paragraphs = await loadPDFs(pdfPaths);

  if (paragraphs.length === 0) {
    console.warn("Hiçbir PDF paragrafı yüklenemedi!");
    return state;
  }

  console.log(`${paragraphs.length} paragraf çıkarıldı, bellekteki listeye ekleniyor...`);

  // Paragrafları modül seviyesindeki listeye yükle (bkz. makeMindmapRetriever).
  localDocuments = paragraphs;

  console.log(`✓ ${paragraphs.length} paragraf belleğe eklendi.`);

  // ---- Embedding hesapla (yerel Ollama üzerinden) ----
  // Bu adım OPSİYONEL: Ollama o an çalışmıyorsa ya da model kurulu değilse,
  // sadece uyarı verip localEmbeddings'i null bırakıyoruz — pipeline BM25 ile
  // (embedding'siz) çalışmaya devam eder, çökmez (bkz.
  // makeMindmapEmbeddingRetriever ve rag_tool_citation.ts'teki try/catch).
  try {
    console.log("Embedding'ler hesaplanıyor (yerel Ollama, nomic-embed-text)...");
    const embeddingClient = new OllamaEmbeddingClient();
    const EMBED_BATCH_SIZE = 64; // Tek istekte gönderilecek paragraf sayısı
    const embeddings: number[][] = [];

    for (let i = 0; i < paragraphs.length; i += EMBED_BATCH_SIZE) {
      const batch = paragraphs.slice(i, i + EMBED_BATCH_SIZE);
      const batchVectors = await embeddingClient.embed(batch.map(d => d.pageContent));
      embeddings.push(...batchVectors);
    }

    localEmbeddings = embeddings;
    console.log(`✓ ${embeddings.length} embedding hesaplandı.`);
  } catch (error: any) {
    console.warn(
      "⚠️ Embedding hesaplanamadı (Ollama çalışmıyor olabilir), sadece BM25 ile devam edilecek:",
      error.message
    );
    localEmbeddings = null;
  }

  return {
    ...state,
    documents: paragraphs
  };
}
