//rag memory for mindmap
//
// ORİJİNAL sürüm Azure OpenAI Embeddings + Azure AI Search kullanıyordu.
// Kullanıcının Azure erişimi olmadığı için burada iki değişiklik yapıldı:
//   1) AzureOpenAIEmbeddings -> OllamaEmbeddings (yerel embedding modeli)
//   2) AzureAISearchVectorStore -> MemoryVectorStore (LangChain'in RAM'de
//      çalışan basit vektör deposu). Azure Search bir "bulut veritabanı"
//      olduğu için process'ler arasında kalıcıydı; MemoryVectorStore ise
//      sadece bu Node.js sürecinin hafızasında yaşar. Bu proje tek bir
//      "workflow.invoke()" çağrısı içinde baştan sona çalıştığı için
//      (docIngest -> map -> reduce -> tools sırayla, aynı süreçte) bu fark
//      etmiyor: docIngestNodeMindmap vektör deposunu doldurur, birkaç adım
//      sonra toolNode -> getCitationTool -> makeMindmapRetriever aynı
//      depodan okur.
import { OllamaEmbeddings } from "@langchain/ollama";
import { MemoryVectorStore } from "langchain/vectorstores/memory";
import { PDFLoader } from "@langchain/community/document_loaders/fs/pdf";
import { VectorStoreRetriever } from "@langchain/core/vectorstores";
import type { Document } from "@langchain/core/documents";
import { readdirSync } from "fs";
import path from "path";
import 'dotenv/config';

//embeddings instance (yerel Ollama embedding modeli)
function getEmbeddingModel() {
  return new OllamaEmbeddings({
    model: process.env.OLLAMA_EMBED_MODEL || "nomic-embed-text"
  });
}

// Azure AI Search'ün yerini alan, bellekte tutulan basit vektör deposu.
// docIngestNodeMindmap tarafından doldurulur, makeMindmapRetriever tarafından
// okunur. Modül seviyesinde (dosya genelinde) tutulduğu için, aynı Node.js
// süreci içindeki her iki fonksiyon da aynı depoyu paylaşır.
let localVectorStore: MemoryVectorStore | null = null;

//retriever oluştur - mindmap için daha fazla sonuç
export function makeMindmapRetriever(): VectorStoreRetriever {
  if (!localVectorStore) {
    throw new Error(
      "Vektör deposu henüz oluşturulmadı. makeMindmapRetriever() çağrılmadan önce " +
      "docIngestNodeMindmap() çalışıp PDF'leri işlemiş olmalı."
    );
  }
  return localVectorStore.asRetriever({
    k: 3 // Her sorguda en benzer 3 paragrafı getir
  });
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
  console.log("PDF INGEST NODE (Mindmap - yerel Ollama embedding)");

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

  console.log(`${paragraphs.length} paragraf çıkarıldı, yerel embedding hesaplanıyor (Ollama)...`);

  const embeddingModel = getEmbeddingModel();

  // Tüm paragrafları embed edip bellekteki vektör deposuna yükle.
  // (Azure AI Search'e "upload" etmenin yerini bu alıyor.)
  localVectorStore = await MemoryVectorStore.fromDocuments(paragraphs, embeddingModel);

  console.log(`✓ ${paragraphs.length} paragraf yerel vektör deposuna eklendi.`);

  return {
    ...state,
    documents: paragraphs
  };
}
