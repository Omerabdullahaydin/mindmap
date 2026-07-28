// ============================================================================
// BU DOSYA NE İŞE YARAR?
// ----------------------------------------------------------------------------
// Bu, mindmap'teki her MADDEYE (ör. "1.1 Solidarizm kavramının tanımı")
// bir "kaynak" (citation) bulup ekleyen araçtır (LangChain "tool" olarak
// tanımlanmış, MCP DEĞİL — mindmap_visualization_server.ts'teki MCP tool'undan
// farklı, o ayrı bir şey).
//
// GENEL AKIŞ:
//   1. Her madde için, yerel vektör deposunda (bkz. rag_memory2.ts) BENZERLİK
//      ARAMASI yap -> o maddeye anlamca en yakın 3 PDF paragrafını bul
//   2. Bu 3 adayı bir LLM'e gösterip "bunlardan hangisi/hangileri GERÇEKTEN
//      bu maddeyle ilgili?" diye sor (LLM burada bir "seçici/hakem" gibi
//      davranıyor — sadece benzerlik skoruna değil, anlam uygunluğuna bakıyor)
//   3. Seçilen kaynakları mindmap markdown'ına "[Kaynaklar: 0, 2]" gibi
//      rozetler olarak ekle
// ============================================================================
import { z } from "zod"; // Veri şekillerini (schema) tanımlamamızı ve doğrulamamızı sağlayan kütüphane
import { makeMindmapRetriever } from "../Rag/rag_memory2.js";
// AzureChatOpenAI yerine yerel Ollama modeli kullanılıyor (bkz. Utils/helper.ts)
import { getOllamaChatModel } from "../Utils/helper.js";

/**
 * Citation Tool: Finds relevant sources for mindmap items using vector similarity search
 */
export function getCitationTool() {
  // retriever: "bir metin ver, bana anlamca en yakın kayıtları getir" diyebildiğimiz obje.
  // rag_memory2.ts'teki yerel vektör deposundan (MemoryVectorStore) besleniyor.
  const retriever = makeMindmapRetriever();

  return {
    name: "add_citations_to_mindmap",
    description: "Finds and adds citations (source references) to mindmap items using vector similarity search. For each mindmap item, searches the local vector store and returns the top-3 most relevant PDF pages with page numbers.",
    // schema: bu tool'un HANGİ PARAMETRELERİ beklediğini tanımlar (zod ile).
    // Bir LLM bu tool'u "çağırmak" istediğinde, bu şekle uyması gerekir.
    schema: z.object({
      mindmap_markdown: z.string().describe("The markdown mindmap content"),
      items: z.array(z.string()).describe("List of mindmap items to find citations for")
    }),
    // func: tool GERÇEKTEN çağrıldığında çalışacak asıl fonksiyon.
    func: async ({ mindmap_markdown, items }: { mindmap_markdown: string; items: string[] }) => {
      try {
        // ---- ADIM 1: Her madde için PARALEL benzerlik araması yap ----
        // "items.map(item => retriever.invoke(item))" -> her madde için AYNI ANDA
        // bir arama başlatır (hepsi birbirini beklemeden), Promise.all ile
        // hepsinin bitmesini bekleriz. Bu, 30 maddeyi TEK TEK, sırayla
        // aramaktan çok daha hızlıdır.
        const searchPromises = items.map(item => retriever.invoke(item));
        const searchResults = await Promise.all(searchPromises);

        // ---- Sonuçları LLM'e sunmak için düzenle ----
        // Her madde için, bulunan en iyi 3 paragrafı (page) bir arada tutan
        // bir yapı oluşturuyoruz.
        interface MaddeWithPages {
          madde: string;
          pages: { pageIdx: number; page: number; content: string; source: string }[];
        }

        const maddelerWithPages: MaddeWithPages[] = [];

        searchResults.forEach((docs, idx) => {
          const pages = docs.slice(0, 3).map((doc, pageIdx) => {
            let pageNum = 0;
            let sourcePath = "bilinmeyen";

            if (doc.metadata) {
              // Bazı vektör deposu implementasyonları metadata'yı iç içe
              // ("0" anahtarlı bir obje içinde) döndürebiliyor; bu, o
              // durumu da güvenle ele almak için bir kontrol.
              let actualMetadata = doc.metadata;
              if (doc.metadata["0"]) {
                actualMetadata = doc.metadata["0"];
              }
              sourcePath = actualMetadata.source || actualMetadata.Source || "bilinmeyen";
            }

            // Paragrafın metninde "[Sayfa 5]" gibi bir etiket var, oradan sayfa numarasını çıkar.
            const pageMatch = doc.pageContent.match(/\[Sayfa (\d+)\]/);
            if (pageMatch) {
              pageNum = parseInt(pageMatch[1]) || 0;
            }

            return {
              pageIdx,           // Bu maddenin adayları arasında kaçıncı sırada (0, 1, 2)
              page: pageNum,     // PDF'in kaçıncı sayfası
              content: doc.pageContent,
              source: sourcePath
            };
          });

          if (pages.length > 0) {
            maddelerWithPages.push({
              madde: items[idx], // Bu adayların ait olduğu mindmap maddesinin TAM metni
              pages
            });
          }
        });

        // ---- ADIM 2: LLM'e "en alakalı kaynağı seç" diye sor ----
        const llm = getOllamaChatModel();

        // BatchCitationSchema: LLM'den beklediğimiz cevabın TAM ŞEKLİ.
        // "withStructuredOutput" bu şemayı kullanarak LLM'i, düz metin yerine
        // bu şekle uyan bir JSON üretmeye zorlar (mümkün olduğunca).
        const BatchCitationSchema = z.object({
          citations: z.array(z.object({
            item_index: z.number().describe("Item index (0-based)"),
            source_indices: z.array(z.number()).describe("Selected source indices (0, 1, 2)")
          }))
        });

        const structuredLLM = llm.withStructuredOutput(BatchCitationSchema);

        // ------------------------------------------------------------------
        // KÜÇÜK GRUPLAR (BATCH) HALİNDE İŞLE — sağlamlık için eklendi
        // ------------------------------------------------------------------
        // ESKİDEN tüm maddeler (30+ olabiliyor) TEK bir istekte modele
        // gönderiliyordu. Büyük/bulut modellerde (Azure GPT) bu sorun değildi,
        // ama küçük yerel bir modelle (qwen2.5:7b) denendiğinde model şemayı
        // (BatchCitationSchema) takip edemeyip kendi uydurduğu bir JSON formatı
        // döndürdü (ör. {"ITEM 27": [...]}) ve ayrıştırma (parse) tamamen
        // başarısız oldu — SONUÇ: hiç kaynak eklenmedi.
        //
        // Çözüm iki katmanlı:
        //   1) Maddeleri küçük gruplara (BATCH_SIZE) bölüp her grubu AYRI bir
        //      istekte göndermek — daha az madde, modelin şemayı takip etme
        //      ihtimalini artırır (daha basit bir görev).
        //   2) Bir grup yine de şemayı bozarsa, o grubu BİR KEZ daha dener
        //      (RETRY_COUNT); o da başarısız olursa panikleyip tüm işlemi
        //      durdurmak yerine, o gruptaki maddelere en azından EN ALAKALI
        //      (arama sonucunun 0. sırasındaki) kaynağı otomatik atar.
        // Böylece TEK bir grubun başarısızlığı asla TÜM mindmap'in kaynaksız
        // kalmasına yol açmaz.
        // ------------------------------------------------------------------
        const BATCH_SIZE = 5;
        const RETRY_COUNT = 1; // İlk denemeden sonra kaç kez daha denensin (fallback'e düşmeden önce)
        const allCitations: any[] = [];
        let citationId = 0;

        for (let i = 0; i < maddelerWithPages.length; i += BATCH_SIZE) {
          const batch = maddelerWithPages.slice(i, i + BATCH_SIZE);
          const batchNumber = i / BATCH_SIZE + 1;

          // Bu gruptaki maddeleri ve adaylarını, LLM'in okuyacağı düz metne çevir.
          let batchContext = "";
          batch.forEach((maddeData, localIdx) => {
            batchContext += `\n=== ITEM ${localIdx}: "${maddeData.madde}" ===\n`;
            batchContext += `Sources:\n`;
            maddeData.pages.forEach((page, pageIdx) => {
              // İçerik çok uzunsa kısalt (LLM'in isteğini gereksiz şişirmemek için)
              const content = page.content.length > 500 ? page.content.substring(0, 500) + "..." : page.content;
              batchContext += `[${pageIdx}] Page ${page.page}: ${content}\n`;
            });
          });

          const batchPrompt = `You are a citation assistant. Select the MOST RELEVANT source(s) for each item.

FOR EACH ITEM:
- Source indices are 0, 1, 2 (the item's own sources)
- Select at least 1 source
- If no sources are available, return an empty array

ITEMS AND THEIR SOURCES:
${batchContext}

Return citations for each item as JSON matching the required schema:`;

          // applyCitations: LLM'den gelen (şemaya uygun) sonucu allCitations'a ekleyen küçük yardımcı fonksiyon.
          const applyCitations = (batchResult: { citations: { item_index: number; source_indices: number[] }[] }) => {
            batchResult.citations.forEach((citation) => {
              const maddeIdx = citation.item_index; // DİKKAT: bu, grup İÇİNDEKİ yerel indeks (0-based), tüm listedeki değil
              if (maddeIdx >= 0 && maddeIdx < batch.length) {
                const maddeData = batch[maddeIdx];

                citation.source_indices.forEach((sourceIdx: number) => {
                  if (sourceIdx >= 0 && sourceIdx < maddeData.pages.length) {
                    const page = maddeData.pages[sourceIdx];
                    allCitations.push({
                      source_id: citationId++, // Her kaynağa benzersiz, artan bir numara
                      pdf_name: page.source,
                      page_number: page.page,
                      page_id: page.page,
                      content: page.content,
                      madde: maddeData.madde // Bu alan, tıklama modalının çalışması için ZORUNLU (bkz. mindmap_visualization_server.ts'teki bug fix)
                    });
                  }
                });
              }
            });
          };

          // applyFallback: LLM hiçbir şekilde başarılı olamazsa, en azından
          // her maddeye vektör aramasının EN İYİ eşleşmesini (0. sıradaki
          // kaynağı) otomatik ata — "hiç kaynak yok"tan iyidir.
          const applyFallback = () => {
            batch.forEach(maddeData => {
              if (maddeData.pages.length > 0) {
                const page = maddeData.pages[0];
                allCitations.push({
                  source_id: citationId++,
                  pdf_name: page.source,
                  page_number: page.page,
                  page_id: page.page,
                  content: page.content,
                  madde: maddeData.madde
                });
              }
            });
          };

          // Bu grubu en fazla (1 + RETRY_COUNT) kez dene; hepsi başarısız olursa fallback'e düş.
          let succeeded = false;
          for (let attempt = 0; attempt <= RETRY_COUNT && !succeeded; attempt++) {
            try {
              const batchResult = await structuredLLM.invoke(batchPrompt);
              applyCitations(batchResult);
              succeeded = true;
            } catch (batchError: any) {
              if (attempt < RETRY_COUNT) {
                console.warn(`⚠️ Batch ${batchNumber} citation seçimi başarısız (deneme ${attempt + 1}), tekrar deneniyor...`);
              } else {
                console.warn(`⚠️ Batch ${batchNumber} citation seçimi ${RETRY_COUNT + 1} denemede de başarısız oldu (${batchError.message}), bu gruba en alakalı kaynak otomatik ekleniyor.`);
                applyFallback();
              }
            }
          }
        }

        // ---- ADIM 3: Seçilen kaynakları mindmap markdown'ına rozet olarak ekle ----
        // Önce citation'ları "hangi maddeye ait olduğuna" göre gruplandır
        // (bir maddenin birden fazla kaynağı olabilir).
        const citationsByMadde = new Map<string, any[]>();
        allCitations.forEach(citation => {
          if (!citationsByMadde.has(citation.madde)) {
            citationsByMadde.set(citation.madde, []);
          }
          citationsByMadde.get(citation.madde)!.push(citation);
        });

        // Markdown'ı satır satır gezip, her maddenin (bulunabiliyorsa) sonuna
        // "[Kaynaklar: 0, 2]" gibi bir rozet ekle. Başlık ("#") ve boş
        // satırlar olduğu gibi bırakılır.
        const lines = mindmap_markdown.split('\n');
        const newLines: string[] = [];

        for (const line of lines) {
          const trimmed = line.trim();
          if (!trimmed || trimmed.startsWith('#')) {
            newLines.push(line);
          } else if (trimmed.startsWith('-')) {
            const itemText = trimmed.substring(1).trim();
            if (citationsByMadde.has(itemText)) {
              const citations = citationsByMadde.get(itemText)!;
              const sourceIds = citations.map(c => c.source_id).join(', ');
              newLines.push(`${line} [Kaynaklar: ${sourceIds}]`);
            } else {
              newLines.push(line); // Bu maddeye kaynak bulunamadıysa, rozet eklemeden bırak
            }
          } else {
            newLines.push(line);
          }
        }

        const mindmapWithCitations = newLines.join('\n');

        // LangChain tool'ları genelde metin (string) döndürür; biz burada
        // sonucu JSON'a çevirip string olarak döndürüyoruz. Çağıran taraf
        // (mindmap_citation_v2.ts'teki toolNode) bunu JSON.parse ile geri
        // gerçek objeye çevirecek.
        return JSON.stringify({
          success: true,
          mindmap_with_citations: mindmapWithCitations,
          citations: allCitations,
          citation_count: allCitations.length
        });

      } catch (error: any) {
        // Beklenmedik bir hata olursa (ör. vektör deposu hiç oluşturulmamışsa),
        // en azından ORİJİNAL (kaynaksız) mindmap'i geri döndür — kullanıcı
        // hiçbir şey alamamak yerine en azından ham mindmap'i görsün.
        console.error("Citation tool error:", error.message);
        return JSON.stringify({
          success: false,
          error: error.message,
          mindmap_with_citations: mindmap_markdown,
          citations: []
        });
      }
    }
  };
}
