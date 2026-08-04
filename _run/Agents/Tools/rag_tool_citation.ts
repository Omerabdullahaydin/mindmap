// ============================================================================
// BU DOSYA NE İŞE YARAR?
// ----------------------------------------------------------------------------
// Bu, mindmap'teki her MADDEYE (ör. "1.1 Solidarizm kavramının tanımı")
// bir "kaynak" (citation) bulup ekleyen araçtır (LangChain "tool" olarak
// tanımlanmış, MCP DEĞİL — mindmap_visualization_server.ts'teki MCP tool'undan
// farklı, o ayrı bir şey).
//
// ADVANCED RAG HATTI (HİBRİT: BM25 + yerel Ollama embedding):
//   1. PRE-RETRIEVAL — Sorgu genişletme: Her madde için LLM'den ek arama
//      terimleri (eş anlamlı/ilgili kavramlar) iste (bkz. queryExpansion.ts)
//   2. RETRIEVAL — Multi-query + Hibrit Fusion: Her madde için HEM orijinal
//      metinle HEM genişletilmiş sorguyla, HEM BM25 (kelime tabanlı, bkz.
//      rag_memory2.ts) HEM embedding (anlam tabanlı, yerel Ollama üzerinden —
//      Azure kaynağında embedding deployment'ı olmadığı için) araması yap;
//      sonuçları RRF (Reciprocal Rank Fusion, bkz. fusion.ts) ile birleştir ->
//      geniş bir aday havuzu (en fazla 8 paragraf) elde et. Ollama
//      çalışmıyorsa embedding katmanı sessizce atlanır, sadece BM25 kullanılır.
//   3. POST-RETRIEVAL — LLM reranking: Bu geniş aday havuzunu bir LLM'e
//      gösterip "bunlardan hangisi/hangileri GERÇEKTEN bu maddeyle ilgili?"
//      diye sor (LLM burada bir "seçici/hakem" gibi davranıyor — sadece BM25
//      skoruna değil, anlam uygunluğuna bakıyor)
//   4. Seçilen kaynakları mindmap markdown'ına "[Kaynaklar: 0, 2]" gibi
//      rozetler olarak ekle
// ============================================================================
import { z } from "zod"; // Veri şekillerini (schema) tanımlamamızı ve doğrulamamızı sağlayan kütüphane
import type { Document } from "@langchain/core/documents";
import { makeMindmapRetriever, makeMindmapEmbeddingRetriever } from "../Rag/rag_memory2.js";
import { expandQueriesBatch } from "../Rag/queryExpansion.js";
import { reciprocalRankFusion } from "../Rag/fusion.js";
import { getAzureChatModel } from "../Utils/helper.js";

const CANDIDATE_POOL_SIZE = 8; // Reranker'a sunulacak, fusion sonrası en fazla aday sayısı
const PER_QUERY_K = 5;         // Her TEK sorgu (orijinal ya da genişletilmiş) için BM25'ten kaç sonuç istensin

/**
 * Citation Tool: Finds relevant sources for mindmap items using an
 * embedding-free Advanced RAG pipeline (query expansion + multi-query BM25
 * fusion + LLM reranking).
 */
export function getCitationTool() {
  // retriever: "bir metin ver, bana en alakalı (BM25) kayıtları getir" diyebildiğimiz obje.
  // rag_memory2.ts'teki BM25 indeksinden besleniyor.
  const retriever = makeMindmapRetriever();

  // embeddingRetriever: BM25'in kaçırdığı eş anlamlı/parafraz eşleşmelerini
  // yakalamak için EK bir arama katmanı (yerel Ollama embedding'i, bkz.
  // rag_memory2.ts). Ollama o an çalışmıyorsa ya da ingest sırasında
  // embedding hesaplanamadıysa, burada null kalır ve aşağıda sadece BM25
  // sonuçlarıyla devam edilir — embedding zorunlu bir bağımlılık değil.
  let embeddingRetriever: ReturnType<typeof makeMindmapEmbeddingRetriever> | null = null;
  try {
    embeddingRetriever = makeMindmapEmbeddingRetriever();
  } catch (error: any) {
    console.warn("⚠️ Embedding retriever kullanılamıyor, sadece BM25 ile devam edilecek:", error.message);
  }

  return {
    name: "add_citations_to_mindmap",
    description: "Finds and adds citations (source references) to mindmap items using an embedding-free Advanced RAG pipeline (query expansion + multi-query BM25 fusion + LLM reranking). For each mindmap item, returns the most relevant PDF pages with page numbers.",
    // schema: bu tool'un HANGİ PARAMETRELERİ beklediğini tanımlar (zod ile).
    // Bir LLM bu tool'u "çağırmak" istediğinde, bu şekle uyması gerekir.
    schema: z.object({
      mindmap_markdown: z.string().describe("The markdown mindmap content"),
      items: z.array(z.string()).describe("List of mindmap items to find citations for")
    }),
    // func: tool GERÇEKTEN çağrıldığında çalışacak asıl fonksiyon.
    func: async ({ mindmap_markdown, items }: { mindmap_markdown: string; items: string[] }) => {
      try {
        // ---- ADIM 1 (pre-retrieval): Tüm maddeler için sorgu genişletme ----
        console.log("🔎 Sorgular genişletiliyor (query expansion)...");
        const expansions = await expandQueriesBatch(items);

        // ---- ADIM 2 (retrieval + fusion): Her madde için multi-query, HİBRİT (BM25 + embedding) arama ----
        // Her madde için BM25 (kelime tabanlı, kesin eşleşme) VE, varsa,
        // embedding (anlam tabanlı, parafraz/eş anlamlı eşleşme) aramaları
        // AYRI AYRI yapılır — her ikisi de HEM orijinal metinle HEM
        // genişletilmiş sorguyla (madde + LLM'in ürettiği ek terimler)
        // çalıştırılır. Sonuçta 2-4 ayrı sıralı liste elde edilir; bunlar RRF
        // (Reciprocal Rank Fusion, bkz. fusion.ts) ile TEK bir sıralı aday
        // havuzunda birleştirilir — ham skorları değil, her listedeki SIRAyı
        // esas alır (BM25 skoru ile cosine similarity karşılaştırılabilir
        // birimlerde değil).
        const searchPromises = items.map(async item => {
          const expansionWords = expansions.get(item) || [];
          const expandedQuery = expansionWords.length > 0 ? `${item} ${expansionWords.join(' ')}` : null;

          const [bm25Original, bm25Expanded] = await Promise.all([
            retriever.invoke(item, PER_QUERY_K),
            expandedQuery ? retriever.invoke(expandedQuery, PER_QUERY_K) : Promise.resolve([] as Document[])
          ]);

          const resultLists: Document[][] = [bm25Original, bm25Expanded];

          if (embeddingRetriever) {
            try {
              const [embedOriginal, embedExpanded] = await Promise.all([
                embeddingRetriever.invoke(item, PER_QUERY_K),
                expandedQuery ? embeddingRetriever.invoke(expandedQuery, PER_QUERY_K) : Promise.resolve([] as Document[])
              ]);
              resultLists.push(embedOriginal, embedExpanded);
            } catch (error: any) {
              // Bu maddeye özel embedding araması başarısız olursa (ör. Ollama
              // o an geçici bir hata verdiyse), sadece BM25 sonuçlarıyla devam
              // et — tüm işlemi durdurma.
              console.warn(`⚠️ Embedding araması başarısız oldu ("${item.slice(0, 40)}..."):`, error.message);
            }
          }

          const fused = reciprocalRankFusion(resultLists);
          return fused.slice(0, CANDIDATE_POOL_SIZE);
        });
        const searchResults = await Promise.all(searchPromises);

        // ---- Sonuçları LLM'e sunmak için düzenle ----
        // Her madde için, fusion sonrası bulunan adayları (page) bir arada
        // tutan bir yapı oluşturuyoruz.
        interface MaddeWithPages {
          madde: string;
          pages: { pageIdx: number; page: number; content: string; source: string }[];
        }

        const maddelerWithPages: MaddeWithPages[] = [];

        searchResults.forEach((docs, idx) => {
          const pages = docs.map((doc, pageIdx) => {
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

        // ---- ADIM 3 (post-retrieval): LLM'e "en alakalı kaynağı seç" diye sor ----
        // (reranking — geniş fusion havuzundan gerçekten alakalı olanları eler)
        const llm = getAzureChatModel();

        // BatchCitationSchema: LLM'den beklediğimiz cevabın TAM ŞEKLİ.
        // Kullandığımız Azure Responses istemcisinde LangChain'in
        // "withStructuredOutput" (şema zorlayan) özelliği YOK — bunun yerine
        // modele "SADECE bu şekle uyan JSON döndür" diye talimat veriyoruz
        // (bkz. batchPrompt) ve cevabı aşağıda zod ile DOĞRULUYORUZ. Zod
        // doğrulaması başarısız olursa (model şemayı bozarsa), bu zaten var
        // olan retry/fallback mekanizmasını tetikler (aşağıda).
        const BatchCitationSchema = z.object({
          citations: z.array(z.object({
            item_index: z.number().describe("Item index (0-based)"),
            source_indices: z.array(z.number()).describe(`Selected source indices (0 to ${CANDIDATE_POOL_SIZE - 1})`)
          }))
        });

        // parseCitationResponse: LLM'in düz metin cevabından JSON'ı çıkarır
        // (bazen ```json ... ``` gibi kod bloğuna sarılı gelebilir) ve
        // BatchCitationSchema'ya göre doğrular. Uymazsa fırlatır (throw) —
        // bu, çağıran taraftaki retry/fallback döngüsünü tetikler.
        const parseCitationResponse = (raw: string): z.infer<typeof BatchCitationSchema> => {
          const start = raw.indexOf('{');
          const end = raw.lastIndexOf('}');
          if (start === -1 || end === -1 || end < start) {
            throw new Error("Cevapta JSON bulunamadı");
          }
          const jsonText = raw.slice(start, end + 1);
          return BatchCitationSchema.parse(JSON.parse(jsonText));
        };

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

          const batchPrompt = `You are a citation assistant. Each item below has a pool of CANDIDATE sources (found via keyword search, not all are necessarily relevant). Select ONLY the source(s) that are ACTUALLY relevant to each item.

FOR EACH ITEM:
- Source indices refer to the item's own numbered candidate list (starts at 0)
- Select 1-3 truly relevant sources; ignore candidates that are off-topic even if they share some keywords
- If no candidates are actually relevant, return an empty array

ITEMS AND THEIR CANDIDATE SOURCES:
${batchContext}

Respond with ONLY a single JSON object, no other text, no markdown code fences, matching EXACTLY this shape:
{"citations":[{"item_index":0,"source_indices":[0,1]}]}`;

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
              const rawResult = await llm.invoke(batchPrompt);
              const batchResult = parseCitationResponse(rawResult.content);
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
