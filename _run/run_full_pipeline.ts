// ============================================================================
// BU DOSYA NE İŞE YARAR?
// ----------------------------------------------------------------------------
// Bu, mindmap_citation_v2.ts'te tanımlanan GERÇEK, orijinal tasarımdaki çok
// adımlı LangGraph pipeline'ını (docIngest -> map -> reduce -> tools ->
// expandItems) uçtan uca çalıştıran KÜÇÜK bir başlatma (driver) script'idir.
// "npm start" komutu bu dosyayı çalıştırır.
//
// Artık Azure yerine tamamen yerel Ollama modelleri kullanıyor — bu dosyanın
// kendisi hiçbir LLM çağrısı yapmıyor, sadece pipeline'ı başlatıp sonucu
// okunaklı bir şekilde terminale yazdırıyor.
// ============================================================================
import { mindmapCitationAgentV2 } from "./Agents/mindmap_citation_v2.js";

// Komut satırından bir konu verilmişse onu kullan (ör: npm start -- "Solidarizm"),
// verilmemişse varsayılan olarak documents/ klasöründeki PDF'in konusuna uygun bir başlık kullan.
const topic = process.argv[2] || "Bilgisayar oyunlarının akademik başarıya etkisi";

console.log(`Pipeline başlatılıyor. Konu: "${topic}"`);

// .invoke({topic}): LangGraph'a "bu başlangıç durumuyla (state) baştan sona
// çalış" der. Bu TEK satır, arka planda sırasıyla şunları tetikler:
//   docIngestNodeMindmap -> mapNode -> reduceNode -> toolNode -> expandItemsNode
// Her adımın çıktısı otomatik olarak bir sonrakine "state" üzerinden aktarılır
// (bkz. mindmap_citation_v2.ts'teki MindmapStateAnnotation).
const result = await mindmapCitationAgentV2.invoke({ topic });

// result, TÜM adımların birikmiş son durumunu (state'in son hâlini) içerir.
// Burada sadece en çok merak edilen alanları özet olarak yazdırıyoruz.
console.log("\n==== SONUÇ ====");
console.log("Özet sayısı:", result.summaries?.length ?? 0);               // map aşamasında kaç parça özetlendi
console.log("Citation sayısı:", result.citations?.length ?? 0);           // kaç kaynak (sayfa referansı) bulundu
console.log("HTML dosyası:", result.html_file_path);                      // görsel mindmap'in diskteki yolu
console.log("Genişletilmiş kategori sayısı:", result.expanded_items?.length ?? 0); // kaç kategori için anlatım metni üretildi
