// ============================================================================
// SORGU GENİŞLETME (Query Expansion) — Advanced RAG'in "pre-retrieval"
// (arama öncesi) aşamasındaki klasik bir teknik.
// ----------------------------------------------------------------------------
// PROBLEM: BM25 gibi kelime-tabanlı bir arama, sorguda GEÇMEYEN ama anlamca
// ilgili kelimeleri (eş anlamlılar, farklı çekim/yazım biçimleri, ilgili
// kavramlar) yakalayamaz. Örnek: mindmap maddesi "öğrencilerin motivasyonu"
// dese, PDF paragrafı "katılımcıların isteklendirilmesi" dese, BM25 bu ikisi
// arasında HİÇ ortak kelime bulamaz ve eşleşmeyi kaçırır.
//
// ÇÖZÜM: Aramadan ÖNCE, LLM'e her maddeyi göstererek "bu konuyla ilgili
// birkaç ek arama terimi/eş anlamlı üret" diye soruyoruz. Bu ek terimler,
// orijinal madde metniyle BİRLEŞTİRİLİP genişletilmiş bir sorgu oluşturur;
// bu genişletilmiş sorgu BM25'e verilince, tek başına orijinal metnin
// bulamayacağı paragrafları da yakalama ihtimali artar.
//
// Tüm maddeler TEK bir (ya da birkaç, gruplar halinde) istekte işlenir —
// her madde için ayrı bir LLM çağrısı yapmak hem yavaş hem pahalı olurdu.
// ============================================================================
import { getAzureChatModel } from "../Utils/helper.js";

// extractJson: LLM'in cevabından (bazen ```json ... ``` gibi kod bloğuna
// sarılı gelebilir) JSON objesini çıkarır.
function extractJson(raw: string): string {
  const start = raw.indexOf('{');
  const end = raw.lastIndexOf('}');
  if (start === -1 || end === -1 || end < start) {
    throw new Error("Cevapta JSON bulunamadı");
  }
  return raw.slice(start, end + 1);
}

// expandQueriesBatch: Verilen madde listesindeki HER madde için 3-5 ek
// arama terimi üretir. Sonuç, "madde metni -> [ek terimler]" şeklinde bir
// Map'tir. Bir grup başarısız olursa (LLM şemayı bozarsa), o gruptaki
// maddeler için boş dizi döner (genişletme olmadan devam edilir) — arama
// tamamen çökmez, sadece o maddeler için genişletme faydası kaybolur.
export async function expandQueriesBatch(items: string[]): Promise<Map<string, string[]>> {
  const result = new Map<string, string[]>();
  if (items.length === 0) return result;

  const llm = getAzureChatModel({ temperature: 0.2, maxTokens: 1500 });
  const BATCH_SIZE = 12;

  for (let i = 0; i < items.length; i += BATCH_SIZE) {
    const batch = items.slice(i, i + BATCH_SIZE);
    const listText = batch.map((item, idx) => `${idx}: ${item}`).join('\n');

    const prompt = `You are a search query expansion assistant. For each numbered Turkish text below, list 3-5 ADDITIONAL Turkish search keywords (synonyms, word variants, closely related concepts) that would help find matching passages about the SAME topic in a source document. Do not repeat words already in the text.

ITEMS:
${listText}

Respond with ONLY a single JSON object, no other text, no markdown fences, matching EXACTLY this shape (keys are the item numbers as strings):
{"0":["kelime1","kelime2","kelime3"],"1":["kelime1","kelime2"]}`;

    try {
      const res = await llm.invoke(prompt);
      const parsed = JSON.parse(extractJson(res.content)) as Record<string, string[]>;
      batch.forEach((item, idx) => {
        const keywords = parsed[String(idx)];
        result.set(item, Array.isArray(keywords) ? keywords : []);
      });
    } catch (error: any) {
      console.warn(`⚠️ Sorgu genişletme başarısız oldu (madde ${i}-${i + batch.length - 1}), bu maddeler genişletme olmadan aranacak:`, error.message);
      batch.forEach(item => result.set(item, []));
    }
  }

  return result;
}
