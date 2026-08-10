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
// Tüm maddeler gruplar halinde (BATCH_SIZE), PARALEL isteklerle işlenir —
// her madde için ayrı bir LLM çağrısı yapmak hem yavaş hem pahalı olurdu,
// tüm grupları SIRAYLA göndermek de gereksiz yavaş olurdu (gruplar
// birbirinden bağımsız).
// ============================================================================
import { getAzureChatModel } from "../Utils/helper.js";

// extractJson: LLM'in cevabından (bazen ```json ... ``` gibi kod bloğuna
// sarılı gelebilir) JSON objesini çıkarır. Export edilmiş — rag_tool_citation.ts
// da LLM cevaplarından JSON çıkarırken aynı mantığı kullanıyor, ikinci bir
// kopya yazmak yerine bunu paylaşıyor.
export function extractJson(raw: string): string {
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

  const batches: string[][] = [];
  for (let i = 0; i < items.length; i += BATCH_SIZE) {
    batches.push(items.slice(i, i + BATCH_SIZE));
  }

  // Gruplar birbirinden bağımsız olduğu için PARALEL gönderiliyor; her grup
  // kendi try/catch'i içinde, biri başarısız olsa diğerlerini etkilemiyor.
  const batchResults = await Promise.all(batches.map(async (batch, batchIdx) => {
    const listText = batch.map((item, idx) => `${idx}: ${item}`).join('\n');

    const prompt = `You are a search query expansion assistant. For each numbered Turkish text below, list 3-5 ADDITIONAL Turkish search keywords (synonyms, word variants, closely related concepts) that would help find matching passages about the SAME topic in a source document. Do not repeat words already in the text.

ITEMS:
${listText}

Respond with ONLY a single JSON object, no other text, no markdown fences, matching EXACTLY this shape (keys are the item numbers as strings):
{"0":["kelime1","kelime2","kelime3"],"1":["kelime1","kelime2"]}`;

    try {
      const res = await llm.invoke(prompt);
      const parsed = JSON.parse(extractJson(res.content)) as Record<string, string[]>;
      return batch.map((item, idx): [string, string[]] => {
        const keywords = parsed[String(idx)];
        return [item, Array.isArray(keywords) ? keywords : []];
      });
    } catch (error: any) {
      const start = batchIdx * BATCH_SIZE;
      console.warn(`⚠️ Sorgu genişletme başarısız oldu (madde ${start}-${start + batch.length - 1}), bu maddeler genişletme olmadan aranacak:`, error.message);
      return batch.map((item): [string, string[]] => [item, []]);
    }
  }));

  batchResults.flat().forEach(([item, keywords]) => result.set(item, keywords));

  return result;
}
