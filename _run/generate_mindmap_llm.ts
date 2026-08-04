// ============================================================================
// BU DOSYA NE İŞE YARAR? (Baştan sona, adım adım)
// ----------------------------------------------------------------------------
// Bu, PDF'ten GERÇEK bir yapay zeka (LLM) kullanarak mindmap üreten script'tir.
// Azure OpenAI'nin bir "chat" (Responses API) modelini kullanır — PDF metnini
// özetlemek ve mindmap'in kategori/madde yapısını yazmak için.
//
// KAYNAK EŞLEŞTİRME NASIL YAPILIYOR? (Advanced RAG, embedding OLMADAN)
// Kullanıcının Azure kaynağında bir EMBEDDING (anlamsal arama) modeli YOK,
// sadece bir chat modeli var. Bu yüzden her mindmap maddesini PDF
// paragraflarıyla eşleştirirken üç adımlı, embedding gerektirmeyen bir
// "Advanced RAG" hattı kullanıyoruz (bkz. Agents/Rag/bm25.ts ve
// Agents/Rag/queryExpansion.ts, bu script onları doğrudan paylaşıyor):
//   1) Pre-retrieval: her madde için LLM'den ek arama terimleri (eş anlamlı/
//      ilgili kavramlar) iste (query expansion)
//   2) Retrieval + fusion: hem orijinal madde metniyle hem genişletilmiş
//      sorguyla ayrı ayrı BM25 araması yap, sonuçları birleştir
//   3) Post-retrieval: LLM'e bu aday havuzunu gösterip GERÇEKTEN alakalı
//      olanları seçtir (reranking) — bkz. Adım 4'ün içindeki matchCitations
//
// GENEL AKIŞ:
//   1. PDF'i oku, sayfa sayfa metne çevir, paragrafları yeniden birleştir
//   2. MAP AŞAMASI: Sayfaları 8'erli gruplara böl, her grubu LLM'e gönderip
//      Türkçe bir özet çıkart (uzun bir PDF'i tek seferde LLM'e vermek yerine
//      parça parça özetlemek, "map-reduce" denen klasik bir tekniktir)
//   3. REDUCE AŞAMASI: Tüm özetleri birleştirip LLM'e tekrar gönder, bu sefer
//      "bana numaralı, kategorili bir mindmap markdown'ı yaz" diye iste
//   4. KAYNAK EŞLEŞTİRME: Yukarıdaki Advanced RAG hattıyla her maddeyi PDF
//      paragraflarıyla eşleştir (tıklanınca "kaynak gösteren" citation
//      özelliğini besler)
//   5. Sonucu, mindmap_visualization_server.ts'teki GERÇEK HTML üretme
//      fonksiyonuna verip görsel mindmap dosyasını oluştur
// ============================================================================

import { PDFLoader } from "@langchain/community/document_loaders/fs/pdf"; // PDF'i sayfa sayfa metne çevirir
import { AzureResponsesChatModel } from "./Agents/Utils/azureResponsesClient.js"; // Azure OpenAI (Responses API) chat modeline bağlanmak için minimal istemci
import { BM25Index } from "./Agents/Rag/bm25.js";
import { expandQueriesBatch } from "./Agents/Rag/queryExpansion.js";
import { createLiveMindmapHTML } from "./mcp/mindmap_visualization_server.js"; // Gerçek HTML mindmap üretici
import path from "path";
import { readdirSync } from "fs";
import "dotenv/config";

// tokenize: bkz. Agents/Rag/rag_memory2.ts'teki aynı fonksiyonun ayrıntılı açıklaması.
function tokenize(text: string): string[] {
  return (text.toLowerCase().match(/[\p{L}\p{N}]+/gu) || []).filter(w => w.length > 2);
}

// ---- Ayarlar (istenirse ortam değişkeniyle değiştirilebilir) ----
const PDF_DIR = "./documents";

const PAGES_PER_CHUNK = 8;       // Map aşamasında kaç sayfa bir arada özetlensin
const TOTAL_MINUTES = 12;        // Mindmap'teki (MM:SS) süre etiketlerinin toplamı (sadece görsel/kozmetik)

// ----------------------------------------------------------------------------
// pageNumberOf: PDFLoader'ın sayfa numarasını sakladığı alan sürüme göre
// değişebiliyor (loc.pageNumber / page / pageNumber). Hangisi doluysa onu al.
// ----------------------------------------------------------------------------
function pageNumberOf(page: any): number {
  return page.metadata?.loc?.pageNumber ?? page.metadata?.page ?? page.metadata?.pageNumber ?? 0;
}

// ----------------------------------------------------------------------------
// reconstructParagraphs: PDF'ten çıkan ham metin genelde satır satır
// kırılmış olur (PDF'in kendi satır kaydırmasından dolayı), bu da gerçek
// paragrafları bozar. Bu fonksiyon, satırları tekrar mantıklı paragraflar
// haline GERİ BİRLEŞTİRİR. (Bu, orijinal projedeki rag_memory2.ts dosyasının
// "reconstructParagraphs" fonksiyonuyla AYNI mantığı kullanır — sadece o
// dosya Azure'a bağımlı olduğu için, aynı sezgisel kuralları burada,
// bağımsız bir kopya olarak yeniden yazdık.)
//
// Kurallar (satır satır, yukarıdan aşağıya kontrol edilir):
//   - Satır boşsa -> o ana kadar biriken paragrafı kaydet, sıfırla
//   - Satır "- ", "• ", "1. " gibi bir madde işaretiyle başlıyorsa -> yeni paragraf başlat
//   - Satır kısa (<50 karakter) ve büyük harfle başlıyorsa -> muhtemelen bir
//     alt başlık, ayrı bir paragraf olarak kaydet
//   - Satır tire (-) ile bitip bir sonraki satır küçük harfle başlıyorsa ->
//     kelime PDF'te "tire ile bölünmüş" demektir (örn. "günlerinden bi-" +
//     "rinde" = "birinde"), tireyi kaldırıp birleştir
//   - Bunların hiçbiri değilse -> normal bir devam satırı, mevcut paragrafa ekle
// ----------------------------------------------------------------------------
function reconstructParagraphs(rawText: string, pageNumber: number): { text: string; page: number }[] {
  const paragraphs: { text: string; page: number }[] = [];
  const lines = rawText.split(/\r?\n/);
  let current = ""; // Şu an inşa edilmekte olan paragraf metni

  for (let i = 0; i < lines.length; i++) {
    const trimmed = lines[i].trim();
    const next = lines[i + 1]?.trim() || ""; // Bir sonraki satıra bakmak için (tire birleştirme kuralı için lazım)

    if (trimmed === "") {
      if (current.trim()) { paragraphs.push({ text: current.trim(), page: pageNumber }); current = ""; }
      continue;
    }
    if (/^[-•*]\s/.test(trimmed) || /^\d+[.)]\s/.test(trimmed)) {
      if (current.trim()) { paragraphs.push({ text: current.trim(), page: pageNumber }); current = ""; }
      current = trimmed;
      continue;
    }
    if (trimmed.length < 50 && trimmed.length > 0 && /^[A-ZÇĞİÖŞÜ]/.test(trimmed)) {
      if (current.trim()) { paragraphs.push({ text: current.trim(), page: pageNumber }); current = ""; }
      paragraphs.push({ text: trimmed, page: pageNumber });
      continue;
    }
    if (trimmed.endsWith('-') && next && /^[a-zçğıöşü]/.test(next)) {
      current += trimmed.slice(0, -1); // Tireyi at, kelimenin devamı bir sonraki satırda gelecek
      continue;
    }
    current += (current ? " " : "") + trimmed; // Normal devam satırı: araya boşluk koyup ekle
  }
  if (current.trim()) paragraphs.push({ text: current.trim(), page: pageNumber }); // Sayfa bitti, kalan varsa kaydet
  return paragraphs;
}

// ----------------------------------------------------------------------------
// extractMarkdown: LLM'e "sadece mindmap'i yaz, başka bir şey ekleme" desek
// bile bazı modeller yine de başına "Tabii, işte mindmap'iniz:" gibi bir
// giriş cümlesi ekleyebiliyor. Bu fonksiyon, cevabın içinde ilk "# " (başlık)
// satırını bulup, ondan ÖNCEKİ her şeyi atar. Böylece gereksiz giriş
// cümlelerinden temizlenmiş, doğrudan markdown ile başlayan bir metin kalır.
// ----------------------------------------------------------------------------
function extractMarkdown(raw: string): string {
  const idx = raw.search(/^#\s+/m); // "m" bayrağı: her satırın başını ayrı ayrı kontrol et
  return idx >= 0 ? raw.slice(idx).trim() : raw.trim();
}

// ----------------------------------------------------------------------------
// main: Script çalıştırıldığında baştan sona yürütülen asıl akış.
// ----------------------------------------------------------------------------
async function main() {
  // ---- PDF dosyasını bul ----
  const pdfArg = process.argv[2]; // Komut satırından elle bir yol verilmiş mi? (opsiyonel)
  let pdfPath = pdfArg;

  if (!pdfPath) {
    // Verilmemişse documents/ klasöründeki ilk .pdf dosyasını otomatik seç
    const files = readdirSync(PDF_DIR).filter(f => f.toLowerCase().endsWith('.pdf'));
    if (files.length === 0) {
      console.error(`"${PDF_DIR}" klasöründe PDF bulunamadı.`);
      process.exit(1);
    }
    pdfPath = path.join(PDF_DIR, files[0]);
  }

  const pdfName = path.basename(pdfPath); // Sadece dosya adı (klasör yolu olmadan)
  const topic = pdfName.replace(/\.pdf$/i, '').replace(/[_-]+/g, ' '); // ".pdf" uzantısını ve alt çizgileri temizle -> konu başlığı

  // ---- 1) PDF'i oku ----
  console.log(`PDF okunuyor: ${pdfPath}`);
  const loader = new PDFLoader(pdfPath);
  const pages = await loader.load(); // pages: her biri bir sayfanın { pageContent, metadata } objesi
  console.log(`${pages.length} sayfa yüklendi.`);

  // Her sayfanın metnini düzgün paragraflara çevirip TEK bir listede topla.
  // Bu liste, ileride "kaynak eşleştirme" (embedding) adımında kullanılacak.
  const allParagraphs: { text: string; page: number }[] = [];
  pages.forEach(p => allParagraphs.push(...reconstructParagraphs(p.pageContent, pageNumberOf(p))));
  console.log(`${allParagraphs.length} paragraf çıkarıldı.`);

  // ---- LLM bağlantısını kur ----
  // AzureResponsesChatModel: kimlik bilgilerini (AZURE_OPENAI_RESPONSES_URL,
  // AZURE_OPENAI_API_KEY, AZURE_OPENAI_MODEL) çalışma anında ortam
  // değişkenlerinden okur (bkz. .env.example).
  //   - temperature: 0.3 -> düşük değer = daha "tutarlı/az yaratıcı" cevaplar
  //     (mindmap gibi yapılandırılmış bir çıktı için yüksek yaratıcılık istemeyiz)
  //   - maxTokens: 2048 -> modelin üretebileceği maksimum kelime parçası (token)
  //     sayısı. Bunu yeterince yüksek tutmazsak, uzun bir mindmap yarıda kesilebilir.
  const llm = new AzureResponsesChatModel({ temperature: 0.3, maxTokens: 2048 });

  // ---- 2) MAP AŞAMASI: PDF'i parça parça özetle ----
  // Neden tüm PDF'i tek seferde LLM'e vermiyoruz? Çünkü modellerin bir
  // "bağlam penceresi" (context window) sınırı vardır ve çok uzun metinler
  // hem bu sınırı aşabilir hem de model kalitesi düşebilir. Bunun yerine
  // sayfaları 8'erli gruplara (chunk) bölüp her grubu AYRI AYRI özetliyoruz.
  const chunks: { pages: number[]; text: string }[] = [];
  for (let i = 0; i < pages.length; i += PAGES_PER_CHUNK) {
    const slice = pages.slice(i, i + PAGES_PER_CHUNK);
    chunks.push({
      pages: slice.map(pageNumberOf),
      text: slice.map(p => `[Sayfa ${pageNumberOf(p)}]\n${p.pageContent}`).join('\n\n')
    });
  }

  console.log(`Map aşaması: ${chunks.length} parça özetlenecek (Azure OpenAI)...`);
  const summaries: string[] = [];
  for (let i = 0; i < chunks.length; i++) {
    const c = chunks[i];
    // Modelden Türkçe, akıcı, sıralı bir özet istiyoruz. "Sadece özeti yaz"
    // demek, gereksiz giriş/kapanış cümlelerini engellemeye yardımcı olur.
    const prompt = `Aşağıdaki PDF sayfalarını (Sayfa ${c.pages[0]}-${c.pages[c.pages.length - 1]}) "${topic}" konusu hakkında Türkçe, akıcı bir paragraf halinde, sırayı koruyarak ve önemli bilgileri kaybetmeden özetle. En fazla 400 kelime kullan, sadece özeti yaz, başka açıklama ekleme.

METIN:
${c.text}

ÖZET (Türkçe):`;
    const res = await llm.invoke(prompt); // Modele isteği gönder, cevabı bekle
    // res.content genelde düz metindir (string), ama bazı modellerde parça
    // parça (array) da gelebilir; ihtimale karşı JSON'a çevirip garantiye alıyoruz.
    const summary = typeof res.content === 'string' ? res.content : JSON.stringify(res.content);
    summaries.push(`[Bölüm ${i + 1}/${chunks.length} - Sayfa ${c.pages[0]}-${c.pages[c.pages.length - 1]}]\n${summary}`);
    console.log(`  ✓ Parça ${i + 1}/${chunks.length} özetlendi`);
  }

  // ---- 3) REDUCE AŞAMASI: Özetleri birleştirip mindmap üret ----
  console.log("Reduce aşaması: mindmap oluşturuluyor...");
  const reducePrompt = `Sen bir eğitim tasarım uzmanısın. Aşağıdaki özetleri kullanarak "${topic}" hakkında DETAYLI bir Türkçe mindmap oluştur.

KURALLAR:
1. Kategorileri (1., 2., 3. ...) ve maddeleri (1.1, 1.2, 2.1 ...) numaralandır
2. 4 ila 8 arası ana kategori, her kategoride 3 ila 6 madde olsun
3. Özetlerdeki mantıksal sırayı koru
4. Her kategori başlığının yanına süre ekle, format: (MM:SS), toplamda yaklaşık ${TOTAL_MINUTES} dakika olacak şekilde
5. SADECE mindmap içeriğini ver, başka hiçbir açıklama/giriş cümlesi ekleme

FORMAT:
# Başlık
## 1. Kategori (2:30)
- 1.1 Madde
- 1.2 Madde
## 2. Kategori (3:00)
- 2.1 Madde

ÖZETLER:
${summaries.join('\n\n---\n\n')}

Yukarıdaki kurallara uyarak "${topic}" için DETAYLI, SIRALI ve NUMARALANDIRILMIŞ Türkçe mindmap oluştur:`;

  const reduceRes = await llm.invoke(reducePrompt);
  // extractMarkdown: modelin cevabının başına eklediği olası "İşte mindmap'iniz:"
  // gibi giriş cümlelerini atıp, doğrudan "# Başlık" ile başlayan kısmı alır.
  const mindmapRaw = extractMarkdown(typeof reduceRes.content === 'string' ? reduceRes.content : JSON.stringify(reduceRes.content));
  console.log("✓ Mindmap taslağı oluşturuldu.");

  // ---- 4) KAYNAK EŞLEŞTİRME: Advanced RAG (query expansion + BM25 fusion + LLM reranking) ----
  console.log("Kaynaklar için BM25 indeksi oluşturuluyor...");
  const bm25 = new BM25Index(allParagraphs, p => p.text, tokenize);

  // Mindmap markdown'ındaki "- " ile başlayan satırları (yani maddeleri) bul.
  const lines = mindmapRaw.split('\n');
  const items: string[] = [];
  lines.forEach(line => {
    const t = line.trim();
    if (t.startsWith('-')) items.push(t.substring(1).trim()); // Baştaki "-" işaretini at, metni sakla
  });

  // ---- Pre-retrieval: sorgu genişletme (bkz. dosyanın başındaki açıklama) ----
  console.log(`${items.length} madde için sorgular genişletiliyor (query expansion)...`);
  const expansions = await expandQueriesBatch(items);

  // ---- Retrieval + fusion: her madde için multi-query BM25 araması ----
  console.log(`${items.length} madde için kaynak aranıyor (BM25 + fusion)...`);
  const PER_QUERY_K = 5;
  const CANDIDATE_POOL_SIZE = 8;

  const candidatesByItem = new Map<string, { text: string; page: number }[]>();
  for (const item of items) {
    const expansionWords = expansions.get(item) || [];
    const expandedQuery = expansionWords.length > 0 ? `${item} ${expansionWords.join(' ')}` : null;

    const originalResults = bm25.search(item, PER_QUERY_K).map(r => r.item);
    const expandedResults = expandedQuery ? bm25.search(expandedQuery, PER_QUERY_K).map(r => r.item) : [];

    // Fusion: iki sonucu birleştir, aynı paragrafı (metnine göre) tekrar etme.
    const seen = new Set<string>();
    const fused: { text: string; page: number }[] = [];
    for (const doc of [...originalResults, ...expandedResults]) {
      if (!seen.has(doc.text)) {
        seen.add(doc.text);
        fused.push(doc);
      }
    }
    candidatesByItem.set(item, fused.slice(0, CANDIDATE_POOL_SIZE));
  }

  // ---- Post-retrieval: LLM reranking — aday havuzundan gerçekten alakalı olanları seç ----
  console.log("LLM ile kaynaklar yeniden sıralanıyor (reranking)...");
  const citations: any[] = [];
  // Aynı madde metni birden fazla yerde geçse bile doğru eşleşsin diye,
  // her maddenin metnini anahtar (key) yapıp bulunan kaynak ID'lerini saklıyoruz.
  const citationIdsByItem = new Map<string, number[]>();
  let citationId = 0; // Her kaynağa benzersiz bir numara veriyoruz (0, 1, 2, ...)

  const RERANK_BATCH_SIZE = 5;
  const RERANK_RETRY_COUNT = 1;

  const extractJsonBlock = (raw: string): string => {
    const start = raw.indexOf('{');
    const end = raw.lastIndexOf('}');
    if (start === -1 || end === -1 || end < start) throw new Error("Cevapta JSON bulunamadı");
    return raw.slice(start, end + 1);
  };

  for (let i = 0; i < items.length; i += RERANK_BATCH_SIZE) {
    const batchItems = items.slice(i, i + RERANK_BATCH_SIZE);
    const batchNumber = i / RERANK_BATCH_SIZE + 1;

    let batchContext = "";
    batchItems.forEach((item, localIdx) => {
      const candidates = candidatesByItem.get(item) || [];
      batchContext += `\n=== ITEM ${localIdx}: "${item}" ===\nSources:\n`;
      candidates.forEach((c, cIdx) => {
        const content = c.text.length > 500 ? c.text.slice(0, 500) + "..." : c.text;
        batchContext += `[${cIdx}] Page ${c.page}: ${content}\n`;
      });
    });

    const batchPrompt = `You are a citation assistant. Each item has a pool of CANDIDATE sources (found via keyword search, not all necessarily relevant). Select ONLY the truly relevant source(s) for each item.

FOR EACH ITEM:
- Source indices refer to the item's own numbered candidate list (starts at 0)
- Select 1-2 truly relevant sources; ignore off-topic candidates even if they share some keywords
- If none of the candidates are actually relevant, return an empty array

ITEMS AND THEIR CANDIDATE SOURCES:
${batchContext}

Respond with ONLY a single JSON object, no other text, no markdown code fences, matching EXACTLY this shape:
{"citations":[{"item_index":0,"source_indices":[0,1]}]}`;

    const applyCitations = (parsed: { citations: { item_index: number; source_indices: number[] }[] }) => {
      parsed.citations.forEach(c => {
        const item = batchItems[c.item_index];
        if (!item) return;
        const candidates = candidatesByItem.get(item) || [];
        const ids = citationIdsByItem.get(item) || [];
        c.source_indices.forEach(srcIdx => {
          const cand = candidates[srcIdx];
          if (!cand) return;
          const id = citationId++;
          citations.push({ source_id: id, pdf_name: pdfName, page_number: cand.page, content: cand.text, madde: item });
          ids.push(id);
        });
        citationIdsByItem.set(item, ids);
      });
    };

    // applyFallback: LLM hiçbir şekilde başarılı olamazsa, en azından her
    // maddeye BM25 aramasının EN İYİ eşleşmesini otomatik ata.
    const applyFallback = () => {
      batchItems.forEach(item => {
        const candidates = candidatesByItem.get(item) || [];
        if (candidates.length > 0) {
          const cand = candidates[0];
          const id = citationId++;
          citations.push({ source_id: id, pdf_name: pdfName, page_number: cand.page, content: cand.text, madde: item });
          citationIdsByItem.set(item, [...(citationIdsByItem.get(item) || []), id]);
        }
      });
    };

    let succeeded = false;
    for (let attempt = 0; attempt <= RERANK_RETRY_COUNT && !succeeded; attempt++) {
      try {
        const res = await llm.invoke(batchPrompt);
        const parsed = JSON.parse(extractJsonBlock(res.content));
        applyCitations(parsed);
        succeeded = true;
      } catch (error: any) {
        if (attempt < RERANK_RETRY_COUNT) {
          console.warn(`⚠️ Batch ${batchNumber} kaynak seçimi başarısız (deneme ${attempt + 1}), tekrar deneniyor...`);
        } else {
          console.warn(`⚠️ Batch ${batchNumber} kaynak seçimi başarısız oldu (${error.message}), en alakalı kaynak otomatik ekleniyor.`);
          applyFallback();
        }
      }
    }
  }

  // Şimdi markdown'ı satır satır gezip, her maddenin sonuna bulduğumuz
  // kaynak numaralarını "[Kaynaklar: 0, 1]" şeklinde ekliyoruz.
  const finalMindmap = lines.map(line => {
    const t = line.trim();
    if (t.startsWith('-')) {
      const itemText = t.substring(1).trim();
      const ids = citationIdsByItem.get(itemText);
      if (ids && ids.length > 0) return `${line} [Kaynaklar: ${ids.join(', ')}]`;
    }
    return line; // Madde satırı değilse (başlık, boş satır vb.) olduğu gibi bırak
  }).join('\n');

  console.log(`✓ ${citations.length} kaynak eşleştirildi.`);

  // ---- 5) Son adım: Görsel HTML mindmap'i üret ----
  const outPath = await createLiveMindmapHTML(finalMindmap, citations, "mindmap_llm.html");
  console.log("Oluşturuldu:", outPath);
}

// Script'i çalıştır. Bir hata olursa (örn. Azure OpenAI kimlik bilgileri
// eksikse, ya da PDF bozuksa) programın sessizce takılıp kalması yerine hatayı ekrana yazıp
// çıkış kodu 1 (= "başarısız") ile sonlanmasını sağla.
main().catch(e => { console.error("HATA:", e); process.exit(1); });
