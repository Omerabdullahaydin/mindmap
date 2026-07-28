// ============================================================================
// BU DOSYA NE İŞE YARAR? (Baştan sona, adım adım)
// ----------------------------------------------------------------------------
// Bu, PDF'ten GERÇEK bir yapay zeka (LLM) kullanarak mindmap üreten script'tir.
// Azure OpenAI yerine, bilgisayarınızda YEREL olarak çalışan "Ollama" adlı
// programı ve onun üzerinde çalışan iki modeli kullanır:
//
//   1) Bir "chat" (sohbet/metin üretme) modeli -> PDF metnini özetlemek ve
//      mindmap'in kategori/madde yapısını yazmak için (varsayılan: qwen2.5:7b)
//   2) Bir "embedding" (metni sayılara çevirme) modeli -> her mindmap
//      maddesinin, PDF'in HANGİ paragrafından geldiğini bulmak için
//      (varsayılan: nomic-embed-text)
//
// "Embedding" ne demek, kısaca: Bir cümleyi sayılardan oluşan bir listeye
// (vektöre) çeviririz. Anlamca birbirine yakın iki cümlenin vektörleri de
// birbirine yakın çıkar. Böylece "bu mindmap maddesine en çok hangi PDF
// paragrafı benziyor?" sorusunu, kelimeleri birebir aratmadan, ANLAM bazında
// cevaplayabiliriz. Bu işleme İngilizce'de "semantic search" (anlamsal arama)
// denir. Azure AI Search'ün (ve orijinal projedeki rag_memory2.ts'in) yaptığı
// iş de tam olarak buydu; biz burada aynısını, bulutta değil, yerel bilgisayarda
// (Ollama ile) ve LangChain'in basit "MemoryVectorStore" (bellekte vektör
// deposu) aracıyla yapıyoruz.
//
// GENEL AKIŞ:
//   1. PDF'i oku, sayfa sayfa metne çevir, paragrafları yeniden birleştir
//   2. MAP AŞAMASI: Sayfaları 8'erli gruplara böl, her grubu LLM'e gönderip
//      Türkçe bir özet çıkart (uzun bir PDF'i tek seferde LLM'e vermek yerine
//      parça parça özetlemek, "map-reduce" denen klasik bir tekniktir)
//   3. REDUCE AŞAMASI: Tüm özetleri birleştirip LLM'e tekrar gönder, bu sefer
//      "bana numaralı, kategorili bir mindmap markdown'ı yaz" diye iste
//   4. KAYNAK EŞLEŞTİRME: Mindmap'teki her maddeyi, embedding modeliyle
//      PDF'in orijinal paragraflarıyla karşılaştırıp en çok benzeyenleri bul
//      (bu, tıklanınca "kaynak gösteren" citation özelliğini besler)
//   5. Sonucu, mindmap_visualization_server.ts'teki GERÇEK HTML üretme
//      fonksiyonuna verip görsel mindmap dosyasını oluştur
// ============================================================================

import { PDFLoader } from "@langchain/community/document_loaders/fs/pdf"; // PDF'i sayfa sayfa metne çevirir
import { ChatOllama, OllamaEmbeddings } from "@langchain/ollama";          // Yerel Ollama modellerine bağlanmak için LangChain araçları
import { MemoryVectorStore } from "langchain/vectorstores/memory";        // Basit, bellekte (RAM'de) çalışan bir "vektör veritabanı"
import { Document } from "@langchain/core/documents";                    // LangChain'in standart "metin parçası" objesi
import { createLiveMindmapHTML } from "./mcp/mindmap_visualization_server.js"; // Gerçek HTML mindmap üretici
import path from "path";
import { readdirSync } from "fs";

// ---- Ayarlar (istenirse ortam değişkeniyle değiştirilebilir) ----
const PDF_DIR = "./documents";

// qwen2.5:7b: doğrudan cevap veren (thinking/reasoning ön hazırlığı YAPMAYAN)
// bir "instruct" (talimat izleyen) model. Neden bunu seçtik?
// İlk denemede en büyük/en yeni yerel model olan "gemma4"ü kullandık, ama o
// bir "thinking" modeliydi: cevap vermeden önce sayfalarca kendi kendine
// "düşünme" metni yazıyor. Bu düşünme metni token (kelime parçası) sınırına
// takılıp asıl mindmap'e HİÇ ULAŞAMADAN kesildi. qwen2.5:7b böyle bir "iç
// monolog" yapmadan doğrudan cevap verdiği için bu iş için daha güvenilir.
const CHAT_MODEL = process.env.OLLAMA_CHAT_MODEL || "qwen2.5:7b";
const EMBED_MODEL = process.env.OLLAMA_EMBED_MODEL || "nomic-embed-text";

const PAGES_PER_CHUNK = 8;       // Map aşamasında kaç sayfa bir arada özetlensin
const TOTAL_MINUTES = 12;        // Mindmap'teki (MM:SS) süre etiketlerinin toplamı (sadece görsel/kozmetik)
const CITATIONS_PER_ITEM = 2;    // Her madde için en fazla kaç kaynak (benzer paragraf) bulunsun

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
  // ChatOllama: yerel Ollama sunucusuna (varsayılan olarak http://127.0.0.1:11434
  // adresinde çalışır) bağlanıp seçtiğimiz modelle konuşmamızı sağlayan LangChain sınıfı.
  //   - temperature: 0.3 -> düşük değer = daha "tutarlı/az yaratıcı" cevaplar
  //     (mindmap gibi yapılandırılmış bir çıktı için yüksek yaratıcılık istemeyiz)
  //   - numPredict: 2048 -> modelin üretebileceği maksimum kelime parçası (token)
  //     sayısı. Bunu yeterince yüksek tutmazsak, uzun bir mindmap yarıda kesilebilir.
  const llm = new ChatOllama({ model: CHAT_MODEL, temperature: 0.3, numPredict: 2048 });

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

  console.log(`Map aşaması: ${chunks.length} parça özetlenecek (model: ${CHAT_MODEL})...`);
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

  // ---- 4) KAYNAK EŞLEŞTİRME: Her maddeyi en benzer PDF paragrafıyla eşleştir ----
  console.log(`Kaynaklar embedleniyor (model: ${EMBED_MODEL})...`);

  // OllamaEmbeddings: metni sayı listesine (vektöre) çeviren model bağlantısı.
  const embeddings = new OllamaEmbeddings({ model: EMBED_MODEL });

  // PDF'ten çıkardığımız tüm paragrafları LangChain'in "Document" formatına çevir
  // (pageContent = metin, metadata = ek bilgi; burada hangi sayfadan geldiğini tutuyoruz).
  const docs = allParagraphs.map(p => new Document({ pageContent: p.text, metadata: { page: p.page } }));

  // MemoryVectorStore.fromDocuments: TÜM paragrafları embedding modeline gönderip
  // vektörlerini hesaplar ve bunları bellekte (RAM'de) bir "arama dizini" olarak saklar.
  // Bu, Azure AI Search'ün yaptığı işin basit/yerel bir karşılığıdır.
  const vectorStore = await MemoryVectorStore.fromDocuments(docs, embeddings);

  // Mindmap markdown'ındaki "- " ile başlayan satırları (yani maddeleri) bul.
  const lines = mindmapRaw.split('\n');
  const items: string[] = [];
  lines.forEach(line => {
    const t = line.trim();
    if (t.startsWith('-')) items.push(t.substring(1).trim()); // Baştaki "-" işaretini at, metni sakla
  });

  console.log(`${items.length} madde için kaynak aranıyor...`);
  const citations: any[] = [];
  // Aynı madde metni birden fazla yerde geçse bile doğru eşleşsin diye,
  // her maddenin metnini anahtar (key) yapıp bulunan kaynak ID'lerini saklıyoruz.
  const citationIdsByItem = new Map<string, number[]>();
  let citationId = 0; // Her kaynağa benzersiz bir numara veriyoruz (0, 1, 2, ...)

  for (const item of items) {
    // similaritySearch: "Bu maddeye ANLAMCA en çok hangi paragraflar benziyor?"
    // sorusunu sorar ve en yakın CITATIONS_PER_ITEM (2) tanesini döndürür.
    const results = await vectorStore.similaritySearch(item, CITATIONS_PER_ITEM);
    const ids: number[] = [];
    for (const r of results) {
      const id = citationId++;
      citations.push({
        source_id: id,
        pdf_name: pdfName,
        page_number: r.metadata.page, // Bu paragrafın PDF'in kaçıncı sayfasından geldiği
        content: r.pageContent,       // Paragrafın gerçek metni (tıklayınca gösterilecek)
        // "madde" alanı, bu kaynağın mindmap'teki HANGİ maddeye ait olduğunu
        // birebir metin eşleşmesiyle belirtir (tıklama modalının çalışması için ZORUNLU).
        madde: item
      });
      ids.push(id);
    }
    citationIdsByItem.set(item, ids);
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

// Script'i çalıştır. Bir hata olursa (örn. Ollama sunucusu kapalıysa, ya da
// PDF bozuksa) programın sessizce takılıp kalması yerine hatayı ekrana yazıp
// çıkış kodu 1 (= "başarısız") ile sonlanmasını sağla.
main().catch(e => { console.error("HATA:", e); process.exit(1); });
