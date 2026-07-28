// ============================================================================
// BU DOSYA NE İŞE YARAR?
// ----------------------------------------------------------------------------
// Bu bir "deneme/örnek" script'idir. Amacı: mindmap HTML'ini üreten gerçek
// fonksiyonu (createLiveMindmapHTML) elle yazılmış SAHTE (örnek) bir markdown
// ve örnek kaynak (citation) verisiyle çalıştırıp, sonucu bir HTML dosyası
// olarak diske yazmaktır.
//
// Neden lazım? Gerçek PDF + Azure/LLM olmadan da mindmap'in NASIL GÖRÜNDÜĞÜNÜ
// test edebilmek için. Yani: "PDF'den geldiğini varsayalım" diyip sahte veri
// veriyoruz, gerçek görselleştirme kodu (mindmap_visualization_server.ts)
// bu sahte veriyle de gerçek veriyle de aynı şekilde çalışır.
// ============================================================================

// createLiveMindmapHTML: markdown + kaynak listesini alıp interaktif bir
// D3.js mindmap HTML dosyası üreten asıl fonksiyon. Bu fonksiyona hiç
// dokunmadık, olduğu gibi kullanıyoruz.
import { createLiveMindmapHTML } from "./mcp/mindmap_visualization_server.js";

// ---- 1) Sahte (örnek) mindmap markdown'ı ----
// Gerçek sistemde bu metni bir LLM üretir (bkz. generate_mindmap_llm.ts).
// Burada elle yazıyoruz ki Azure/Ollama olmadan da test edebilelim.
// Format kuralları:
//   "# Başlık"                -> mindmap'in kök (ortadaki) düğümü
//   "## 1. Kategori (MM:SS)"  -> ana dal (kategori), yanındaki süre kozmetik
//   "- 1.1 Madde metni"       -> o kategorinin altındaki bir alt düğüm (yaprak)
//   "[Kaynaklar: 0, 1]"       -> bu maddenin hangi kaynak numaralarına
//                                bağlı olduğunu belirten "rozet" (badge)
const sampleMarkdown = `# Fransa'da Solidarizm
## 1. Giriş (2:30)
- 1.1 Solidarizm kavramının tanımı [Kaynaklar: 0, 1]
- 1.2 Tarihsel arka plan [Kaynaklar: 2]
- 1.3 Fransa'daki ortaya çıkışı

## 2. Temel İlkeler (3:45)
- 2.1 Dayanışma ilkesi [Kaynaklar: 0]
- 2.2 Sosyal sözleşme yaklaşımı
- 2.3 Bireysel ve toplumsal denge [Kaynaklar: 1, 2]
- 2.4 Léon Bourgeois'nın katkıları

## 3. Sonuçlar (2:00)
- 3.1 Sosyal politikalara etkisi [Kaynaklar: 2]
- 3.2 Günümüzdeki yansımaları`;

// ---- 2) Sahte kaynak (citation) listesi ----
// Her citation, mindmap'teki bir maddenin "kanıtı"dır: hangi PDF'in, hangi
// sayfasından geldiği ve o sayfadaki gerçek metin (content). Kullanıcı
// mindmap'te bir kutuya tıklayınca bu bilgiler bir pencerede (modal) açılır.
const sampleCitations = [
  { source_id: 0, pdf_name: "FransaSolidarizm.pdf", page_number: 4, content: "Solidarizm, bireyler arasındaki karşılıklı bağımlılığı..." },
  { source_id: 1, pdf_name: "FransaSolidarizm.pdf", page_number: 7, content: "19. yüzyıl sonunda Fransa'da..." },
  { source_id: 2, pdf_name: "FransaSolidarizm.pdf", page_number: 12, content: "Léon Bourgeois'nın 'Solidarité' adlı eseri..." }
];

// ÖNEMLİ: Her citation'da bir "madde" alanı OLMAK ZORUNDA.
// Bu alan, citation'ın mindmap'teki HANGİ maddeye ait olduğunu, o maddenin
// TAM METNİYLE (kelimesi kelimesine) eşleştirerek belirtir. Bu eşleştirme
// olmazsa (ya da yanlış yazılırsa) tıklama modalı hiçbir şey göstermez.
// (Bu alanın eksik gelmesi, daha önce bulduğumuz "citation.madde" çökme
// hatasının tam olarak sebebiydi — bkz. repro_missing_madde.ts)
const withMadde = [
  { ...sampleCitations[0], madde: "1.1 Solidarizm kavramının tanımı [Kaynaklar: 0, 1]" },
  { ...sampleCitations[1], madde: "1.2 Tarihsel arka plan [Kaynaklar: 2]" },
  { ...sampleCitations[2], madde: "2.1 Dayanışma ilkesi [Kaynaklar: 0]" }
];

// ---- 3) HTML'i üret ----
// createLiveMindmapHTML üç parametre alır:
//   1. markdown metni
//   2. citation listesi
//   3. çıktı dosyasının adı (mindmaps/ klasörü altına yazılır)
// "await" kullanıyoruz çünkü bu fonksiyon "async" (dosya yazma işlemi bitene
// kadar bekler).
const outPath = await createLiveMindmapHTML(sampleMarkdown, withMadde, "mindmap_preview.html");

// Üretilen dosyanın tam yolunu terminale yazdır, böylece tarayıcıda
// açabilmek için nereye gideceğimizi görürüz.
console.log("Generated:", outPath);
