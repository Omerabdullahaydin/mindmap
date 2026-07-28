// ============================================================================
// BU DOSYA NE İŞE YARAR?
// ----------------------------------------------------------------------------
// Normalde bir HTML/JavaScript sayfasının gerçekten çalışıp çalışmadığını
// görmek için tarayıcıda (Chrome, Edge vb.) açıp konsola (F12) bakarız.
// Bu script'in amacı, TARAYICI AÇMADAN, terminalde/kod içinde SAHTE bir
// tarayıcı ortamı kurup (buna "jsdom" denir) üretilen mindmap HTML'ini
// orada çalıştırmak ve varsa hataları yakalamaktır.
//
// Neden işimize yaradı? "citation.madde" hatasını (ve başka olası hataları)
// gerçek bir tarayıcı açmadan, otomatik olarak tespit edebilmek için.
//
// ÖNEMLİ NOT: jsdom, gerçek bir tarayıcının HER özelliğini desteklemez.
// Örneğin SVG metinlerinin piksel genişliğini ölçen `getComputedTextLength`
// fonksiyonu jsdom'da YOKTUR — bu yüzden bu script çalıştırıldığında
// "getComputedTextLength is not a function" diye bir hata görebilirsiniz.
// Bu hata GERÇEK bir tarayıcıda (Chrome/Edge/Firefox) OLUŞMAZ, sadece
// jsdom'un bir eksikliğidir. Yani bu script'i "hiç hata yok" beklentisiyle
// değil, "beklenmedik/farklı bir hata var mı" diye kontrol etmek için kullanın.
// ============================================================================

import { readFile } from "fs/promises";
import { JSDOM, VirtualConsole } from "jsdom";

// 1) Daha önce generate.ts ile üretilmiş HTML dosyasını oku (metin olarak).
const html = await readFile("./mindmaps/mindmap_preview.html", "utf-8");

// 2) jsdom'un içindeki console.log/warn/error çağrılarını YAKALAYIP bizim
// gerçek terminalimize yazdırmak için bir "sanal konsol" kuruyoruz.
// Bunu yapmazsak, sahte tarayıcı içindeki console.log'lar hiçbir yere gitmez.
const virtualConsole = new VirtualConsole();
virtualConsole.on("jsdomError", (e) => {
  console.log("[jsdom error]", e.message);
  if ((e as any).detail) console.log("  detail:", (e as any).detail);
});
virtualConsole.on("error", (...args) => console.log("[console.error]", ...args));
virtualConsole.on("warn", (...args) => console.log("[console.warn]", ...args));
virtualConsole.on("log", (...args) => console.log("[console.log]", ...args));

// 3) Sahte tarayıcıyı (jsdom) kur ve HTML'i içine yükle.
//    - runScripts: "dangerously" -> HTML içindeki <script> etiketlerini
//      GERÇEKTEN çalıştır (varsayılan olarak jsdom script'leri çalıştırmaz,
//      güvenlik nedeniyle "dangerously" yazarak bilerek izin veriyoruz).
//    - resources: "usable" -> <script src="https://cdn...d3.js">  gibi
//      İNTERNETTEN yüklenen dosyaları da indirip çalıştırmasına izin ver
//      (d3.js ve marked.js kütüphaneleri CDN'den geliyor).
//    - url -> sahte bir sayfa adresi (bazı tarayıcı API'leri URL ister).
const dom = new JSDOM(html, {
  runScripts: "dangerously",
  resources: "usable",
  url: "https://example.org/mindmap_preview.html",
  virtualConsole,
  pretendToBeVisual: true
});

// 4) Sayfa içinde yakalanmamış bir JavaScript hatası (uncaught error) olursa
// bunu da terminale yazdır.
dom.window.addEventListener("error", (event: any) => {
  console.log("[window error]", event.error?.stack || event.message);
});

// 5) CDN'den d3.js ve marked.js'in indirilip çalışması biraz zaman alır.
// Bu yüzden 6 saniye bekliyoruz ki script'ler tam olarak yüklenip mindmap'i
// çizsin, daha sonra kontrol edelim.
await new Promise((resolve) => setTimeout(resolve, 6000));

// 6) Basit bir "sağlık kontrolü": editör kutusu (textarea) var mı,
// mindmap'te kaç tane düğüm (node) çizilmiş, onu yazdır.
console.log("---- editor value present:", !!dom.window.document.getElementById("editor"));
console.log("---- node count in SVG:", dom.window.document.querySelectorAll(".node").length);

// 7) Sahte tarayıcıyı kapat (bellek/kaynak temizliği).
dom.window.close();
