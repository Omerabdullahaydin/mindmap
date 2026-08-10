// ============================================================================
// BU DOSYA NE İŞE YARAR? (Genel bakış)
// ----------------------------------------------------------------------------
// Bu, tüm mindmap üretim sürecini yöneten ANA dosyadır. "LangGraph" adlı bir
// kütüphane kullanarak, işi birbirine bağlı küçük ADIMLARA (node/düğüm) böler:
//
//   docIngest ---> map ---> reduce ---> tools ---> expandItems ---> (bitti)
//
// Her adım bir öncekinin çıktısını girdi olarak alır (buna "state" - durum -
// denir, aşağıda MindmapStateAnnotation ile tanımlanır). Adımlar sırasıyla:
//
//   1) docIngest    -> PDF'i oku, paragraflara ayır, embedding'lerini
//                      hesaplayıp yerel vektör deposuna yükle (rag_memory2.ts)
//   2) map          -> PDF'i 10'ar sayfalık parçalara bölüp HER PARÇAYI
//                      ayrı ayrı özetle (LLM'e paralel istekler)
//   3) reduce       -> Tüm özetleri birleştirip TEK bir mindmap markdown'ı üret
//   4) tools        -> Mindmap'teki her maddeye kaynak (citation) ekle, sonra
//                      görsel HTML dosyasını üret
//   5) expandItems  -> Her kategori için ayrıca uzun bir "eğitim anlatım
//                      metni" (podcast/seslendirme senaryosu gibi) üret
//
// Not: Buradaki TÜM LLM çağrıları Azure OpenAI üzerinden yapılıyor
// (getAzureChatModel, bkz. Utils/helper.ts).
//
// MİMARİ: Bütün pipeline `MindmapPipeline` class'ı içinde toplanıyor. Beş
// adım da bu class'ın private metotları — her biri `this.ragStore`/
// `this.citationTool`/`this.visualizer`/`this.*LLM` gibi instance alanlarını
// kullanıyor. ESKİDEN bu bağımlılıklar modül seviyesinde (ör. rag_memory2.ts'
// teki `let localDocuments`) paylaşılan mutable state'ti — bu, aynı process
// içinde iki pipeline çalışması üst üste binerse birinin diğerini ezmesine
// yol açabilecek bir concurrency riskiydi. Her `new MindmapPipeline()` kendi
// izole `MindmapRagStore`'unu oluşturduğu için (bkz. Rag/rag_memory2.ts), bu
// risk artık yok — kullanım şekli "her çalıştırma için yeni bir instance"
// olduğu sürece (bkz. run_full_pipeline.ts).
// ============================================================================

import "dotenv/config"; // .env dosyasındaki ortam değişkenlerini (varsa) otomatik yükle
import { StateGraph, START, END, Annotation } from "@langchain/langgraph"; // Adım adım iş akışı (graph) kurmamızı sağlayan kütüphane
import { getAzureChatModel } from "./Utils/helper.js";
import type { AzureResponsesChatModel } from "./Utils/azureResponsesClient.js";
import type { Document } from "@langchain/core/documents";
import { MindmapRagStore, extractPageNumber } from "./Rag/rag_memory2.js";
import { CitationTool } from "./Tools/rag_tool_citation.js";
import { MindmapVisualizer } from "../mcp/mindmap_visualization_server.js";

// ----------------------------------------------------------------------------
// STATE ANNOTATION: Bu "iş akışının hafızası" gibi düşünülebilir. Her adım
// (node) bu obje üzerinden veri okur ve yeni veri yazar. Her alanın bir
// "reducer" fonksiyonu var: "(x, y) => y ?? x" demek, "yeni değer (y) varsa
// onu kullan, yoksa eskisini (x) koru" demektir - yani bir sonraki adım bu
// alanı güncellememişse eski değer kaybolmaz. Bu bir şema/tip tanımı olduğu
// için (mutable state DEĞİL), module-level ve export kalıyor — StateGraph'ın
// constructor'ı ve her node'un tip imzası buna statik bir değer olarak
// ihtiyaç duyuyor.
// ----------------------------------------------------------------------------
export const MindmapStateAnnotation = Annotation.Root({
  topic: Annotation<string>({           // Kullanıcının verdiği konu başlığı (ör. "Bilgisayar oyunları...")
    reducer: (x, y) => y ?? x,
    default: () => ""
  }),

  documents: Annotation<Document[]>({   // docIngest adımından çıkan, PDF'ten ayrıştırılmış paragraflar
    reducer: (x, y) => y ?? x,
    default: () => []
  }),

  summaries: Annotation<string[]>({     // map adımından çıkan, her sayfa grubunun özeti
    reducer: (x, y) => y ?? x,
    default: () => []
  }),

  mindmap_without_citations: Annotation<string>({  // reduce adımından çıkan HAM mindmap markdown'ı (henüz kaynak yok)
    reducer: (x, y) => y ?? x,
    default: () => ""
  }),

  mindmap_with_citations: Annotation<string>({     // tools adımından çıkan, [Kaynaklar: ...] rozetleri eklenmiş mindmap
    reducer: (x, y) => y ?? x,
    default: () => ""
  }),

  citations: Annotation<any[]>({        // tools adımından çıkan, her kaynağın detaylarını (sayfa, içerik vb.) tutan liste
    reducer: (x, y) => y ?? x,
    default: () => []
  }),

  html_file_path: Annotation<string>({  // Üretilen görsel mindmap HTML dosyasının diskteki yolu
    reducer: (x, y) => y ?? x,
    default: () => ""
  }),

  expanded_items: Annotation<Array<{    // expandItems adımından çıkan, her kategori için üretilen anlatım metinleri
    categoryId: number,
    categoryTitle: string,
    timeMinutes: number,
    timeSeconds: number,
    totalSeconds: number,
    content: string
  }>>({
    reducer: (x, y) => y ?? x,
    default: () => []
  })
});

type MindmapState = typeof MindmapStateAnnotation.State;

// ============================================================================
// MindmapPipeline
// ----------------------------------------------------------------------------
// Beş adımlı LangGraph pipeline'ını (docIngest -> map -> reduce -> tools ->
// expandItems) kapsülleyen sınıf. Kullanım: `await new MindmapPipeline().run(topic)`.
// ============================================================================
export class MindmapPipeline {
  private ragStore: MindmapRagStore;
  private citationTool: CitationTool;
  private visualizer: MindmapVisualizer;
  private mapLLM: AzureResponsesChatModel;
  private reduceLLM: AzureResponsesChatModel;
  private expandLLM: AzureResponsesChatModel;
  private graph: any; // workflow.compile()'ın döndürdüğü CompiledStateGraph

  constructor() {
    this.ragStore = new MindmapRagStore();
    this.citationTool = new CitationTool(this.ragStore);
    this.visualizer = new MindmapVisualizer();
    this.mapLLM = getAzureChatModel();
    this.reduceLLM = getAzureChatModel();
    // Bu adım uzun anlatım metinleri ürettiği için maxTokens'ı daha yüksek tutuyoruz.
    this.expandLLM = getAzureChatModel({ maxTokens: 3000 });

    // ------------------------------------------------------------------------
    // İŞ AKIŞI (WORKFLOW) TANIMI
    // ------------------------------------------------------------------------
    // Beş node'u isimlendirip fonksiyonlarına bağlıyoruz (.addNode), sonra
    // hangi düğümün hangi düğümden sonra çalışacağını (.addEdge) belirtiyoruz:
    //
    //   START -> docIngest -> map -> reduce -> tools -> expandItems -> END
    //
    // Node'lar bu class'ın PRIVATE METOTLARI — LangGraph'a "plain function"
    // referansı vermek gerektiği için `.bind(this)` ile bağlanıyorlar. Bu
    // beş bağlama noktası burada, constructor'da tek yerde toplanmış oluyor
        // (projedeki diğer tüm class'lar da prototype method kullanıyor, arrow
    // class field yerine bu deseni tercih ettik — tutarlılık için).
    // ------------------------------------------------------------------------
    const workflow = new StateGraph(MindmapStateAnnotation)
      .addNode("docIngest", this.docIngestNode.bind(this))
      .addNode("map", this.mapNode.bind(this))
      .addNode("reduce", this.reduceNode.bind(this))
      .addNode("tools", this.toolNode.bind(this))
      .addNode("expandItems", this.expandItemsNode.bind(this))

      .addEdge(START, "docIngest")
      .addEdge("docIngest", "map")
      .addEdge("map", "reduce")
      .addEdge("reduce", "tools")
      .addEdge("tools", "expandItems")
      .addEdge("expandItems", END);

    this.graph = workflow.compile();
    this.graph.name = "Mindmap Citation Agent V2 (Tool-Based)";
    this.graph.description = "PDF dökümanlarından Map-Reduce + LLM Tools ile citation ve HTML mindmap oluşturur";
  }

  // run: LangGraph'a "bu başlangıç durumuyla (state) baştan sona çalış" der.
  // Bu tek çağrı, arka planda sırasıyla tüm node'ları tetikler; her adımın
  // çıktısı otomatik olarak bir sonrakine state üzerinden aktarılır.
  async run(topic: string): Promise<MindmapState> {
    return this.graph.invoke({ topic });
  }

  // ============================================================================
  // ADIM 1: DOC INGEST NODE — PDF'leri oku, paragraflara ayır, embedding hesapla
  // ----------------------------------------------------------------------------
  // GİRDİ: yok (documents/ klasöründen okur)
  // ÇIKTI: documents (PDF'ten ayrıştırılmış paragraflar)
  // ============================================================================
  private async docIngestNode(_state: MindmapState) {
    const paragraphs = await this.ragStore.ingest();
    return { documents: paragraphs };
  }

  // ============================================================================
  // ADIM 2: MAP NODE — PDF'i parça parça özetle (her 10 sayfa)
  // ----------------------------------------------------------------------------
  // GİRDİ (state'ten okunan): topic, documents (docIngest'ten gelen paragraflar)
  // ÇIKTI (state'e yazılan): summaries (her parçanın Türkçe özeti, string dizisi)
  // ============================================================================
  private async mapNode(state: MindmapState) {
    const topic = state.topic;
    const documents = state.documents ?? [];

    const llm = this.mapLLM;

    // Hiç doküman (PDF paragrafı) yoksa özetlenecek bir şey yok, boş dön.
    if (documents.length === 0) {
      return { summaries: [] };
    }

    // ---- Konu (topic) temizleme ----
    // Kullanıcı "bilgisayar oyunları hakkında detaylı bir mindmap oluştur" gibi
    // bir cümle yazmış olabilir. Burada, konunun ANLAMINI bozmayan ama mindmap
    // için gereksiz olan kalıp kelimeleri ("hakkında", "oluştur", "lütfen" vb.)
    // eleyip geriye asıl ANAHTAR KELİMELERİ bırakıyoruz.
    const stopWords = [
      "hakkında", "hakkinda", "ile", "ilgili", "için", "icin",
      "zihin", "haritası", "haritasi", "mindmap", "mind", "map",
      "oluştur", "olustur", "yap", "hazırla", "hazirla", "çiz", "ciz",
      "bana", "lütfen", "lutfen", "bir", "tane", "detaylı", "detayli"
    ];

    const cleanedTopic = topic
      .toLowerCase()
      .split(/\s+/)
      .filter(word => !stopWords.includes(word) && word.length > 2)
      .join(" ");

    const topicWords = cleanedTopic.split(/\s+/).filter(w => w.length > 0);

    // ---- Hızlı anahtar kelime kontrolü ----
    // PDF'in gerçekten bu konuyla ilgili olup olmadığına dair basit bir "akıl
    // kontrolü": ilk 100 paragrafın başında konu kelimelerinden biri geçiyor mu?
    // Bu SADECE bir uyarı amaçlıdır — geçmese bile işlem durdurulmaz, sadece
    // konsola bir uyarı yazılır (belki kullanıcı konuyu biraz farklı yazmıştır).
    let quickCheck = "";
    documents.slice(0, 100).forEach((doc) => {
      quickCheck += doc.pageContent.substring(0, 500) + " ";
    });

    let hasKeyword = false;
    for (const word of topicWords) {
      if (quickCheck.toLowerCase().includes(word.toLowerCase())) {
        hasKeyword = true;
        break;
      }
    }

    if (!hasKeyword) {
      console.warn(`⚠️ "${cleanedTopic}" keyword'ü bulunamadı, ama Map-Reduce ile devam ediliyor...`);
    }

    // ---- MAP AŞAMASI: paragrafları sayfa numarasına göre grupla, 10'ar
    // sayfalık parçalara (chunk) böl ----
    // Neden? Uzun bir PDF'i TEK seferde LLM'e vermek yerine, parça parça
    // özetlemek hem modelin bağlam sınırını aşmayı önler hem de PARALEL
    // (aynı anda) birden fazla istek göndererek zamandan kazandırır.
    const PAGES_PER_CHUNK = 10;

    // Paragrafın metninde "[Sayfa 5]" gibi bir etiket var (rag_memory2.ts
    // bunu paragrafları oluştururken ekliyor); extractPageNumber onu geri
    // okuyor — sayfa numarası burada BİR KEZ çıkarılıp chunk.docs'a taşınıyor,
    // aşağıdaki chunkContent oluşturma adımında tekrar regex çalıştırılmıyor.
    const pageMap = new Map<number, Document[]>();
    documents.forEach(doc => {
      const pageNum = extractPageNumber(doc.pageContent);
      if (!pageMap.has(pageNum)) {
        pageMap.set(pageNum, []);
      }
      pageMap.get(pageNum)!.push(doc);
    });

    const sortedPages = Array.from(pageMap.keys()).sort((a, b) => a - b); // Sayfaları küçükten büyüğe sırala
    const chunks: { pages: number[]; docs: { pageNum: number; doc: Document }[] }[] = [];

    for (let i = 0; i < sortedPages.length; i += PAGES_PER_CHUNK) {
      const chunkPages = sortedPages.slice(i, i + PAGES_PER_CHUNK);
      const chunkDocs: { pageNum: number; doc: Document }[] = [];
      chunkPages.forEach(pageNum => {
        (pageMap.get(pageNum) || []).forEach(doc => chunkDocs.push({ pageNum, doc }));
      });
      chunks.push({ pages: chunkPages, docs: chunkDocs });
    }

    // Her parça (chunk) için AYRI bir özetleme isteği hazırla. "chunks.map(async...)"
    // ile hepsi PARALEL olarak (aynı anda) başlatılır, en altta Promise.all ile
    // hepsinin bitmesi beklenir.
    const summaryPromises = chunks.map(async (chunk, chunkIdx) => {
      const chunkContent = chunk.docs
        .map(({ pageNum, doc }) => `[Sayfa ${pageNum}] ${doc.pageContent}`)
        .join('\n\n');

      const pageRange = `${chunk.pages[0]}-${chunk.pages[chunk.pages.length - 1]}`;
      const summaryPrompt = `Create a comprehensive condensed version of the following ${chunk.pages.length} pages (Page ${pageRange}) about "${topic}".

INSTRUCTIONS:
1. Maintain ALL context and content flow - this should be a shortened version of the PDF, not just key points
2. Preserve sequential order - follow the document's natural chronological flow
3. Write in paragraph form - create a flowing narrative that captures the full content
4. Include relevant details, explanations, and examples from the source material
5. Keep the summary under 500 words maximum while preserving the complete context

GOAL: Create a condensed but complete version of these pages that maintains the full context and narrative flow.

PAGES:
${chunkContent}

COMPREHENSIVE SUMMARY (in Turkish, paragraph form):`;

      // ÖNEMLİ (sağlamlık): Bu try/catch SADECE bu parçayı etkiler. Bir parçanın
      // özetlenmesi başarısız olsa bile (ör. Azure OpenAI geçici bir hata verse),
      // diğer parçalar etkilenmez ve "Promise.all" tümden çökmez — sadece o
      // parça için "Özetlenemedi" yazan bir yer tutucu metin döner.
      try {
        const result = await llm.invoke(summaryPrompt);
        const summary = typeof result.content === 'string' ? result.content : JSON.stringify(result.content);
        return `[Chunk ${chunkIdx + 1}/${chunks.length} - Sayfa ${pageRange}]\n${summary}`;
      } catch (error: any) {
        console.error(`Chunk ${chunkIdx} özetleme hatası:`, error.message);
        return `[Chunk ${chunkIdx + 1}/${chunks.length}]\nÖzetlenemedi`;
      }
    });

    const summaries = await Promise.all(summaryPromises); // Tüm parçaların özetlenmesini bekle

    return { summaries }; // Bu, state'in "summaries" alanına yazılır
  }

  // ============================================================================
  // ADIM 3: REDUCE NODE — Tüm özetlerden TEK bir mindmap markdown'ı üret
  // ----------------------------------------------------------------------------
  // GİRDİ: topic, summaries (map'ten gelen özetler)
  // ÇIKTI: mindmap_without_citations (henüz kaynaksız, ham mindmap markdown'ı)
  // ============================================================================
  private async reduceNode(state: MindmapState) {
    const topic = state.topic;
    const summaries = state.summaries ?? [];

    // Hiç özet yoksa (map aşaması hiçbir şey üretemediyse), hata mesajı
    // niteliğinde basit bir "mindmap" döndür — böylece sonraki adımlar
    // (tools, expandItems) çökmeden, "işlenecek veri yok" durumunu tanıyıp
    // erken çıkabilir (bkz. toolNode'daki "# Hata" kontrolü).
    if (summaries.length === 0) {
      return {
        mindmap_without_citations: "# Hata\n## Döküman bulunamadı\n- Özetler oluşturulamadı"
      };
    }

    const llm = this.reduceLLM;

    const documentsSummary = summaries.join('\n\n---\n\n');

    // Anlatım (seslendirme) için toplam hedef süre, dakika cinsinden.
    // Bu sadece bir HEDEF — LLM'e "kategorilerin süresi toplamda ~12 dakika
    // olsun" diye talimat veriyoruz. Model bazen bu talimatı tam tutturamıyor
    // (özellikle küçük yerel modellerde), bu yüzden expandItemsNode içinde
    // ayrıca bir "normalize etme" adımı var (aşağıda göreceksiniz).
    const totalTime = 12;

    const mindmapPrompt = `You are an expert instructional designer creating a comprehensive e-learning mindmap.

TASK: Create a DETAILED Turkish mindmap about "${topic}" using ONLY the information from the provided summaries.

CRITICAL REQUIREMENTS:
1. Maintain SEQUENTIAL ORDER - Follow the chronological flow of the document summaries
2. NUMBER categories (1., 2., 3.) and items with hierarchical numbering (1.1, 1.2, 2.1, 2.2, etc.)
3. Create 4 to 10 main categories with 3 to 6 items each (16-50 total items)
4. Use clear, professional language - avoid redundancy
5. Preserve the logical progression of concepts from the source material
6. **IMPORTANT**: Allocate TIME to each category. Total available time: ${totalTime} minutes.
   - Distribute time based on importance and complexity of each category
   - Format: (MM:SS) next to each category title
   - Example: "## 1. Introduction (2:30)" means 2 minutes 30 seconds
   - All category times MUST sum up to approximately ${totalTime} minutes

FORMAT:
# Title
## 1. Category 1 (2:30)
- 1.1 First item
- 1.2 Second item
- 1.3 Third item
- 1.4 Fourth item
## 2. Category 2 (3:45)
- 2.1 Fifth item
- 2.2 Sixth item
- 2.3 Seventh item
- 2.4 Eighth item
## 3. Category 3 (1:15)
- 3.1 Ninth item
- 3.2 Tenth item

DOCUMENT SUMMARIES (in sequential order):
${documentsSummary}

Create a DETAILED, SEQUENTIAL, and NUMBERED mindmap in Turkish about "${topic}". Remember to add time allocation (MM:SS) to EVERY category!`;

    try {
      const result = await llm.invoke(mindmapPrompt);
      const mindmapContent = typeof result.content === 'string'
        ? result.content
        : JSON.stringify(result.content);

      return {
        mindmap_without_citations: mindmapContent
      };
    } catch (error: any) {
      console.error("✗ Mindmap oluşturma hatası:", error.message);
      return {
        mindmap_without_citations: `# Hata\n## Mindmap oluşturulamadı\n- ${error.message}`
      };
    }
  }

  // ============================================================================
  // ADIM 4: TOOL NODE — Kaynak (citation) ekleme + HTML görselleştirme
  // ----------------------------------------------------------------------------
  // GİRDİ: mindmap_without_citations
  // ÇIKTI: mindmap_with_citations, citations, html_file_path
  // ============================================================================
  private async toolNode(state: MindmapState) {
    const mindmap = state.mindmap_without_citations;

    // Bir önceki adım (reduceNode) hata döndürdüyse, burada da devam etmenin
    // anlamı yok - hatayı olduğu gibi yukarı taşıyıp erken çık.
    if (!mindmap || mindmap.startsWith("# Hata")) {
      return {
        mindmap_with_citations: mindmap,
        citations: [],
        html_file_path: ""
      };
    }

    // Mindmap markdown'ındaki "- " ile başlayan satırları (yani maddeleri) topla.
    // Bunlar, citation tool'una "bu maddelere kaynak bul" diye gönderilecek.
    const lines = mindmap.split('\n');
    const items: string[] = [];

    for (const line of lines) {
      if (line.trim().startsWith('-')) {
        items.push(line.trim().substring(1).trim());
      }
    }

    try {
      // ---- ADIM 1: Kaynakları ekle ----
      console.log("🔍 Adding citations...");
      const citationResult = await this.citationTool.addCitations(mindmap, items);
      const finalMindmap = citationResult.mindmap_with_citations;
      const finalCitations = citationResult.citations;

      console.log(`✓ Citations added: ${finalCitations.length} citations`);

      // ---- ADIM 2: HTML görselleştirmesini oluştur ----
      // Aynı process içinde, ağ üzerinden hiçbir şeye bağlanmadan HTML'i üret.
      console.log("🎨 Creating HTML directly...");
      const htmlPath = await this.visualizer.render(finalMindmap, finalCitations);
      console.log(`✓ HTML created directly: ${htmlPath}`);

      return {
        mindmap_with_citations: finalMindmap,
        citations: finalCitations,
        html_file_path: htmlPath
      };

    } catch (error: any) {
      // Citation ya da HTML üretimi sırasında beklenmedik bir hata olursa,
      // en azından KAYNAKSIZ mindmap'i (orijinal haliyle) geri döndür —
      // kullanıcı hiçbir şey alamamak yerine en azından ham mindmap'i görsün.
      console.error("Tool execution error:", error.message);
      return {
        mindmap_with_citations: mindmap,
        citations: [],
        html_file_path: ""
      };
    }
  }

  // ============================================================================
  // ADIM 5: EXPAND ITEMS NODE — Her kategori için eğitim anlatım metni üret
  // ----------------------------------------------------------------------------
  // GİRDİ: mindmap_with_citations, citations, summaries
  // ÇIKTI: expanded_items (her kategori için ayrı bir "anlatım senaryosu")
  //
  // Bu adım, mindmap'teki her ana kategoriyi ("## 1. Kategori (2:30)") alıp,
  // o kategorinin süresine ve maddelerine göre bir SESLENDİRME/PODCAST
  // senaryosu (giriş - gelişme - sonuç yapısında) üretir.
  // ============================================================================
  private async expandItemsNode(state: MindmapState) {
    const mindmap = state.mindmap_with_citations;
    const citations = state.citations;
    const summaries = state.summaries;

    console.log("📚 Expanding categories with training content...");

    // ---- BİRİNCİ GEÇİŞ: Kategorileri bul ----
    // Regex açıklaması: "## 1. Kategori Adı (2:30)" gibi bir satırı yakalar.
    // Gruplar: (1)=numara, (2)=başlık metni, (3)=dakika, (4)=saniye
    const categoryRegex = /^##\s+(\d+)\.\s+(.+?)\s+\((\d+):(\d+)\)/gm;

    interface Category {
      id: number;
      title: string;
      timeMinutes: number;
      timeSeconds: number;
      totalSeconds: number;
      items: Array<{ id: string, text: string, citationIds: number[] }>;
      startIndex: number;
    }

    const categories: Category[] = [];
    let categoryMatch;

    while ((categoryMatch = categoryRegex.exec(mindmap)) !== null) {
      const categoryId = parseInt(categoryMatch[1]);
      const categoryTitle = categoryMatch[2].trim();
      const minutes = parseInt(categoryMatch[3]);
      const seconds = parseInt(categoryMatch[4]);

      categories.push({
        id: categoryId,
        title: categoryTitle,
        timeMinutes: minutes,
        timeSeconds: seconds,
        totalSeconds: minutes * 60 + seconds,
        items: [],
        startIndex: categoryMatch.index
      });
    }

    // ---- İKİNCİ GEÇİŞ: Maddeleri ilgili kategorilere ata ----
    // Regex: "- 1.1 Madde metni [Kaynaklar: 0, 2]" gibi bir satırı yakalar.
    // Gruplar: (1)=kategori no, (2)=madde no, (3)=madde metni, (4)=kaynak ID'leri (varsa)
    const itemRegex = /^\s*-\s+(\d+)\.(\d+)\s+(.+?)(?:\[Kaynaklar:\s*([\d,\s]+)\])?$/gm;
    let itemMatch;

    while ((itemMatch = itemRegex.exec(mindmap)) !== null) {
      const categoryId = parseInt(itemMatch[1]);
      const itemId = `${itemMatch[1]}.${itemMatch[2]}`;
      const itemText = itemMatch[3].trim();
      const citationText = itemMatch[4];
      const citationIds = citationText
        ? citationText.split(',').map(id => parseInt(id.trim())).filter(id => !isNaN(id))
        : [];

      const category = categories.find(c => c.id === categoryId);
      if (category) {
        category.items.push({ id: itemId, text: itemText, citationIds });
      }
    }

    console.log(`📝 Found ${categories.length} categories with items`);

    // Hiç kategori bulunamadıysa (LLM beklenen formatta yazmadıysa), boş bir
    // sonuç döndürüp devam et — bu, tüm programın çökmesi yerine sadece
    // "eğitim senaryosu üretilemedi" anlamına gelir.
    if (categories.length === 0) {
      console.log("⚠️ No categories found. Mindmap sample:", mindmap.substring(0, 300));
      return { expanded_items: [] };
    }

    categories.forEach(cat => {
      console.log(`   Category ${cat.id}: "${cat.title}" (${cat.timeMinutes}:${cat.timeSeconds.toString().padStart(2, '0')}) - ${cat.items.length} items`);
    });

    // ------------------------------------------------------------------------
    // SAĞLAMLIK ADIMI: SÜRE NORMALİZASYONU
    // ------------------------------------------------------------------------
    // reduceNode, LLM'e "kategori sürelerinin toplamı ~12 dakika olsun" diye
    // talimat veriyordu, ama küçük yerel modeller (ör. qwen2.5:7b) bu talimatı
    // bazen ciddi şekilde kaçırıyor — denemelerimizde bazı kategorilere "90:00"
    // (90 DAKİKA!) gibi anlamsız süreler verildiğini gördük. Bu, aşağıdaki
    // expandPromises kısmında "Introduction (${timeDuration * 0.25} min)" gibi
    // hesaplamaları saçmalaştırır (ör. "22.5 dakikalık giriş" gibi).
    //
    // Çözüm: Toplam süre, hedeften (TARGET_TOTAL_SECONDS) %30'dan fazla
    // sapmışsa, TÜM kategorilerin sürelerini ORANLI olarak yeniden ölçekleyip
    // toplamı hedefe getiriyoruz. Kategoriler arası GÖRECELİ denge (hangi
    // kategori daha uzun/kısa) korunur, sadece mutlak değerler makul hale gelir.
    // ------------------------------------------------------------------------
    const TARGET_TOTAL_SECONDS = 12 * 60; // reduceNode'daki "totalTime = 12" ile aynı hedef
    const rawTotalTime = categories.reduce((sum, cat) => sum + cat.totalSeconds, 0);

    if (rawTotalTime > 0) {
      const deviation = Math.abs(rawTotalTime - TARGET_TOTAL_SECONDS) / TARGET_TOTAL_SECONDS;
      if (deviation > 0.3) { // %30'dan fazla sapma varsa yeniden ölçekle
        console.warn(`⚠️ Kategori süreleri toplamı (${rawTotalTime}s) hedeften (${TARGET_TOTAL_SECONDS}s) çok sapmış, oranlı olarak düzeltiliyor...`);
        const scale = TARGET_TOTAL_SECONDS / rawTotalTime;
        categories.forEach(cat => {
          cat.totalSeconds = Math.max(20, Math.round(cat.totalSeconds * scale)); // En az 20 saniye kalsın
          cat.timeMinutes = Math.floor(cat.totalSeconds / 60);
          cat.timeSeconds = cat.totalSeconds % 60;
        });
      }
    }

    const totalAllocatedTime = categories.reduce((sum, cat) => sum + cat.totalSeconds, 0);
    console.log(`⏱️ Total allocated time: ${Math.floor(totalAllocatedTime / 60)}:${(totalAllocatedTime % 60).toString().padStart(2, '0')} (${totalAllocatedTime}s)`);

    console.log(`🎯 Expanding all ${categories.length} categories`);

    // Paralel eğitim içeriği oluştur (Azure OpenAI modeli).
    const llm = this.expandLLM;

    // Her kategori için AYRI bir "anlatım senaryosu" isteği hazırla, hepsi paralel çalışır.
    const expandPromises = categories.map(async (category, index) => {
      // Debug: sadece İLK kategori için ekstra ayrıntı yazdır (hepsi için
      // yazdırmak konsolu gereksiz kalabalıklaştırırdı).
      if (index === 0) {
        console.log("📋 Debug - Category:", category.title);
        console.log("📋 Debug - Category time:", `${category.timeMinutes}:${category.timeSeconds}`);
        console.log("📋 Debug - Items count:", category.items.length);
        console.log("📋 Debug - Citations array length:", citations.length);
        if (citations.length > 0) {
          console.log("📋 Debug - Citation[0] keys:", Object.keys(citations[0]));
          console.log("📋 Debug - Citation[0] sample:", JSON.stringify(citations[0]).substring(0, 300));
        }
      }

      // Bu kategorideki TÜM maddelerin kaynak ID'lerini tek bir kümede (Set) topla
      // (Set kullanmamızın sebebi: aynı kaynak birden fazla maddede geçebilir,
      // ama bir kez kullanmamız yeterli — Set otomatik olarak tekrarları eler).
      const allCitationIds = new Set<number>();
      category.items.forEach(item => {
        item.citationIds.forEach(id => allCitationIds.add(id));
      });

      if (index === 0) {
        console.log("📋 Debug - All citation IDs in category:", Array.from(allCitationIds));
      }

      // Kategorideki tüm madde metinlerini tek bir metinde birleştir (LLM'e context olarak verilecek)
      const allItemsText = category.items
        .map(item => `- ${item.id} ${item.text}`)
        .join('\n');

      // Bu kategoriye ait TÜM kaynakların gerçek PDF metnini bul ve birleştir
      // (LLM'in "sadece kaynak metne sadık kal" talimatını uygulayabilmesi için gerekli).
      const relatedCitations = Array.from(allCitationIds)
        .map(id => {
          const citation = citations[id];
          return citation?.content || citation?.text || '';
        })
        .filter(c => c)
        .join('\n\n---\n\n');

      // Bu kategoriye özel kaynak bulunamadıysa, en azından İLK kaynağı kullan
      // (hiç kaynak metni olmadan LLM'e "boş" bir prompt vermemek için).
      const citationText = relatedCitations || citations[0]?.content || citations[0]?.text || "Kaynak metin bulunamadı";
      const summary = summaries.join('\n\n');

      // MM:SS'i ondalıklı dakikaya çevir (ör. 2 dakika 30 saniye -> 2.5 dakika)
      const timeDuration = category.timeMinutes + (category.timeSeconds / 60);

      const prompt = `
# Role: You are an expert instructional designer and scriptwriter. Your goal is to transform the provided source materials into an engaging, clear, and educational narration script suitable for a voice-over or podcast.

# Input Data:
Time Duration: ${timeDuration.toFixed(2)} minutes (${category.timeMinutes}:${category.timeSeconds.toString().padStart(2, '0')})
**CATEGORY:** ${category.title}
**SUB-TOPICS:**
${allItemsText}
**SOURCE TEXT:** ${citationText}
**SUMMARY:** ${summary}

# Your Task:
1. Analyze the TOPIC, SUMMARY, and SOURCE TEXT deeply.
2. Synthesize this information to create a cohesive narration script.
3. Ensure the content is designed for *listening*, not just reading (use shorter sentences, natural pauses, and conversational transitions).
4. Strictly adhere to the provided information. Do not hallucinate or add external facts.

# Output Requirements:

## Format:
- Write in a script format.
- Include audio cues in brackets if necessary (e.g., [Pause], [Emphasis on this term]).
- Use timestamps (e.g., (00:00)) to mark sections.

## Pacing & Word Count Constraints:
- **Target Speed:** Write for a speaking pace of approximately **110-140 words per minute** (WPM).

## Structure:
The script must have a clear, logical flow based on the total time:
1. **Introduction (${timeDuration * 0.25} min):**
   - Introduce the topic.
   - Hook the listener and state the objective based on the Summary.
2. **Main Body (${timeDuration * 0.5} min):**
   - Break down the core concepts from the Source Text.
   - Explain the technical or educational details clearly using the Source Text.
   - Flow naturally between points.
3. **Conclusion (${timeDuration * 0.25} min):**
   - Summarize the key takeaways.
   - Provide a final closing thought based on the inputs.

## Tone:
- Professional, educational, yet accessible and engaging.
- Direct address to the audience ("You," "We").

## Fidelity:
- Use ONLY the information from the provided source text and summary.
- Do NOT add external knowledge.
- Write in Turkish.
`;

      // ÖNEMLİ (sağlamlık): Bu try/catch SADECE bu kategoriyi etkiler. 9-15
      // kategoriden BİR TANESİNİN LLM isteği başarısız olsa (ör. Azure
      // OpenAI'ın o an rate-limit/zaman aşımı vermesi vb.), "Promise.all"
      // TÜMÜYLE reddedilmez (reject) — sadece o kategori için bir hata notu
      // içeren yer tutucu metin döner, diğer kategorilerin anlatım metinleri
      // kaybolmaz.
      try {
        const result = await llm.invoke(prompt);
        const rawContent = result.content as string;

        return {
          expandedItem: {
            categoryId: category.id,
            categoryTitle: category.title,
            timeMinutes: category.timeMinutes,
            timeSeconds: category.timeSeconds,
            totalSeconds: category.totalSeconds,
            content: rawContent
          }
        };
      } catch (error: any) {
        console.error(`✗ Kategori ${category.id} ("${category.title}") için anlatım metni oluşturulamadı:`, error.message);
        return {
          expandedItem: {
            categoryId: category.id,
            categoryTitle: category.title,
            timeMinutes: category.timeMinutes,
            timeSeconds: category.timeSeconds,
            totalSeconds: category.totalSeconds,
            content: `[Bu kategori için anlatım metni oluşturulamadı: ${error.message}]`
          }
        };
      }
    });

    const results = await Promise.all(expandPromises); // Tüm kategorilerin bitmesini bekle

    const expandedItems = results.map(r => r.expandedItem);

    console.log(`✅ Expanded ${expandedItems.length} categories`);

    expandedItems.forEach(item => {
      console.log(`   Category ${item.categoryId}: "${item.categoryTitle}" - ${item.timeMinutes}:${item.timeSeconds.toString().padStart(2, '0')} (${item.totalSeconds}s)`);
    });

    return {
      expanded_items: expandedItems
    };
  }
}
