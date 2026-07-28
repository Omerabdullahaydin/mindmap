// ============================================================================
// BU DOSYA NE İŞE YARAR?
// ----------------------------------------------------------------------------
// MCP (Model Context Protocol), araçları (tool'ları) AYRI bir sunucu süreci
// olarak çalıştırıp, oraya ağ üzerinden (bu durumda SSE - Server-Sent Events -
// ile) bağlanmayı sağlayan bir protokoldür. Yani mindmap HTML'ini üreten kod,
// bu process içinde değil, BAŞKA bir sunucu process'inde çalışabilir; bu
// dosya da o uzak sunucuya bağlanıp "şu mindmap'i üret" diye istek göndermeyi
// sağlayan bir "istemci" (client) sınıfıdır.
//
// ÖNEMLİ: Şu anki kurulumda (USE_MCP ortam değişkeni "true" YAPILMADIĞI
// sürece) bu sınıf KULLANILMIYOR — mindmap_citation_v2.ts'teki toolNode,
// varsayılan olarak "doğrudan yöntemi" kullanıyor (yani HTML'i üreten
// fonksiyonu aynı process içinde direkt çağırıyor, ağ üzerinden hiçbir yere
// bağlanmadan). Bu dosya, sadece USE_MCP=true yapılıp ayrı bir MCP sunucusu
// çalıştırıldığında devreye giriyor (bkz. mindmap_citation_v2.ts'in en
// altındaki "Auto-start MCP server" bloğu).
//
// Neden iki yöntem var? MCP yöntemi, mindmap üretme mantığını AYRI bir
// sunucuda çalıştırmayı sağlar (ör. başka programların/ajanların da aynı
// sunucuya bağlanıp aynı aracı kullanabilmesi için). Basit/tek kullanıcılı
// bir kurulumda buna gerek yoktur, o yüzden varsayılan "doğrudan yöntem"dir.
// ============================================================================
import { Client } from '@modelcontextprotocol/sdk/client/index.js';        // MCP protokolünü konuşan genel istemci sınıfı
import { SSEClientTransport } from '@modelcontextprotocol/sdk/client/sse.js'; // Bağlantı yöntemi: Server-Sent Events (tek yönlü, sürekli açık bir HTTP bağlantısı)
import { CallToolRequestSchema } from '@modelcontextprotocol/sdk/types.js';   // "Bir tool'u çağır" isteğinin standart şekli (formatı)

export class MindmapMCPClient {
  private client: Client;         // Asıl MCP bağlantısını yöneten iç obje
  private sessionId: string;      // Bu oturuma özel kimlik (loglarda hangi isteğin hangi oturuma ait olduğunu ayırt etmek için)
  private serverUrl: string;      // MCP sunucusunun adresi (ör. http://localhost:3001)
  private connected: boolean = false; // Şu an bağlı mıyız? (connect() çağrılmadan istek atmayı engellemek için)

  constructor(sessionId: string, serverUrl?: string) {
    this.sessionId = sessionId;
    // Sunucu adresi verilmemişse, önce MCP_SERVER_URL ortam değişkenine,
    // o da yoksa varsayılan olarak localhost:3001'e bak.
    this.serverUrl = serverUrl || process.env.MCP_SERVER_URL || 'http://localhost:3001';

    // MCP istemcisini oluştur. "capabilities: {}" -> bu istemcinin sunucuya
    // sunduğu ekstra özellik yok, sadece standart tool çağırma işini yapacak.
    this.client = new Client(
      {
        name: `mindmap-agent-${sessionId}`,
        version: "1.0.0"
      },
      {
        capabilities: {}
      }
    );
  }

  // connect: MCP sunucusuna GERÇEKTEN bağlantıyı açar. visualizeMindmap()
  // çağrılmadan önce bu fonksiyonun çalışmış (ve başarılı olmuş) olması gerekir.
  async connect(): Promise<void> {
    if (this.connected) {
      console.log(`⚠️ MCP Client already connected (session: ${this.sessionId})`);
      return; // Zaten bağlıysak tekrar bağlanmaya çalışma
    }

    try {
      console.log(`🔌 Connecting to MCP server at ${this.serverUrl}...`);
      // SSE (Server-Sent Events): sunucunun "/mcp/sse" adresine sürekli açık
      // bir bağlantı kurulur, sunucu bu bağlantı üzerinden istemciye mesaj
      // gönderebilir (tek yönlü, sunucudan istemciye canlı veri akışı).
      const transport = new SSEClientTransport(
        new URL(`${this.serverUrl}/mcp/sse`)
      );
      await this.client.connect(transport);
      this.connected = true;
      console.log(`✅ MCP Client connected (session: ${this.sessionId})`);
    } catch (error: any) {
      // Bağlantı kurulamazsa (ör. MCP sunucusu çalışmıyorsa), anlaşılır bir
      // hata fırlat. Bu hata, çağıran taraf (mindmap_citation_v2.ts'teki
      // toolNode) tarafından yakalanıp "doğrudan yönteme" geri dönmek için kullanılıyor.
      console.error(`❌ MCP Client connection failed:`, error.message);
      throw new Error(`Failed to connect to MCP server: ${error.message}`);
    }
  }

  // visualizeMindmap: Bağlı olduğumuz MCP sunucusuna "şu markdown'ı ve
  // kaynakları al, bana bir mindmap HTML'i üret" diye istek gönderir.
  // Bu, mindmap_visualization_server.ts'teki createLiveMindmapHTML
  // fonksiyonunu UZAKTAN (ağ üzerinden) çağırmakla aynı sonucu verir —
  // tek fark, kodun BAŞKA bir process'te çalışmasıdır.
  async visualizeMindmap(
    markdownContent: string,
    citations: any[] = []
  ): Promise<{
    success: boolean;
    html_path?: string;
    error?: string;
    session_id?: string;
    timestamp?: number;
    citations_count?: number;
  }> {
    if (!this.connected) {
      throw new Error('MCP Client not connected. Call connect() first.');
    }

    try {
      console.log(`📊 Requesting mindmap visualization via MCP...`);

      // MCP protokolünün standart isteği: "tools/call" metodu, hangi tool'un
      // ("visualize_mindmap") ve hangi parametrelerle çağrılacağını belirtir.
      // Bu, mindmap_visualization_server.ts'teki getMindmapVisualizationTool()
      // fonksiyonunun tanımladığı tool ile AYNI isim ve parametrelere sahip
      // olmalı — MCP sunucusu bu isteği alıp o fonksiyonu çalıştırır.
      const result = await this.client.request(
        {
          method: "tools/call",
          params: {
            name: "visualize_mindmap",
            arguments: {
              markdown_content: markdownContent,
              citations: citations,
              session_id: this.sessionId
            }
          }
        },
        CallToolRequestSchema
      );

      // MCP sunucusu cevabı, standart bir "content" dizisi içinde, metin
      // (text) olarak döndürür. Bu metin aslında bir JSON string'idir,
      // bu yüzden JSON.parse ile gerçek objeye çeviriyoruz.
      const content = (result as any).content[0].text;
      const parsed = JSON.parse(content);

      console.log(`✅ MCP visualization result:`, parsed);

      return parsed;
    } catch (error: any) {
      // İstek sırasında bir hata olursa (ör. sunucu tool'u çalıştırırken
      // patladıysa), programı çökertmek yerine "success: false" içeren
      // bir sonuç döndür — çağıran taraf bunu kontrol edip fallback yapabilir.
      console.error(`❌ MCP visualization error:`, error.message);
      return {
        success: false,
        error: error.message
      };
    }
  }

  // close: MCP bağlantısını düzgünce kapat (kaynak sızıntısı olmasın diye).
  async close(): Promise<void> {
    if (!this.connected) {
      return; // Zaten bağlı değilsek kapatacak bir şey yok
    }

    try {
      await this.client.close();
      this.connected = false;
      console.log(`🔌 MCP Client disconnected (session: ${this.sessionId})`);
    } catch (error: any) {
      console.error(`⚠️ Error disconnecting MCP client:`, error.message);
    }
  }
}
