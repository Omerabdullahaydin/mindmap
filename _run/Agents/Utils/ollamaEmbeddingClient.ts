// ============================================================================
// BU DOSYA NE İŞE YARAR?
// ----------------------------------------------------------------------------
// Yerelde (localhost) çalışan Ollama üzerinden embedding hesaplayan minimal
// bir istemci. Kullanıcının Azure kaynağında embedding deployment'ı
// olmadığı için (bkz. azureResponsesClient.ts), embedding hesaplamasını
// yerel bir modele devrediyoruz — chat modeli hâlâ Azure'dan, embedding
// modeli yerelden (Ollama) geliyor; RAG hattının bu ikisini karıştırmasında
// bir sorun yok, ikisi de sadece "metin -> ..." arayüzü sağlıyor.
//
// Varsayılan model "nomic-embed-text" (768 boyutlu, hafif, çok dilli metinde
// makul performans). Farklı bir model kullanmak isterseniz .env'e
// OLLAMA_EMBEDDING_MODEL=... ekleyip önce "ollama pull <model>" ile indirin.
// ============================================================================

const DEFAULT_OLLAMA_URL = "http://localhost:11434";
const DEFAULT_MODEL = "nomic-embed-text";

export class OllamaEmbeddingClient {
  private baseUrl: string;
  private model: string;

  constructor(options: { baseUrl?: string; model?: string } = {}) {
    this.baseUrl = options.baseUrl ?? process.env.OLLAMA_URL ?? DEFAULT_OLLAMA_URL;
    this.model = options.model ?? process.env.OLLAMA_EMBEDDING_MODEL ?? DEFAULT_MODEL;
  }

  // embed: Birden fazla metni TEK istekte embedding'e çevirir (Ollama'nın
  // /api/embed uç noktası batch girişi destekliyor — bu, yüzlerce paragrafı
  // tek tek istemekten çok daha hızlı).
  async embed(texts: string[]): Promise<number[][]> {
    if (texts.length === 0) return [];

    const res = await fetch(`${this.baseUrl}/api/embed`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ model: this.model, input: texts })
    });

    if (!res.ok) {
      const errText = await res.text();
      throw new Error(`Ollama embedding hatası (HTTP ${res.status}): ${errText.slice(0, 500)}`);
    }

    const data = await res.json();
    if (!Array.isArray(data.embeddings)) {
      throw new Error("Ollama embedding cevabında 'embeddings' dizisi bulunamadı");
    }
    return data.embeddings;
  }

  // embedOne: Tek bir metin için kısayol (sorgu embedding'i için kullanışlı)
  async embedOne(text: string): Promise<number[]> {
    const [vector] = await this.embed([text]);
    return vector;
  }
}
