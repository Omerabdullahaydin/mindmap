import { ChatOllama } from "@langchain/ollama";

/**
 * Yerel Ollama chat modeli döndürür (Azure OpenAI'nin yerini alıyor).
 * Kullanıcının Azure erişimi olmadığı için, projedeki TÜM LLM çağrıları
 * (mapNode, reduceNode, expandItemsNode, citation seçimi) bu tek fonksiyonu
 * kullanır. Model adı OLLAMA_CHAT_MODEL ortam değişkeniyle değiştirilebilir.
 *
 * numPredict yüksek tutuluyor (2048) çünkü düşük bırakılırsa (Ollama'nın
 * varsayılanı çok düşük, ör. 128) uzun cevaplar (mindmap, özet, senaryo)
 * yarıda kesilebiliyor - bu, "gemma4" thinking modeliyle ilk denememizde
 * gerçekten yaşadığımız bir sorundu.
 */
export function getOllamaChatModel(options: { temperature?: number; numPredict?: number } = {}): ChatOllama {
  return new ChatOllama({
    model: process.env.OLLAMA_CHAT_MODEL || "qwen2.5:7b",
    temperature: options.temperature ?? 0.3,
    numPredict: options.numPredict ?? 2048
  });
}
