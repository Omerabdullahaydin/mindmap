// ============================================================================
// BU DOSYA NE İŞE YARAR?
// ----------------------------------------------------------------------------
// Kullanıcının Azure kaynağı, klasik "Azure OpenAI" (instance name + deployment
// name + chat completions) şeklinde DEĞİL — Azure AI Foundry üzerinden deploy
// edilmiş bir modelin "Responses API" uç noktası şeklinde geliyor:
//
//   https://<kaynak>.cognitiveservices.azure.com/openai/responses?api-version=...
//
// Bu, LangChain'in @langchain/openai paketindeki AzureChatOpenAI sınıfının
// konuştuğu protokolden (Chat Completions) FARKLI bir istek/cevap şekli
// kullanır. Kurulu LangChain sürümü bu API'yi desteklemediği için, burada
// TEK bir fetch() çağrısıyla konuşan, minimal bir istemci yazıyoruz.
//
// Projenin geri kalanı (mapNode, reduceNode, expandItemsNode, citation seçimi,
// generate_mindmap_llm.ts) hep aynı basit arayüzü kullanıyor:
//   const llm = getAzureChatModel();
//   const result = await llm.invoke("bir prompt metni");
//   result.content // -> düz metin cevap
// Bu, önceden ChatOllama/AzureChatOpenAI'nin de sağladığı arayüzle AYNI
// şekilde çalışıyor, böylece çağıran taraflarda başka hiçbir değişiklik
// gerekmiyor.
// ============================================================================

export interface AzureLLMResult {
  content: string;
}

export interface AzureChatModelOptions {
  temperature?: number;
  maxTokens?: number;
}

export class AzureResponsesChatModel {
  constructor(private options: AzureChatModelOptions = {}) {}

  async invoke(input: string): Promise<AzureLLMResult> {
    const url = process.env.AZURE_OPENAI_RESPONSES_URL;
    const apiKey = process.env.AZURE_OPENAI_API_KEY;
    const model = process.env.AZURE_OPENAI_MODEL;

    if (!url || !apiKey || !model) {
      throw new Error(
        "Azure OpenAI ortam değişkenleri eksik: AZURE_OPENAI_RESPONSES_URL, " +
        "AZURE_OPENAI_API_KEY, AZURE_OPENAI_MODEL (.env dosyasına bakın, " +
        "örnek için .env.example)."
      );
    }

    const body: Record<string, unknown> = { model, input };
    if (this.options.temperature !== undefined) body.temperature = this.options.temperature;
    if (this.options.maxTokens !== undefined) body.max_output_tokens = this.options.maxTokens;

    const res = await fetch(url, {
      method: "POST",
      headers: { "Content-Type": "application/json", "api-key": apiKey },
      body: JSON.stringify(body)
    });

    if (!res.ok) {
      const errText = await res.text();
      throw new Error(`Azure OpenAI Responses API hatası (HTTP ${res.status}): ${errText.slice(0, 500)}`);
    }

    const data = await res.json();
    return { content: extractResponseText(data) };
  }
}

// extractResponseText: Responses API cevabı, "output" adında bir dizi
// döndürür (reasoning modellerde araya "reasoning" tipi öğeler de girebilir).
// Biz sadece type==="message" olan öğelerin, type==="output_text" olan
// content parçalarındaki metni birleştirip döndürüyoruz.
function extractResponseText(data: any): string {
  if (!data || !Array.isArray(data.output)) return "";
  let text = "";
  for (const item of data.output) {
    if (item?.type === "message" && Array.isArray(item.content)) {
      for (const part of item.content) {
        if (part?.type === "output_text" && typeof part.text === "string") {
          text += part.text;
        }
      }
    }
  }
  return text;
}
