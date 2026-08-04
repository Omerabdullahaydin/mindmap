import { AzureResponsesChatModel } from "./azureResponsesClient.js";

/**
 * Azure OpenAI (Responses API) chat modeli döndürür. Projedeki TÜM LLM
 * çağrıları (mapNode, reduceNode, expandItemsNode, citation seçimi) bu tek
 * fonksiyonu kullanır.
 *
 * Kimlik bilgileri kod içinde YOK — azureResponsesClient.ts bunları çalışma
 * anında şu ortam değişkenlerinden okur (bkz. _run/.env.example):
 *   AZURE_OPENAI_RESPONSES_URL, AZURE_OPENAI_API_KEY, AZURE_OPENAI_MODEL
 *
 * maxTokens yüksek tutuluyor (2048) çünkü düşük bırakılırsa uzun cevaplar
 * (mindmap, özet, senaryo) yarıda kesilebiliyor.
 */
export function getAzureChatModel(options: { temperature?: number; maxTokens?: number } = {}): AzureResponsesChatModel {
  return new AzureResponsesChatModel({
    temperature: options.temperature ?? 0.3,
    maxTokens: options.maxTokens ?? 2048
  });
}
