// ============================================================================
// BM25: İYİ BİLİNEN, embedding GEREKTİRMEYEN bir "lexical" (kelime tabanlı)
// sıralama (ranking) fonksiyonu. Klasik arama motorlarının (Elasticsearch,
// Lucene vb.) varsayılan sıralama algoritmasıdır.
// ----------------------------------------------------------------------------
// Önceki "kaç kelime ortak?" (naif kelime örtüşmesi) yaklaşımından farkı:
//   1) IDF (Inverse Document Frequency): NADİR geçen kelimelere (ör. "SPSS",
//      "Kolmogorov-Smirnov") daha çok ağırlık verir; HER paragrafta geçen
//      genel kelimelere ("çalışma", "sonuç" gibi) daha az ağırlık verir.
//      Naif sayım bunu hiç ayırt etmiyordu.
//   2) Terim frekansı DOYGUNLUĞU (saturation): bir kelime bir paragrafta 1 kez
//      mi 10 kez mi geçiyor farkı, doğrusal değil AZALAN bir katkı sağlar
//      (10. tekrar, 2. tekrar kadar "değerli" değildir).
//   3) Doküman uzunluğu normalizasyonu: sırf UZUN olduğu için daha çok
//      kelime içeren paragraflar haksız yere öne çıkmaz.
// ============================================================================

interface IndexedDoc<T> {
  item: T;
  docLength: number; // tokens.length — doküman uzunluğu normalizasyonu için lazım, token dizisinin kendisi hiç okunmuyor
  termFreq: Map<string, number>;
}

export interface ScoredResult<T> {
  item: T;
  score: number;
}

export class BM25Index<T> {
  private docs: IndexedDoc<T>[] = [];
  private documentFrequency = new Map<string, number>(); // Bir terim kaç FARKLI dokümanda geçiyor
  private avgDocLength = 0;

  // k1: terim frekansı doygunluğunu kontrol eder (tipik: 1.2-2.0)
  // b: doküman uzunluğu normalizasyonunun gücü (tipik: 0.75)
  constructor(
    items: T[],
    getText: (item: T) => string,
    private tokenize: (text: string) => string[],
    private k1 = 1.5,
    private b = 0.75
  ) {
    let totalLength = 0;
    for (const item of items) {
      const tokens = tokenize(getText(item));
      const termFreq = new Map<string, number>();
      for (const t of tokens) termFreq.set(t, (termFreq.get(t) || 0) + 1);

      this.docs.push({ item, docLength: tokens.length, termFreq });
      totalLength += tokens.length;

      for (const t of termFreq.keys()) {
        this.documentFrequency.set(t, (this.documentFrequency.get(t) || 0) + 1);
      }
    }
    this.avgDocLength = items.length > 0 ? totalLength / items.length : 0;
  }

  // search: sorgu metnini tokenize edip her doküman için BM25 skorunu
  // hesaplar, en yüksek skorlu "k" tanesini döndürür. Skoru 0 olanlar
  // (yani sorguyla HİÇ ortak kelimesi olmayanlar) elenir.
  search(query: string, k: number): ScoredResult<T>[] {
    const queryTokens = this.tokenize(query);
    const N = this.docs.length;
    if (N === 0 || queryTokens.length === 0) return [];

    const scored = this.docs.map(doc => {
      let score = 0;
      const docLen = doc.docLength;

      for (const qt of queryTokens) {
        const tf = doc.termFreq.get(qt) || 0;
        if (tf === 0) continue;

        const df = this.documentFrequency.get(qt) || 0;
        const idf = Math.log((N - df + 0.5) / (df + 0.5) + 1);

        const numerator = tf * (this.k1 + 1);
        const denominator = tf + this.k1 * (1 - this.b + this.b * (docLen / (this.avgDocLength || 1)));
        score += idf * (numerator / denominator);
      }

      return { item: doc.item, score };
    });

    return scored
      .filter(s => s.score > 0)
      .sort((a, b) => b.score - a.score)
      .slice(0, k);
  }
}
