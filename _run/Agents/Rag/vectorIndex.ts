// ============================================================================
// VectorIndex: embedding vektörleri üzerinde basit, bellek-içi cosine
// similarity araması. Azure AI Search gibi harici bir vektör deposu yerine —
// BM25Index (bkz. bm25.ts) ile aynı mantıkta, projenin tamamı zaten bellek-içi
// çalıştığı için (rag_memory2.ts'teki localDocuments) buna gerek yok.
// ============================================================================

interface IndexedVector<T> {
  item: T;
  vector: number[];
}

export interface ScoredVectorResult<T> {
  item: T;
  score: number;
}

function cosineSimilarity(a: number[], b: number[]): number {
  let dot = 0, normA = 0, normB = 0;
  for (let i = 0; i < a.length; i++) {
    dot += a[i] * b[i];
    normA += a[i] * a[i];
    normB += b[i] * b[i];
  }
  if (normA === 0 || normB === 0) return 0;
  return dot / (Math.sqrt(normA) * Math.sqrt(normB));
}

export class VectorIndex<T> {
  private docs: IndexedVector<T>[];

  constructor(items: T[], vectors: number[][]) {
    if (items.length !== vectors.length) {
      throw new Error("VectorIndex: items ve vectors uzunlukları eşleşmiyor");
    }
    this.docs = items.map((item, i) => ({ item, vector: vectors[i] }));
  }

  // search: sorgu vektörüne en yakın (cosine similarity) "k" öğeyi döndürür.
  search(queryVector: number[], k: number): ScoredVectorResult<T>[] {
    return this.docs
      .map(doc => ({ item: doc.item, score: cosineSimilarity(queryVector, doc.vector) }))
      .sort((a, b) => b.score - a.score)
      .slice(0, k);
  }
}
