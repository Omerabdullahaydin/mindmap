// ============================================================================
// Reciprocal Rank Fusion (RRF): Birden fazla ayrı arama sonucunu (BM25,
// embedding, farklı sorgu varyantları) TEK bir sıralı listede birleştirmenin
// standart yolu. Fikir: her listedeki SIRA (rank) önemlidir, ham skor değil —
// BM25 skoru ile cosine similarity skoru zaten karşılaştırılabilir birimlerde
// değil (biri sınırsız, diğeri -1..1), o yüzden skorları toplamak yerine
// sırayı kullanıyoruz. k=60 sabiti, orijinal RRF makalesinde (Cormack et al.,
// 2009) önerilen ve pratikte iyi çalışan varsayılan değer.
// ============================================================================
import type { Document } from "@langchain/core/documents";

export function reciprocalRankFusion(resultLists: Document[][], k = 60): Document[] {
  const scores = new Map<string, number>();
  const docByKey = new Map<string, Document>();

  for (const list of resultLists) {
    list.forEach((doc, rank) => {
      const key = doc.pageContent;
      scores.set(key, (scores.get(key) || 0) + 1 / (k + rank + 1));
      if (!docByKey.has(key)) docByKey.set(key, doc);
    });
  }

  return Array.from(scores.entries())
    .sort((a, b) => b[1] - a[1])
    .map(([key]) => docByKey.get(key)!);
}
