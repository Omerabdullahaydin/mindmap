// ============================================================================
// groupCitationsByMadde: "citation'ları hangi mindmap maddesine ait olduğuna
// göre grupla" mantığı hem rag_tool_citation.ts'te (kaynak rozeti eklerken)
// hem mcp/mindmap_visualization_server.ts'te (tıklama modalı için veri
// hazırlarken) ayrı ayrı yazılmıştı — ikisi de aynı "madde -> citation[]"
// Map'ini kuruyordu, sadece küçük farklarla (badge temizleme, eksik madde
// alanını atlama). Bu paylaşılan yardımcı, o iki farkı opsiyonel parametre
// olarak alıp tek bir yerde tutuyor.
// ============================================================================

export interface GroupableCitation {
  madde?: string;
  [key: string]: any;
}

export interface GroupCitationsOptions {
  // true ise, gruplamadan önce madde metnindeki "[Kaynaklar: ...]" rozetini
  // temizler (mindmap_visualization_server.ts'in ihtiyacı — rag_tool_citation.ts'te
  // madde metninde zaten hiç rozet olmuyor, o yüzden orada gerekmiyor).
  stripBadge?: boolean;
  // true ise, "madde" alanı eksik olan citation'lar için konsola uyarı basar
  // (mindmap_visualization_server.ts'teki eski bug-fix davranışı).
  warnOnMissing?: boolean;
}

export function groupCitationsByMadde<T extends GroupableCitation>(
  citations: T[],
  options: GroupCitationsOptions = {}
): Map<string, T[]> {
  const { stripBadge = false, warnOnMissing = false } = options;
  const grouped = new Map<string, T[]>();

  citations.forEach(citation => {
    if (!citation?.madde) {
      if (warnOnMissing) {
        console.warn("Citation 'madde' alanı eksik, atlanıyor:", citation);
      }
      return;
    }

    const key = stripBadge ? citation.madde.replace(/\[Kaynaklar:.*?\]/g, '').trim() : citation.madde;

    if (!grouped.has(key)) {
      grouped.set(key, []);
    }
    grouped.get(key)!.push(citation);
  });

  return grouped;
}
