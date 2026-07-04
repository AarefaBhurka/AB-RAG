/**
 * Retrieval metrics — the two workhorses of RAG evaluation.
 *
 * Both operate on the RANK of the first relevant result for each
 * question (1-based; undefined if nothing relevant was retrieved).
 *
 * Hit-rate@k ("did we find it at all?"):
 *   fraction of questions where a relevant chunk appears in the top k.
 *   This is the ceiling on your whole RAG system — if the answer isn't
 *   in the context window, no model can produce it.
 *
 * MRR — Mean Reciprocal Rank ("how high did we find it?"):
 *   average of 1/rank. Rank 1 → 1.0, rank 3 → 0.33, missed → 0.
 *   More sensitive than hit-rate: it rewards putting the right chunk
 *   first, which matters because models attend more to earlier context.
 */

/** Rank (1-based) of the first result whose source is in `expected`, else undefined. */
export function firstRelevantRank(
  retrievedSources: string[],
  expectedSources: string[],
): number | undefined {
  const expected = new Set(expectedSources);
  const index = retrievedSources.findIndex((s) => expected.has(s));
  return index === -1 ? undefined : index + 1;
}

export function hitRateAtK(ranks: (number | undefined)[], k: number): number {
  if (ranks.length === 0) return 0;
  const hits = ranks.filter((r) => r !== undefined && r <= k).length;
  return hits / ranks.length;
}

export function meanReciprocalRank(ranks: (number | undefined)[]): number {
  if (ranks.length === 0) return 0;
  const sum = ranks.reduce<number>((acc, r) => acc + (r ? 1 / r : 0), 0);
  return sum / ranks.length;
}
