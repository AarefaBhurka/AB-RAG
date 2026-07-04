/**
 * Hybrid retrieval via Reciprocal Rank Fusion (RRF).
 *
 * Problem: vector scores (cosine, ~0.2–0.9) and BM25 scores (unbounded)
 * live on incompatible scales — you can't just add them. RRF sidesteps
 * this by ignoring scores entirely and fusing on *rank*:
 *
 *   RRF(chunk) = Σ over result lists  1 / (K + rank_in_list)
 *
 * K=60 (the value from the original paper) dampens the advantage of
 * being #1 vs #3, so a chunk that's decent in BOTH lists beats one
 * that's #1 in one list and absent from the other. That's exactly the
 * behavior you want from hybrid search.
 */
import type { ScoredChunk } from "../types.js";

const RRF_K = 60;

export function reciprocalRankFusion(
  resultLists: ScoredChunk[][],
  topK: number,
): ScoredChunk[] {
  const fused = new Map<string, ScoredChunk>();

  for (const list of resultLists) {
    list.forEach((chunk, rank) => {
      const contribution = 1 / (RRF_K + rank + 1);
      const existing = fused.get(chunk.id);
      if (existing) {
        existing.score += contribution;
      } else {
        fused.set(chunk.id, { ...chunk, score: contribution });
      }
    });
  }

  return [...fused.values()].sort((a, b) => b.score - a.score).slice(0, topK);
}
