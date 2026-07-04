/**
 * BM25 — from scratch.
 *
 * Why keep lexical search when we have embeddings? They fail
 * differently. Embeddings match *meaning* ("laptop won't turn on" ↔
 * "computer fails to boot") but blur exact identifiers. BM25 matches
 * *terms* — it nails product codes, function names, error strings —
 * but knows nothing about synonyms. Hybrid retrieval (see hybrid.ts)
 * fuses both.
 *
 * BM25 score of a document D for query terms q1..qn:
 *
 *   Σ IDF(qi) · (tf · (k1 + 1)) / (tf + k1 · (1 - b + b · |D|/avgLen))
 *
 *   IDF: rare terms count more than common ones.
 *   tf saturation (k1): the 10th occurrence of a term adds less than the 2nd.
 *   length norm (b): long documents don't win just by containing everything.
 */
import type { Chunk, ScoredChunk } from "../types.js";

const K1 = 1.2;
const B = 0.75;

function tokenize(text: string): string[] {
  return text
    .toLowerCase()
    .split(/[^a-z0-9]+/)
    .filter((t) => t.length > 1);
}

interface IndexedDoc {
  chunk: Chunk;
  termFreq: Map<string, number>;
  length: number;
}

export class BM25Index {
  private docs: IndexedDoc[] = [];
  private docFreq = new Map<string, number>(); // term -> # docs containing it
  private avgLength = 0;

  /** Build from all chunks at once (cheap enough to rebuild on every load). */
  static build(chunks: Chunk[]): BM25Index {
    const index = new BM25Index();
    let totalLength = 0;
    for (const chunk of chunks) {
      const tokens = tokenize(chunk.text);
      const termFreq = new Map<string, number>();
      for (const t of tokens) termFreq.set(t, (termFreq.get(t) ?? 0) + 1);
      for (const term of termFreq.keys()) {
        index.docFreq.set(term, (index.docFreq.get(term) ?? 0) + 1);
      }
      index.docs.push({ chunk, termFreq, length: tokens.length });
      totalLength += tokens.length;
    }
    index.avgLength = index.docs.length ? totalLength / index.docs.length : 0;
    return index;
  }

  search(query: string, k: number): ScoredChunk[] {
    const queryTerms = tokenize(query);
    const n = this.docs.length;

    const scored: ScoredChunk[] = this.docs.map((doc) => {
      let score = 0;
      for (const term of queryTerms) {
        const tf = doc.termFreq.get(term);
        if (!tf) continue;
        const df = this.docFreq.get(term)!;
        // BM25+ style IDF, floored at 0 so ultra-common terms can't go negative.
        const idf = Math.max(0, Math.log((n - df + 0.5) / (df + 0.5) + 1));
        const norm = 1 - B + B * (doc.length / this.avgLength);
        score += idf * ((tf * (K1 + 1)) / (tf + K1 * norm));
      }
      return { ...doc.chunk, score };
    });

    return scored
      .filter((s) => s.score > 0)
      .sort((a, b) => b.score - a.score)
      .slice(0, k);
  }
}
