/**
 * Retriever — one loaded index, three search strategies.
 *
 * Exposing vector-only, BM25-only, and hybrid side by side is
 * deliberate: the evaluation harness compares them head-to-head, which
 * is how you *learn* what each retriever contributes instead of taking
 * "hybrid is better" on faith.
 */
import { embedQuery } from "./embeddings/embedder.js";
import { VectorIndex } from "./index/vector.js";
import { BM25Index } from "./index/bm25.js";
import { reciprocalRankFusion } from "./index/hybrid.js";
import { rerank } from "./rerank/crossEncoder.js";
import type { ScoredChunk } from "./types.js";

/** How many candidates each retriever contributes before fusion. */
export const CANDIDATES_PER_RETRIEVER = 20;

export class Retriever {
  private constructor(
    private vectorIndex: VectorIndex,
    private bm25Index: BM25Index,
  ) {}

  static async load(indexPath: string): Promise<Retriever> {
    const vectorIndex = await VectorIndex.load(indexPath);
    // BM25 is rebuilt from the stored chunks — cheap at this scale,
    // and keeps the persisted format to a single file.
    const bm25Index = BM25Index.build(vectorIndex.chunks);
    return new Retriever(vectorIndex, bm25Index);
  }

  async vector(query: string, k: number): Promise<ScoredChunk[]> {
    return this.vectorIndex.search(await embedQuery(query), k);
  }

  bm25(query: string, k: number): ScoredChunk[] {
    return this.bm25Index.search(query, k);
  }

  async hybrid(query: string, k: number): Promise<ScoredChunk[]> {
    const [vectorResults, bm25Results] = await Promise.all([
      this.vector(query, CANDIDATES_PER_RETRIEVER),
      Promise.resolve(this.bm25(query, CANDIDATES_PER_RETRIEVER)),
    ]);
    return reciprocalRankFusion([vectorResults, bm25Results], k);
  }

  /**
   * Two-stage retrieval: hybrid recall over the whole index, then a
   * cross-encoder re-scores the fused top candidates jointly with the
   * query. Stage 1 optimizes "don't miss it"; stage 2 optimizes
   * "put the best one first".
   */
  async hybridReranked(query: string, k: number): Promise<ScoredChunk[]> {
    const candidates = await this.hybrid(query, CANDIDATES_PER_RETRIEVER);
    return rerank(query, candidates, k);
  }
}
