/**
 * Vector index — from scratch.
 *
 * This is brute-force exact nearest-neighbor search: score the query
 * against EVERY stored vector, sort, take top-k. O(n·d) per query.
 *
 * That sounds naive, but for < ~100k chunks it's often *faster* than an
 * approximate index once you count index-build time — and it's exact.
 * Milestone 4 replaces this with HNSW (a navigable small-world graph)
 * to learn how approximate search trades recall for speed.
 *
 * Because embeddings are normalized (see embedder.ts), cosine
 * similarity == dot product, so scoring is just a multiply-accumulate.
 */
import fs from "node:fs/promises";
import path from "node:path";
import type { Chunk, ScoredChunk } from "../types.js";

interface Entry {
  chunk: Chunk;
  vector: number[];
}

export class VectorIndex {
  private entries: Entry[] = [];

  get size(): number {
    return this.entries.length;
  }

  get chunks(): Chunk[] {
    return this.entries.map((e) => e.chunk);
  }

  add(chunk: Chunk, vector: number[]): void {
    this.entries.push({ chunk, vector });
  }

  /** Exact top-k search by dot product (== cosine, vectors are normalized). */
  search(queryVector: number[], k: number): ScoredChunk[] {
    const scored: ScoredChunk[] = this.entries.map(({ chunk, vector }) => {
      let dot = 0;
      for (let i = 0; i < vector.length; i++) dot += vector[i] * queryVector[i];
      return { ...chunk, score: dot };
    });
    scored.sort((a, b) => b.score - a.score);
    return scored.slice(0, k);
  }

  /** Persist to disk as JSON. Fine at learning scale; binary format is a later milestone. */
  async save(filePath: string): Promise<void> {
    await fs.mkdir(path.dirname(filePath), { recursive: true });
    await fs.writeFile(filePath, JSON.stringify(this.entries));
  }

  static async load(filePath: string): Promise<VectorIndex> {
    const index = new VectorIndex();
    index.entries = JSON.parse(await fs.readFile(filePath, "utf-8"));
    return index;
  }
}
