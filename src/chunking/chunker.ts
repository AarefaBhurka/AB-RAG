/**
 * Structure-aware chunking.
 *
 * Why chunk at all? Embeddings represent a *whole* text as one vector —
 * embed a 50-page document and the vector is a mushy average that matches
 * nothing well. Chunks small enough to be "about one thing" retrieve
 * precisely; chunks large enough to carry context answer well. The
 * classic tension.
 *
 * Strategy here (deliberately simple, easy to iterate on):
 *   1. Split on paragraph boundaries (blank lines) — never mid-sentence.
 *   2. Greedily pack paragraphs into chunks up to `maxChars`.
 *   3. Overlap: each chunk starts with the tail of the previous one, so
 *      facts that straddle a boundary are retrievable from either side.
 *
 * Things to experiment with later: heading-based splitting for markdown,
 * sentence-level packing, semantic chunking (split where embedding
 * similarity between adjacent windows drops).
 */
import type { Chunk, RawDocument } from "../types.js";

export interface ChunkOptions {
  /** Target maximum chunk size, in characters (~4 chars ≈ 1 token). */
  maxChars?: number;
  /** How much of the previous chunk's tail to repeat at the start of the next. */
  overlapChars?: number;
}

export function chunkDocument(
  doc: RawDocument,
  { maxChars = 1500, overlapChars = 200 }: ChunkOptions = {},
): Chunk[] {
  const paragraphs = doc.text
    .split(/\n\s*\n/)
    .map((p) => p.trim())
    .filter((p) => p.length > 0);

  const chunks: Chunk[] = [];
  let current: string[] = [];
  let currentLen = 0;

  const flush = () => {
    if (currentLen === 0) return;
    const text = current.join("\n\n");
    chunks.push({
      id: `${doc.source}#${chunks.length}`,
      source: doc.source,
      index: chunks.length,
      text,
    });
    // Seed the next chunk with the tail of this one (overlap).
    const tail = text.slice(-overlapChars);
    current = tail ? [tail] : [];
    currentLen = tail.length;
  };

  for (const para of paragraphs) {
    // A single paragraph larger than maxChars gets hard-split.
    if (para.length > maxChars) {
      flush();
      for (let i = 0; i < para.length; i += maxChars - overlapChars) {
        current = [para.slice(i, i + maxChars)];
        currentLen = current[0].length;
        flush();
      }
      continue;
    }
    if (currentLen + para.length > maxChars) flush();
    current.push(para);
    currentLen += para.length + 2;
  }
  flush();

  return chunks;
}
