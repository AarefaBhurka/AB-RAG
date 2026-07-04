/**
 * Core data shapes shared by every stage of the pipeline.
 *
 * A RAG pipeline is a series of transformations:
 *   RawDocument → Chunk[] → (Chunk + embedding)[] → ScoredChunk[] → answer
 */

/** A loaded source file, before chunking. */
export interface RawDocument {
  /** Where this came from (file path, URL, ...). Used for citations. */
  source: string;
  text: string;
}

/** A retrievable unit of text. The atom of a RAG system. */
export interface Chunk {
  /** Stable id: `${source}#${index}` */
  id: string;
  source: string;
  /** Position of this chunk within its source document. */
  index: number;
  text: string;
}

/** A chunk with a retrieval score attached (higher = more relevant). */
export interface ScoredChunk extends Chunk {
  score: number;
}
