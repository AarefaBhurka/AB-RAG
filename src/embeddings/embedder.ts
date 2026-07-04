/**
 * Embeddings — turn text into vectors so "similar meaning" becomes
 * "nearby in space".
 *
 * This runs a small sentence-transformer (all-MiniLM-L6-v2, 384 dims)
 * locally via transformers.js. First run downloads ~25MB of weights to
 * ~/.cache; after that it's fully offline and free. No API key needed
 * for the entire indexing side of the pipeline.
 *
 * Vectors are L2-normalized at the source, so cosine similarity
 * reduces to a plain dot product in the vector index.
 *
 * To swap in a different embedding model later (a larger local model,
 * or a hosted one), implement this same function signature against it —
 * nothing else in the pipeline changes.
 */
import { pipeline } from "@huggingface/transformers";

const MODEL = "Xenova/all-MiniLM-L6-v2";
const BATCH_SIZE = 32;

// Lazy singleton — model loads once per process.
let extractorPromise: Promise<any> | undefined;

async function getExtractor() {
  extractorPromise ??= pipeline("feature-extraction", MODEL);
  return extractorPromise;
}

/** Embed a batch of texts into normalized vectors. */
export async function embed(texts: string[]): Promise<number[][]> {
  const extractor = await getExtractor();
  const vectors: number[][] = [];
  for (let i = 0; i < texts.length; i += BATCH_SIZE) {
    const batch = texts.slice(i, i + BATCH_SIZE);
    const output = await extractor(batch, { pooling: "mean", normalize: true });
    vectors.push(...(output.tolist() as number[][]));
  }
  return vectors;
}

/** Embed a single query string. */
export async function embedQuery(text: string): Promise<number[]> {
  const [vector] = await embed([text]);
  return vector;
}
