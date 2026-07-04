/**
 * Cross-encoder reranking — the second stage of two-stage retrieval.
 *
 * Why a second stage at all? Our embedder is a BI-encoder: it turns the
 * query and each document into vectors *independently*, and relevance
 * is reduced to one dot product. That independence is what makes
 * indexing possible (documents are embedded once, ahead of time) — but
 * it means the model never sees the query and document together, so it
 * can't judge interactions ("does THIS sentence actually answer THAT
 * question?").
 *
 * A CROSS-encoder concatenates [query, document] and runs them through
 * the transformer jointly — every query token attends to every document
 * token. Much more accurate, but the score can't be precomputed: it's
 * one full model forward pass per (query, candidate) pair. Hence the
 * funnel architecture used by every serious search system:
 *
 *   millions of chunks ──(cheap: vector+BM25)──▶ top ~20 ──(expensive:
 *   cross-encoder)──▶ top ~6 for the LLM.
 *
 * Model: ms-marco-MiniLM-L-6-v2 — trained on MS MARCO web-search
 * relevance pairs; runs locally via transformers.js (~23MB, one-time
 * download), same as the embedder. Fully open source.
 *
 * ── Measured caveat (from `rag eval` on this repo's own source) ──────
 * Reranking DROPPED MRR from 1.000 to 0.950 here: for "which file
 * extensions are discovered...", the CE promoted a prose-y CLI usage
 * header (score 0.33) over the chunk containing the actual extension
 * list (0.02). Two lessons:
 *   1. Rerankers have a training distribution. MS MARCO is web prose;
 *      code chunks are out-of-distribution, and the model prefers text
 *      that *reads like* an answer over text that *is* one.
 *   2. Note how low ALL the scores were (<0.35) — absolute CE scores
 *      carry confidence information that rank order hides.
 * The architecture stays (on prose corpora rerankers reliably help),
 * but the decision is per-corpus: run `rag eval` on YOUR data before
 * trusting any stage of the funnel.
 */
import {
  AutoModelForSequenceClassification,
  AutoTokenizer,
} from "@huggingface/transformers";
import type { Chunk, ScoredChunk } from "../types.js";

const MODEL = "Xenova/ms-marco-MiniLM-L-6-v2";
const BATCH_SIZE = 8;

let loadedPromise:
  | Promise<{ tokenizer: any; model: any }>
  | undefined;

async function load() {
  loadedPromise ??= (async () => {
    const tokenizer = await AutoTokenizer.from_pretrained(MODEL);
    const model = await AutoModelForSequenceClassification.from_pretrained(MODEL);
    return { tokenizer, model };
  })();
  return loadedPromise;
}

const sigmoid = (x: number) => 1 / (1 + Math.exp(-x));

/** Score (query, chunk) pairs jointly; return the topK by relevance. */
export async function rerank(
  query: string,
  candidates: Chunk[],
  topK: number,
): Promise<ScoredChunk[]> {
  if (candidates.length === 0) return [];
  const { tokenizer, model } = await load();

  const scored: ScoredChunk[] = [];
  for (let i = 0; i < candidates.length; i += BATCH_SIZE) {
    const batch = candidates.slice(i, i + BATCH_SIZE);
    // The tokenizer packs each pair as [CLS] query [SEP] chunk [SEP] —
    // one joint sequence per candidate, which is the whole point.
    const features = tokenizer(
      batch.map(() => query),
      {
        text_pair: batch.map((c) => c.text),
        padding: true,
        truncation: true,
      },
    );
    const { logits } = await model(features);
    const raw: number[][] = logits.tolist();
    batch.forEach((chunk, j) => {
      scored.push({ ...chunk, score: sigmoid(raw[j][0]) });
    });
  }

  return scored.sort((a, b) => b.score - a.score).slice(0, topK);
}
