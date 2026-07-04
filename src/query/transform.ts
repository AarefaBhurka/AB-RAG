/**
 * Query transformation — fix retrieval by rewriting the QUERY, not the index.
 *
 * The observation both techniques exploit: embeddings of questions and
 * embeddings of answers are systematically different. A user asks
 * "why fuse by rank?" but the chunk that answers it says "RRF sidesteps
 * incompatible score scales..." — related, yet phrased from opposite
 * sides of the conversation. Retrieval quality is capped by how well
 * one phrasing of the question happens to land near the answer.
 *
 * HyDE (Hypothetical Document Embeddings, Gao et al. 2022):
 *   Ask an LLM to WRITE A FAKE ANSWER, then embed the fake answer and
 *   search with that. The fake answer is usually factually wrong in
 *   its details — that's fine. It's *shaped* like the real answer
 *   (same vocabulary, same register, answer-side phrasing), so it
 *   lands closer to the true chunk than the question does. The
 *   hallucination is the feature.
 *
 * Multi-query:
 *   One phrasing is one sample from the space of ways to ask. Generate
 *   N paraphrases, retrieve for each, fuse with RRF — the same "they
 *   fail differently" logic as hybrid search, applied to phrasings
 *   instead of retrievers.
 *
 * Both cost an LLM call per query (~1-2s locally), which is why they're
 * evaluated as experiments in `rag eval` rather than wired into the
 * default path. Promote a winner only after it beats the baseline on
 * YOUR corpus.
 *
 * ── Measured on this repo's own source (59 chunks, 10 questions) ─────
 *   vector 0.833 MRR | hyde 0.675  ← HyDE LOST to plain vector
 *   hybrid 0.933 MRR | multiQ 0.750 ← multi-query LOST to plain hybrid
 *
 * Why, and why that's expected here:
 *   1. Dataset bias: the eval questions were written while reading the
 *      code, so they already share vocabulary with their target chunks.
 *      Query transforms attack vocabulary mismatch — when there is no
 *      mismatch, they can only add noise.
 *   2. HyDE's fake answers (from a 3B model) drift toward GENERIC
 *      technical prose, away from this codebase's idiosyncratic
 *      phrasing. HyDE shines on broad prose corpora with terse or vague
 *      user queries — the opposite of this setup.
 *   3. multiQ dilution: RRF fuses paraphrase lists democratically; when
 *      the original phrasing was already optimal, bad paraphrase lists
 *      drag it down.
 * The experiment to run when you have a real corpus: eval with VAGUE
 * queries (how end users actually type) vs precise ones — transforms
 * should flip from liability to win as queries get vaguer.
 */
import { complete } from "../generate/ollamaClient.js";

/** HyDE: a short fake answer, to be embedded in place of the query. */
export async function hypotheticalDocument(query: string): Promise<string> {
  const passage = await complete(
    "You write short, plausible documentation passages. No preamble, no disclaimers.",
    `Write a 2-4 sentence passage that directly answers the following question, as if excerpted from technical documentation. It does not need to be factually accurate — match the style and vocabulary an answer would use.\n\nQuestion: ${query}`,
  );
  return passage.trim();
}

/** Multi-query: N alternative phrasings (original NOT included). */
export async function queryVariants(query: string, n = 3): Promise<string[]> {
  const raw = await complete(
    "You rephrase search queries. Output ONLY the rephrasings, one per line, no numbering, no commentary.",
    `Rephrase this search query ${n} different ways, varying vocabulary and angle:\n\n${query}`,
  );
  return raw
    .split("\n")
    .map((line) => line.replace(/^[\s\d.\-*)]+/, "").trim())
    .filter((line) => line.length > 0)
    .slice(0, n);
}
