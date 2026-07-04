/**
 * Context management — deciding what actually enters the LLM's window.
 *
 * Retrieval ranks chunks; this module decides how many make the cut
 * and in what order. Two facts drive it:
 *
 * 1. THE WINDOW IS A BUDGET, NOT A SUGGESTION. Ollama's default
 *    context (num_ctx) is ~4096 tokens; overflow is SILENTLY truncated
 *    from the front — the model just never sees your oldest content,
 *    with no error. "Retrieve top-6" is the wrong unit: six tiny CSV
 *    rows fit trivially, six max-size prose chunks may not. Budget in
 *    tokens.
 *
 * 2. MORE CONTEXT IS NOT MONOTONICALLY BETTER. Every marginally
 *    relevant chunk is a distractor the model might cite instead of
 *    the right one — recall@6 in our eval is a ceiling, but stuffing
 *    to the ceiling isn't free. Position matters too: models attend
 *    best to the START and END of the window and worst to the middle
 *    ("Lost in the Middle", Liu et al. 2023) — and small local models
 *    suffer this more than frontier ones.
 *
 * Token counting caveat: the honest count requires the generation
 * model's own tokenizer. The chars/4 heuristic below is within ~20%
 * for English prose — good enough for budgeting with headroom, useless
 * for billing. (CJK text and dense code skew lower than 4 chars/token.)
 */
import type { ScoredChunk } from "../types.js";

/** ~4 chars per token for English prose. Estimate, not truth. */
export function estimateTokens(text: string): number {
  return Math.ceil(text.length / 4);
}

/**
 * Walk chunks best-first, keep each one that still fits the budget.
 * (Skipping a big chunk but keeping a smaller, lower-ranked one is a
 * deliberate greedy choice — a relevant-but-huge chunk shouldn't
 * blank out the rest of the context.)
 */
export function selectWithinBudget(
  ranked: ScoredChunk[],
  budgetTokens: number,
): ScoredChunk[] {
  const selected: ScoredChunk[] = [];
  let used = 0;
  for (const chunk of ranked) {
    const cost = estimateTokens(chunk.text);
    if (used + cost > budgetTokens) continue;
    selected.push(chunk);
    used += cost;
  }
  return selected;
}

/**
 * Mitigate lost-in-the-middle: interleave so the best chunks sit at
 * the EDGES of the context and the weakest land in the middle.
 * Ranked [1,2,3,4,5,6] → ordered [1,3,5,6,4,2]: odd ranks walk in
 * from the front, even ranks walk in from the back.
 */
export function orderForContext(ranked: ScoredChunk[]): ScoredChunk[] {
  const front: ScoredChunk[] = [];
  const back: ScoredChunk[] = [];
  ranked.forEach((chunk, i) => {
    (i % 2 === 0 ? front : back).push(chunk);
  });
  return [...front, ...back.reverse()];
}
