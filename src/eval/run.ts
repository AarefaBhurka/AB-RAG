/**
 * Evaluation runner — compares the three retrieval strategies
 * head-to-head on a labeled question set.
 *
 * The dataset is intentionally cheap to build: each entry is just a
 * question plus the source file(s) that should be retrieved. Labeling
 * at the *source* level (not chunk level) keeps labels stable when you
 * change chunking parameters — which is exactly the kind of experiment
 * this harness exists to score.
 *
 * Workflow it enables:
 *   1. Run `rag eval` → baseline numbers.
 *   2. Change ONE thing (chunk size, RRF K, embedding model, ...).
 *   3. Re-ingest, re-run eval. Did hit@k / MRR move?
 * Never tune retrieval on vibes.
 */
import fs from "node:fs/promises";
import { Retriever, CANDIDATES_PER_RETRIEVER } from "../retriever.js";
import { reciprocalRankFusion } from "../index/hybrid.js";
import { ollamaAvailable } from "../generate/ollamaClient.js";
import { hypotheticalDocument, queryVariants } from "../query/transform.js";
import type { ScoredChunk } from "../types.js";
import { firstRelevantRank, hitRateAtK, meanReciprocalRank } from "./metrics.js";

interface EvalCase {
  question: string;
  expectedSources: string[];
}

const TOP_K = 6; // matches the context budget handed to the LLM

export async function runEval(
  datasetPath: string,
  indexPath: string,
): Promise<void> {
  const dataset: EvalCase[] = JSON.parse(await fs.readFile(datasetPath, "utf-8"));
  const retriever = await Retriever.load(indexPath);

  const strategies: Record<string, (q: string) => Promise<ScoredChunk[]>> = {
    vector: (q) => retriever.vector(q, TOP_K),
    bm25: async (q) => retriever.bm25(q, TOP_K),
    hybrid: (q) => retriever.hybrid(q, TOP_K),
    "hybrid+CE": (q) => retriever.hybridReranked(q, TOP_K),
  };

  // Query-transform strategies need the local LLM. Note: LLM sampling
  // varies between runs, so these rows wobble — rerun before concluding.
  if (await ollamaAvailable()) {
    // HyDE vs plain `vector` isolates the transform: same retriever,
    // only the embedded text changes (fake answer instead of question).
    strategies["hyde"] = async (q) =>
      retriever.vector(await hypotheticalDocument(q), TOP_K);
    // Original + paraphrases, hybrid retrieval each, RRF over all lists.
    strategies["multiQ"] = async (q) => {
      const variants = [q, ...(await queryVariants(q, 3))];
      const lists = await Promise.all(
        variants.map((v) => retriever.hybrid(v, CANDIDATES_PER_RETRIEVER)),
      );
      return reciprocalRankFusion(lists, TOP_K);
    };
  } else {
    console.log("(Ollama not reachable — skipping hyde and multiQ strategies)\n");
  }

  console.log(`Evaluating ${dataset.length} questions, top-${TOP_K} retrieval\n`);

  const misses: string[] = [];
  const rows: Record<string, { ranks: (number | undefined)[] }> = {};

  for (const [name, search] of Object.entries(strategies)) {
    const ranks: (number | undefined)[] = [];
    for (const { question, expectedSources } of dataset) {
      const results = await search(question);
      const rank = firstRelevantRank(results.map((r) => r.source), expectedSources);
      ranks.push(rank);
      if (name === "hybrid" && rank === undefined) {
        misses.push(`  ✗ "${question}" — wanted ${expectedSources.join(", ")}`);
      }
    }
    rows[name] = { ranks };
  }

  // Report
  const header = ["strategy", "hit@1", "hit@3", `hit@${TOP_K}`, "MRR"];
  console.log(header.map((h) => h.padEnd(9)).join(" "));
  console.log("─".repeat(50));
  for (const [name, { ranks }] of Object.entries(rows)) {
    const cells = [
      name,
      hitRateAtK(ranks, 1).toFixed(2),
      hitRateAtK(ranks, 3).toFixed(2),
      hitRateAtK(ranks, TOP_K).toFixed(2),
      meanReciprocalRank(ranks).toFixed(3),
    ];
    console.log(cells.map((c) => c.padEnd(9)).join(" "));
  }

  if (misses.length > 0) {
    console.log(`\nHybrid misses (answer not in the LLM's context at all):`);
    for (const m of misses) console.log(m);
  }
}
