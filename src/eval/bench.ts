/**
 * ANN benchmark — brute force vs HNSW.
 *
 * The eval harness (run.ts) scores retrieval *quality* on labeled
 * questions. This benchmark scores retrieval *mechanics* on synthetic
 * vectors, where exact ground truth is computable:
 *
 *   recall@k  = |HNSW top-k ∩ exact top-k| / k, averaged over queries.
 *   latency   = wall-clock per query.
 *
 * Synthetic vectors are the right tool here: what's being measured is
 * the index structure, not the embedding model, and synthetic data lets
 * us scale n arbitrarily. But HOW they're generated matters enormously:
 *
 *   UNIFORM random vectors in 384 dims are the worst case for graph
 *   search — by concentration of measure, all pairwise distances bunch
 *   together, so no neighbor hop is meaningfully "closer" and greedy
 *   navigation stalls. Measured on this implementation: recall@10 was
 *   only 0.52 at ef=50 (vs 1.00 at 16 dims — same code!).
 *
 *   REAL embeddings are nothing like uniform: text lives on a low-
 *   dimensional manifold, so vectors form clusters with genuine
 *   near/far structure. The generator below mimics that (points =
 *   cluster center + bounded noise), which is the honest scenario to
 *   benchmark.
 *
 * The interesting output is the efSearch sweep: watch recall climb
 * toward 1.0 as latency rises toward brute force. That curve IS the
 * approximate-search tradeoff.
 */
import { HnswIndex } from "../index/hnsw.js";

/** Random unit vector (normalized, so dot product = cosine). */
function randomUnitVector(dims: number): number[] {
  const v = Array.from({ length: dims }, () => Math.random() * 2 - 1);
  const norm = Math.sqrt(v.reduce((s, x) => s + x * x, 0));
  return v.map((x) => x / norm);
}

function bruteForceTopK(corpus: number[][], query: number[], k: number): number[] {
  const scored = corpus.map((vector, id) => {
    let dot = 0;
    for (let i = 0; i < vector.length; i++) dot += vector[i] * query[i];
    return { id, dot };
  });
  scored.sort((a, b) => b.dot - a.dot);
  return scored.slice(0, k).map((s) => s.id);
}

/**
 * Cluster-structured vectors, mimicking real embedding geometry:
 * a point is a shared cluster center plus bounded random noise.
 * noiseScale 0.7 on unit centers ≈ intra-cluster cosine of ~0.8.
 */
function makeClusteredGenerator(dims: number, numClusters = 64, noiseScale = 0.7) {
  const centers = Array.from({ length: numClusters }, () => randomUnitVector(dims));
  return (): number[] => {
    const center = centers[Math.floor(Math.random() * numClusters)];
    const noise = randomUnitVector(dims);
    const v = center.map((c, i) => c + noise[i] * noiseScale);
    const norm = Math.sqrt(v.reduce((s, x) => s + x * x, 0));
    return v.map((x) => x / norm);
  };
}

export async function runBench(
  n = 5000,
  dims = 384,
  numQueries = 50,
  k = 10,
): Promise<void> {
  console.log(
    `ANN benchmark: n=${n} clustered vectors, ${dims} dims, ${numQueries} queries, top-${k}\n`,
  );

  const generate = makeClusteredGenerator(dims);
  const corpus = Array.from({ length: n }, generate);
  const queries = Array.from({ length: numQueries }, generate);

  // Build HNSW (this is the cost brute force never pays)
  const hnsw = new HnswIndex({ M: 16, efConstruction: 100 });
  const buildStart = performance.now();
  for (const vector of corpus) hnsw.add(vector);
  const buildMs = performance.now() - buildStart;
  console.log(`HNSW build: ${(buildMs / 1000).toFixed(2)}s (${(buildMs / n).toFixed(2)}ms/vector)\n`);

  // Exact ground truth + brute-force latency
  const bruteStart = performance.now();
  const groundTruth = queries.map((q) => new Set(bruteForceTopK(corpus, q, k)));
  const bruteMsPerQuery = (performance.now() - bruteStart) / numQueries;

  // Sweep efSearch
  const header = ["method", "recall@10", "ms/query", "speedup"];
  console.log(header.map((h) => h.padEnd(14)).join(" "));
  console.log("─".repeat(56));
  console.log(
    ["brute (exact)", "1.00", bruteMsPerQuery.toFixed(2), "1.0x"]
      .map((c) => c.padEnd(14))
      .join(" "),
  );

  for (const efSearch of [10, 25, 50, 100, 200]) {
    const start = performance.now();
    let totalRecall = 0;
    queries.forEach((q, i) => {
      const approx = hnsw.search(q, k, efSearch);
      const hits = approx.filter((r) => groundTruth[i].has(r.id)).length;
      totalRecall += hits / k;
    });
    const msPerQuery = (performance.now() - start) / numQueries;
    console.log(
      [
        `hnsw ef=${efSearch}`,
        (totalRecall / numQueries).toFixed(2),
        msPerQuery.toFixed(2),
        `${(bruteMsPerQuery / msPerQuery).toFixed(1)}x`,
      ]
        .map((c) => c.padEnd(14))
        .join(" "),
    );
  }

  console.log(
    "\nReading the table: recall climbs toward 1.0 as efSearch grows,\n" +
      "while the speedup over brute force shrinks. Where you sit on that\n" +
      "curve is a product decision, not an algorithmic one.",
  );
}
