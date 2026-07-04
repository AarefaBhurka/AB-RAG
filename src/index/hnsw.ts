/**
 * HNSW (Hierarchical Navigable Small World) — from scratch.
 * Malkov & Yashunin, 2016. The index behind most production vector DBs.
 *
 * ── The problem ──────────────────────────────────────────────────────
 * Brute force compares the query to every vector: O(n) per query.
 * HNSW answers in roughly O(log n) by giving up exactness: it may miss
 * some true neighbors. That miss rate is what the benchmark measures
 * as recall.
 *
 * ── The idea: a skip list generalized to graphs ──────────────────────
 * Every node lives in layer 0. Each node is *also* promoted to higher
 * layers with exponentially decaying probability (like a coin-flip
 * skip list). Higher layers are therefore sparse graphs of long-range
 * links; layer 0 is a dense graph of short-range links.
 *
 * A search starts at the top layer's entry point and greedily hops to
 * whichever neighbor is closest to the query — crossing the space in
 * a few big jumps. Each layer down, the jumps get shorter. At layer 0
 * it switches from greedy (ef=1) to a beam search of width `ef`,
 * exploring candidates until none can improve the current top-ef.
 *
 * ── The knobs (and what turning them teaches) ────────────────────────
 *   M               max links per node per layer (layer 0 gets 2M).
 *                    More links = better recall, more memory.
 *   efConstruction  beam width while BUILDING. Higher = better graph,
 *                    slower build.
 *   efSearch        beam width while QUERYING. The runtime dial between
 *                    recall and latency — the benchmark sweeps this.
 *
 * Vectors are normalized, so distance = 1 - dot(a, b).
 */
import { BinaryHeap } from "./heap.js";

interface Node {
  vector: number[];
  /** neighbors[layer] = ids of linked nodes at that layer. */
  neighbors: number[][];
}

interface Neighbor {
  id: number;
  dist: number;
}

export interface HnswParams {
  M?: number;
  efConstruction?: number;
}

export class HnswIndex {
  private nodes: Node[] = [];
  private entryPoint = -1;
  private topLayer = -1;

  private readonly M: number;
  private readonly maxDegree0: number; // layer 0 allows 2M links
  private readonly efConstruction: number;
  /** Level multiplier 1/ln(M) — keeps expected layer count logarithmic. */
  private readonly levelMult: number;

  constructor({ M = 16, efConstruction = 100 }: HnswParams = {}) {
    this.M = M;
    this.maxDegree0 = 2 * M;
    this.efConstruction = efConstruction;
    this.levelMult = 1 / Math.log(M);
  }

  get size(): number {
    return this.nodes.length;
  }

  private dist(a: number[], b: number[]): number {
    let dot = 0;
    for (let i = 0; i < a.length; i++) dot += a[i] * b[i];
    return 1 - dot;
  }

  /** Exponentially decaying layer assignment — the "coin flips". */
  private randomLayer(): number {
    return Math.floor(-Math.log(Math.random()) * this.levelMult);
  }

  add(vector: number[]): number {
    const id = this.nodes.length;
    const layer = this.randomLayer();
    this.nodes.push({
      vector,
      neighbors: Array.from({ length: layer + 1 }, () => []),
    });

    if (this.entryPoint === -1) {
      this.entryPoint = id;
      this.topLayer = layer;
      return id;
    }

    // Phase 1: greedy descent through layers above the new node's top.
    let ep = this.entryPoint;
    for (let lc = this.topLayer; lc > layer; lc--) {
      ep = this.greedyClosest(vector, ep, lc);
    }

    // Phase 2: at each layer the node belongs to, beam-search for
    // efConstruction candidates, link to the closest M, prune overflow.
    for (let lc = Math.min(layer, this.topLayer); lc >= 0; lc--) {
      const candidates = this.searchLayer(vector, ep, this.efConstruction, lc);
      const maxDegree = lc === 0 ? this.maxDegree0 : this.M;

      for (const neighbor of this.selectNeighbors(vector, candidates, this.M)) {
        this.nodes[id].neighbors[lc].push(neighbor.id);
        this.nodes[neighbor.id].neighbors[lc].push(id);
        this.pruneNeighbors(neighbor.id, lc, maxDegree);
      }
      ep = candidates[0].id; // best candidate seeds the next layer down
    }

    if (layer > this.topLayer) {
      this.topLayer = layer;
      this.entryPoint = id;
    }
    return id;
  }

  /**
   * Heuristic neighbor selection (paper's Algorithm 4).
   *
   * Naive "keep the M closest" fails on clustered data: all M links
   * point INTO the node's own cluster, the pruning severs the few
   * links that bridge clusters, and queries landing near the wrong
   * cluster can never cross over. Measured on this codebase: naive
   * pruning plateaued at ~0.94 recall on clustered 384-dim vectors.
   *
   * The fix: walk candidates closest-first and keep one only if it is
   * closer to the base than to every neighbor already kept. A candidate
   * that's closer to an existing neighbor is redundant — that neighbor
   * already routes toward it — so its slot goes to a link in a genuinely
   * different direction. Diversity of direction > raw proximity.
   */
  private selectNeighbors(
    base: number[],
    candidates: Neighbor[], // sorted ascending by dist to base
    maxCount: number,
  ): Neighbor[] {
    const selected: Neighbor[] = [];
    const skipped: Neighbor[] = [];
    for (const candidate of candidates) {
      if (selected.length >= maxCount) break;
      const vec = this.nodes[candidate.id].vector;
      const redundant = selected.some(
        (s) => this.dist(vec, this.nodes[s.id].vector) < candidate.dist,
      );
      (redundant ? skipped : selected).push(candidate);
    }
    // keepPrunedConnections: top up so nodes don't end up under-linked.
    while (selected.length < maxCount && skipped.length > 0) {
      selected.push(skipped.shift()!);
    }
    return selected;
  }

  /** Re-select links via the heuristic when a node's list overflows. */
  private pruneNeighbors(id: number, layer: number, maxDegree: number): void {
    const links = this.nodes[id].neighbors[layer];
    if (links.length <= maxDegree) return;
    const base = this.nodes[id].vector;
    const candidates: Neighbor[] = links
      .map((nb) => ({ id: nb, dist: this.dist(base, this.nodes[nb].vector) }))
      .sort((a, b) => a.dist - b.dist);
    this.nodes[id].neighbors[layer] = this.selectNeighbors(base, candidates, maxDegree).map(
      (n) => n.id,
    );
  }

  /** ef=1 search: hop to the closest neighbor until no hop improves. */
  private greedyClosest(query: number[], ep: number, layer: number): number {
    let current = ep;
    let currentDist = this.dist(query, this.nodes[current].vector);
    let improved = true;
    while (improved) {
      improved = false;
      for (const nb of this.nodes[current].neighbors[layer]) {
        const d = this.dist(query, this.nodes[nb].vector);
        if (d < currentDist) {
          current = nb;
          currentDist = d;
          improved = true;
        }
      }
    }
    return current;
  }

  /**
   * Beam search within one layer. Returns up to `ef` closest nodes,
   * sorted ascending by distance.
   *
   * Two heaps drive it:
   *   candidates (min-heap) — frontier, closest first
   *   results    (max-heap) — best ef so far, WORST on top so it's O(1)
   *                           to check "can this candidate still help?"
   * Termination: the closest unexplored candidate is farther than the
   * worst kept result → no path can improve the result set.
   */
  private searchLayer(
    query: number[],
    ep: number,
    ef: number,
    layer: number,
  ): Neighbor[] {
    const visited = new Set<number>([ep]);
    const epDist = this.dist(query, this.nodes[ep].vector);

    const candidates = new BinaryHeap<Neighbor>((a, b) => a.dist < b.dist);
    const results = new BinaryHeap<Neighbor>((a, b) => a.dist > b.dist);
    candidates.push({ id: ep, dist: epDist });
    results.push({ id: ep, dist: epDist });

    while (candidates.size > 0) {
      const current = candidates.pop()!;
      if (current.dist > results.peek()!.dist && results.size >= ef) break;

      for (const nb of this.nodes[current.id].neighbors[layer]) {
        if (visited.has(nb)) continue;
        visited.add(nb);
        const d = this.dist(query, this.nodes[nb].vector);
        if (results.size < ef || d < results.peek()!.dist) {
          candidates.push({ id: nb, dist: d });
          results.push({ id: nb, dist: d });
          if (results.size > ef) results.pop();
        }
      }
    }
    return results.drain().reverse(); // max-heap drains worst-first
  }

  /** Approximate top-k. `efSearch` is the runtime recall↔latency dial. */
  search(query: number[], k: number, efSearch = 50): Neighbor[] {
    if (this.entryPoint === -1) return [];
    let ep = this.entryPoint;
    for (let lc = this.topLayer; lc >= 1; lc--) {
      ep = this.greedyClosest(query, ep, lc);
    }
    const results = this.searchLayer(query, ep, Math.max(efSearch, k), 0);
    return results.slice(0, k);
  }
}
