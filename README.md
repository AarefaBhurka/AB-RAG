# AB-RAG

**A Retrieval-Augmented Generation system built from scratch in TypeScript — no frameworks, no cloud APIs, every algorithm hand-implemented and every design decision backed by measurement.**

Most RAG projects glue together LangChain, a hosted vector database, and a paid LLM API. This project deliberately does the opposite: the retrieval algorithms (BM25, vector search, HNSW, rank fusion), the parsers, and the evaluation harness are all written by hand, and the entire pipeline — embeddings, retrieval, and generation — runs **100% locally with zero API keys**. The goal: demonstrate a working understanding of how retrieval systems actually work, down to the data structures.

---

## How it works

```
 INGESTION
 files ──▶ Loaders (txt/code · PDF · DOCX · HTML · CSV)
              └─▶ Chunker (paragraph packing + overlap)
                     └─▶ Embedder (all-MiniLM-L6-v2, runs locally via transformers.js)
                            ├─▶ VectorIndex  (exact cosine search)
                            └─▶ BM25Index   (lexical search)

 QUERY
 question ─▶ condense (multi-turn only: rewrite follow-ups standalone)
     ├─▶ vector top-20 ──┐
     └─▶ BM25 top-20 ────┴─▶ Reciprocal Rank Fusion
                                 └─▶ token budget ─▶ edge ordering ("lost in the middle")
                                        └─▶ local LLM (Ollama) ─▶ answer + [n] citations

  = implemented from scratch
```

**The pipeline, stage by stage:**

1. **Loading** — one interface, five formats. The CSV parser is a hand-written RFC 4180 state machine; rows become individual retrieval units serialized as `header: value` pairs so each embedding carries a complete record. The HTML-to-text converter is also from scratch (block-tag-aware, entity decoding).
2. **Chunking** — paragraph-boundary packing with configurable overlap, so facts straddling a boundary are retrievable from either side.
3. **Hybrid retrieval** — dense vectors catch *meaning* ("laptop won't turn on" ↔ "computer fails to boot"); BM25 catches *exact terms* (IDs, function names, error strings). They fail on different queries, so results are fused by **rank** (RRF), sidestepping their incompatible score scales.
4. **Context management** — chunks are selected by **token budget**, not a fixed count (local models silently truncate overflow), and ordered so the strongest evidence sits at the edges of the context window, where models attend best.
5. **Generation with citations** — a local open-weights model answers strictly from the retrieved chunks, marking claims with `[n]`; markers are parsed back to exact sources (down to the CSV row: `catalog.csv:row2`).
6. **Multi-turn chat** — follow-ups like *"and how big are they?"* are unretrievable as-is; a condense-question step rewrites them into standalone queries before retrieval, and history lives under its own token budget.

---

## Measured, not assumed

The project includes its own **evaluation harness** (hit-rate@k and MRR over a labeled question set) and an **ANN benchmark** (recall vs latency against exact ground truth). Every architectural choice was tested — including the popular techniques that *lost*.

### Retrieval strategy comparison (`npm run rag eval`)

| Strategy | hit@1 | MRR | Verdict |
|---|---|---|---|
| **Hybrid (vector + BM25 + RRF)** | **0.90** | **0.933** | ✅ shipped as the default |
| Cross-encoder reranking | 0.80 | 0.875 | ❌ hurt on this corpus |
| Vector only | 0.70 | 0.833 | |
| Multi-query expansion | 0.70 | 0.750 | ❌ diluted good rankings |
| BM25 only | 0.50 | 0.708 | |
| HyDE | 0.60 | 0.675 | ❌ lost to plain vector |

The negative results are kept and **diagnosed in the code**: the MS MARCO cross-encoder is out-of-distribution on code chunks; HyDE and multi-query attack vocabulary mismatch, and this eval set has none. The default pipeline follows the measurements, not the folklore — and the one-command harness makes the experiment repeatable on any new corpus.

### HNSW vs brute force (`npm run rag bench`)

Hand-implemented HNSW (probabilistic layer assignment, beam search, binary heaps — also from scratch), benchmarked on 20,000 clustered 384-dim vectors:

| Method | recall@10 | ms/query | Speedup |
|---|---|---|---|
| Brute force (exact) | 1.00 | 20.6 | 1× |
| HNSW, ef=25 | 0.98 | 0.53 | **39×** |
| HNSW, ef=50 | 1.00 | 0.67 | **31×** |

The first implementation plateaued at 0.94 recall on clustered data. Root cause: naive closest-M pruning severs the links that bridge clusters. Implementing the paper's heuristic neighbor selection (keep links for *diversity of direction*, not just proximity) lifted recall to 1.00 — the debugging arc is documented in [`src/index/hnsw.ts`](src/index/hnsw.ts).

---

## Quickstart

```sh
npm install

# Generation runs on a local open-weights model via Ollama:
brew install ollama
ollama serve
ollama pull llama3.2
```

Indexing and search need no setup at all — the embedding model (~25MB) downloads on first run and everything stays on-device.

```sh
npm run rag ingest ./docs               # index a folder (md, code, pdf, docx, html, csv)
npm run rag search "how does X work"    # inspect retrieval — no LLM involved
npm run rag ask "how does X work"       # answer with cited sources
npm run rag chat                        # multi-turn conversation over the corpus
npm run rag eval                        # score retrieval strategies on a labeled dataset
npm run rag bench 20000                 # HNSW vs brute force: recall + latency
```

`search` is the debugging window: if the right chunk isn't in the retrieved set, no prompt can save the answer. Retrieval quality is 90% of RAG quality.

---

## Project structure

```
src/
├── ingest/          loaders: text/code, PDF, DOCX, HTML (from scratch), CSV (from scratch)
├── chunking/        paragraph-packing chunker with overlap
├── embeddings/      local sentence-transformer wrapper
├── index/           vector index, BM25, RRF fusion, HNSW + binary heap — all from scratch
├── rerank/          cross-encoder second stage (kept as a measured experiment)
├── query/           HyDE + multi-query transforms (kept as measured experiments)
├── context/         token budgeting + lost-in-the-middle ordering
├── generate/        local LLM client, streaming answers, citation parsing
├── eval/            hit@k / MRR harness + ANN benchmark
├── retriever.ts     one loaded index, three search strategies
├── chat.ts          conversational RAG (condense-question + history window)
└── cli.ts           ingest / search / ask / chat / eval / bench
eval/dataset.json    labeled questions for the eval harness
testdata/            fixtures covering all supported formats
```

Every module's header comment explains the *why* — the tradeoff it navigates, the paper it implements, or the measured result that shaped it.

---

## Engineering lessons captured in this codebase

- **Hybrid beats clever.** Two simple retrievers fused by rank outperformed every single-technique upgrade tried against them — verified, not assumed.
- **Evaluation before optimization.** The harness (built in milestone 3 of 7) caught three would-be regressions that folklore says are improvements.
- **Know your model's training distribution.** A web-prose reranker misjudges code; a 3B model parrots in-domain style examples verbatim (the fix: examples from an unmistakably different domain).
- **The context window is a budget.** Selection by token count with headroom, strongest evidence at the edges, history under its own budget — because local runtimes truncate silently.
- **High-dimensional geometry is unintuitive.** Uniform random vectors made HNSW look broken (0.52 recall); the same code hit 1.00 on realistically clustered data. Benchmark data distribution is part of the benchmark.

## Stack

TypeScript · Node 22 · [transformers.js](https://github.com/huggingface/transformers.js) (local embeddings + cross-encoder) · [Ollama](https://ollama.com) (local generation) · `unpdf` + `mammoth` (PDF/DOCX parsing). Everything else — including both search indexes — is hand-written and dependency-free.
