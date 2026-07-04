/**
 * AB-RAG CLI
 *
 *   npm run rag ingest <path>       Index a file or directory
 *   npm run rag ask "<question>"    Retrieve + answer with citations
 *   npm run rag search "<query>"    Retrieval only (inspect what the LLM would see)
 */
import path from "node:path";
import { discoverFiles, loadDocuments } from "./ingest/loader.js";
import { chunkDocument } from "./chunking/chunker.js";
import { embed } from "./embeddings/embedder.js";
import { VectorIndex } from "./index/vector.js";
import { Retriever } from "./retriever.js";
import { answerWithOllama } from "./generate/ollama.js";
import { selectWithinBudget } from "./context/budget.js";
import { runEval } from "./eval/run.js";

const INDEX_PATH = path.join(process.cwd(), ".ragdata", "index.json");
const CHUNK_BUDGET_TOKENS = 1800; // context handed to the LLM, in tokens

async function ingest(target: string): Promise<void> {
  const files = await discoverFiles(target);
  if (files.length === 0) {
    console.error(`No loadable files found under ${target}`);
    process.exit(1);
  }
  console.log(`Found ${files.length} file(s). Loading & chunking...`);

  // Loaders may return several documents per file (CSV: one per row).
  const docs = (await Promise.all(files.map(loadDocuments))).flat();
  const allChunks = docs.flatMap((doc) => chunkDocument(doc));
  console.log(`Loaded ${docs.length} document(s) → ${allChunks.length} chunks. Embedding (local model)...`);

  const vectors = await embed(allChunks.map((c) => c.text));

  const index = new VectorIndex();
  allChunks.forEach((chunk, i) => index.add(chunk, vectors[i]));
  await index.save(INDEX_PATH);
  console.log(`Indexed ${index.size} chunks → ${INDEX_PATH}`);
}

async function loadRetriever(): Promise<Retriever> {
  return Retriever.load(INDEX_PATH).catch(() => {
    console.error(`No index found at ${INDEX_PATH}. Run: npm run rag ingest <path>`);
    process.exit(1);
  });
}

async function retrieve(query: string) {
  const retriever = await loadRetriever();
  // Pipeline follows the eval, not the folklore: on this corpus `rag
  // eval` scores plain hybrid ABOVE hybrid+CE (0.933 vs 0.875 MRR), so
  // hybrid is the default. Selection is token-budgeted, not a fixed
  // count (see context/budget.ts). Re-run the eval and revisit when
  // the corpus changes; hybridReranked() is one edit away.
  const candidates = await retriever.hybrid(query, 20);
  return selectWithinBudget(candidates, CHUNK_BUDGET_TOKENS);
}

async function main() {
  const [command, ...args] = process.argv.slice(2);

  switch (command) {
    case "ingest": {
      if (!args[0]) return usage();
      await ingest(args[0]);
      break;
    }
    case "search": {
      if (!args[0]) return usage();
      const results = await retrieve(args.join(" "));
      for (const r of results) {
        console.log(`\n[${r.score.toFixed(4)}] ${r.id}`);
        console.log(r.text.slice(0, 300).replace(/\s+/g, " ") + "…");
      }
      break;
    }
    case "ask": {
      if (!args[0]) return usage();
      const question = args.join(" ");
      const chunks = await retrieve(question);
      console.log(`Retrieved ${chunks.length} chunks from: ${[...new Set(chunks.map((c) => c.source))].join(", ")}\n`);
      await answerWithOllama(question, chunks);
      break;
    }
    case "chat": {
      const { runChat } = await import("./chat.js");
      await runChat(INDEX_PATH);
      break;
    }
    case "bench": {
      const { runBench } = await import("./eval/bench.js");
      await runBench(args[0] ? Number(args[0]) : undefined);
      break;
    }
    case "eval": {
      const datasetPath = args[0] ?? path.join(process.cwd(), "eval", "dataset.json");
      await runEval(datasetPath, INDEX_PATH);
      break;
    }
    default:
      usage();
  }
}

function usage() {
  console.log(`AB-RAG — a from-scratch RAG system

Usage:
  npm run rag ingest <path>        Index a file or directory
  npm run rag search "<query>"     Show retrieval results (no LLM call)
  npm run rag ask "<question>"     Answer with citations (local model via Ollama)
  npm run rag chat                 Multi-turn conversation over the corpus
  npm run rag eval [dataset.json]  Score vector vs BM25 vs hybrid retrieval
  npm run rag bench [n]            Brute force vs HNSW on n synthetic vectors

Setup for ask:
  brew install ollama && ollama serve && ollama pull llama3.2
  Override with OLLAMA_MODEL / OLLAMA_URL env vars.`);
}

main().catch((err) => {
  console.error(err);
  process.exit(1);
});
