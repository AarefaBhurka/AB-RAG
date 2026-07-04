/**
 * Open-source generation via Ollama (local, free, no API key).
 *
 * Ollama serves open-weight models (Llama, Qwen, Mistral, ...) over a
 * local HTTP API. Setup:
 *
 *   brew install ollama
 *   ollama serve            # or launch the app
 *   ollama pull llama3.2    # ~2GB, runs comfortably on a Mac
 *
 * Citations use the classic prompt-based technique:
 *
 *   1. Number each chunk [1]..[n] in the prompt.
 *   2. Instruct the model to mark claims with [n].
 *   3. Parse the markers out of the response and map them back to chunks.
 *
 * The tradeoff to internalize: prompt-based citations tell you which
 * chunk the model *says* it used — they are model-claimed, not
 * verified. Smaller models sometimes cite wrong or not at all;
 * grounding quality is a function of model capability. A future
 * milestone can verify citations by checking the answer's claims
 * against the cited chunk's text.
 */
import type { ScoredChunk } from "../types.js";
import { orderForContext } from "../context/budget.js";
import { OLLAMA_MODEL, OLLAMA_URL } from "./ollamaClient.js";

export interface ChatMessage {
  role: "user" | "assistant";
  content: string;
}

// Small open models follow examples better than abstract rules — an
// earlier version that said "mark every claim with [n]" produced
// answers consisting of ONLY the marker. Show, don't tell.
//
// Second lesson, also learned the hard way: the example must be from
// an OBVIOUSLY different domain than the corpus. An earlier example
// about vector indexes got parroted VERBATIM into answers about this
// codebase (complete with a bogus citation), because the model
// couldn't tell style-example from source material.
const SYSTEM_PROMPT = `You are a retrieval-augmented assistant. Answer the user's question in 2-6 full sentences of prose, using ONLY the numbered documents provided.

After each claim, append the number of the supporting document in brackets. Here is a style example about an unrelated topic (baking) — copy its FORMAT only, never its words:

  Croissant dough is laminated by folding butter into it repeatedly [2]. The resting time between folds keeps the butter from melting [1].

If the documents don't contain the answer, say so plainly — do not guess.`;

/**
 * Answer a question from retrieved chunks. `history` (optional) carries
 * prior conversation turns for multi-turn chat. Returns the answer text.
 */
export async function answerWithOllama(
  question: string,
  chunks: ScoredChunk[],
  history: ChatMessage[] = [],
): Promise<string> {
  // Best chunks at the edges of the context, weakest in the middle
  // (lost-in-the-middle mitigation — see context/budget.ts). Citation
  // numbers refer to this ordered list.
  const ordered = orderForContext(chunks);
  const context = ordered
    .map((chunk, i) => `[${i + 1}] (source: ${chunk.id})\n${chunk.text}`)
    .join("\n\n---\n\n");

  let response: Response;
  try {
    response = await fetch(`${OLLAMA_URL}/api/chat`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        model: OLLAMA_MODEL,
        stream: true,
        messages: [
          { role: "system", content: SYSTEM_PROMPT },
          ...history,
          { role: "user", content: `Documents:\n\n${context}\n\nQuestion: ${question}` },
        ],
      }),
    });
  } catch {
    console.error(
      `Could not reach Ollama at ${OLLAMA_URL}.\n` +
        `Install and start it:\n  brew install ollama\n  ollama serve\n  ollama pull ${OLLAMA_MODEL}`,
    );
    process.exit(1);
  }
  if (!response.ok || !response.body) {
    const body = await response.text().catch(() => "");
    console.error(`Ollama error ${response.status}: ${body.slice(0, 300)}`);
    if (response.status === 404) {
      console.error(`Model may not be pulled yet. Run: ollama pull ${OLLAMA_MODEL}`);
    }
    process.exit(1);
  }

  // Ollama streams newline-delimited JSON objects.
  const decoder = new TextDecoder();
  let buffer = "";
  let fullText = "";
  const reader = response.body.getReader();
  while (true) {
    const { done, value } = await reader.read();
    if (done) break;
    buffer += decoder.decode(value, { stream: true });
    const lines = buffer.split("\n");
    buffer = lines.pop() ?? ""; // keep any partial trailing line
    for (const line of lines) {
      if (!line.trim()) continue;
      const event = JSON.parse(line);
      const delta: string = event.message?.content ?? "";
      if (delta) {
        fullText += delta;
        process.stdout.write(delta);
      }
    }
  }
  process.stdout.write("\n");

  // Map [n] markers back to the (ordered) chunks they reference.
  const citedNumbers = new Set(
    [...fullText.matchAll(/\[(\d+)\]/g)].map((m) => Number(m[1])),
  );
  const cited = [...citedNumbers]
    .filter((n) => n >= 1 && n <= ordered.length)
    .sort((a, b) => a - b);

  if (cited.length > 0) {
    console.log("\n─── Sources (model-claimed) ───");
    for (const n of cited) console.log(`[${n}] ${ordered[n - 1].id}`);
  }

  return fullText;
}
