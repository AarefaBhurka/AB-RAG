/**
 * Conversational RAG — `rag chat`.
 *
 * The hard problem multi-turn adds is that FOLLOW-UPS ARE NOT QUERIES.
 * After "how does the chunker work?", a user asks "and how big are
 * they?" — embed that literally and you retrieve garbage: "they" and
 * "big" match nothing. The retriever is stateless; the conversation
 * isn't.
 *
 * The standard fix (the "condense question" step): before retrieving,
 * ask the LLM to rewrite the follow-up into a STANDALONE question
 * using the conversation history — "how large are the chunks produced
 * by the chunker?" — then retrieve with that. One extra LLM call per
 * turn buys retrieval that understands pronouns.
 *
 * The second problem is that history grows without bound while the
 * context window doesn't (Ollama silently drops overflow from the
 * front — including our system prompt!). Cheapest fix, used here: a
 * sliding window of recent turns under a token budget. The upgrade
 * path is summarization: compress evicted turns into a running summary
 * instead of dropping them (that's what production chat systems do).
 */
import readline from "node:readline/promises";
import { Retriever } from "./retriever.js";
import { answerWithOllama, type ChatMessage } from "./generate/ollama.js";
import { complete, ollamaAvailable } from "./generate/ollamaClient.js";
import { estimateTokens, selectWithinBudget } from "./context/budget.js";

// Budgets sized for Ollama's default ~4k window: leave room for the
// system prompt, the condensed question, and the answer itself.
const CHUNK_BUDGET_TOKENS = 1800;
const HISTORY_BUDGET_TOKENS = 800;
const CANDIDATES = 20;

/** Rewrite a follow-up into a standalone, retrieval-ready question. */
async function condenseQuestion(
  history: ChatMessage[],
  question: string,
): Promise<string> {
  if (history.length === 0) return question;
  const transcript = history
    .map((m) => `${m.role === "user" ? "User" : "Assistant"}: ${m.content}`)
    .join("\n");
  const standalone = await complete(
    "You rewrite follow-up questions as standalone search queries. Resolve pronouns and references using the conversation. Output ONLY the rewritten question.",
    `Conversation so far:\n${transcript}\n\nFollow-up: ${question}\n\nStandalone question:`,
  );
  return standalone.trim() || question;
}

/** Drop oldest turns until history fits its token budget. */
function trimHistory(history: ChatMessage[]): ChatMessage[] {
  const trimmed = [...history];
  while (
    trimmed.length > 0 &&
    trimmed.reduce((sum, m) => sum + estimateTokens(m.content), 0) >
      HISTORY_BUDGET_TOKENS
  ) {
    trimmed.shift();
  }
  return trimmed;
}

export async function runChat(indexPath: string): Promise<void> {
  if (!(await ollamaAvailable())) {
    console.error("Ollama is not reachable — start it with: ollama serve");
    process.exit(1);
  }
  const retriever = await Retriever.load(indexPath);
  // Async iteration (not rl.question) so lines typed or piped while the
  // LLM is busy are buffered instead of lost, and EOF ends the loop
  // cleanly — this is what makes `printf "q1\nq2\n" | rag chat` work.
  const rl = readline.createInterface({ input: process.stdin, output: process.stdout });
  let history: ChatMessage[] = [];

  console.log('Chat with your corpus. "exit" to quit.\n');
  process.stdout.write("you › ");

  for await (const line of rl) {
    const question = line.trim();
    if (question === "exit" || question === "quit") break;
    if (question) {
      const standalone = await condenseQuestion(history, question);
      if (standalone !== question) console.log(`    (retrieving for: "${standalone}")`);

      const candidates = await retriever.hybrid(standalone, CANDIDATES);
      const chunks = selectWithinBudget(candidates, CHUNK_BUDGET_TOKENS);

      process.stdout.write("rag › ");
      const answer = await answerWithOllama(standalone, chunks, history);
      console.log();

      // History carries the conversation, not the retrieved documents —
      // chunks are re-retrieved fresh every turn.
      history.push({ role: "user", content: question });
      history.push({ role: "assistant", content: answer });
      history = trimHistory(history);
    }
    process.stdout.write("you › ");
  }
  rl.close();
}
