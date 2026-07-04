/**
 * HTML → text — from scratch.
 *
 * A real browser builds a DOM; we don't need one to get readable text.
 * The pipeline:
 *   1. Drop content that is never prose: <script>, <style>, comments.
 *   2. Turn block-level closing tags into paragraph breaks so the
 *      chunker sees document structure (a <p> boundary in HTML should
 *      be a chunk boundary candidate, same as a blank line in text).
 *   3. Strip all remaining tags, decode common entities, tidy whitespace.
 *
 * Honest limitations of regex-over-HTML: CDATA, conditional comments,
 * and pathological markup can slip through, and there's no readability
 * heuristic (nav bars and footers are ingested too). For a crawled-web
 * corpus you'd graduate to a real parser + boilerplate removal — this
 * is the 80% that makes local HTML files retrievable.
 */
import fs from "node:fs/promises";
import path from "node:path";
import type { RawDocument } from "../types.js";

const BLOCK_TAGS =
  "p|div|h[1-6]|li|ul|ol|tr|table|section|article|header|footer|blockquote|pre|figure";

const ENTITIES: Record<string, string> = {
  "&nbsp;": " ",
  "&lt;": "<",
  "&gt;": ">",
  "&quot;": '"',
  "&#39;": "'",
  "&apos;": "'",
  "&amp;": "&", // decoded last so &amp;lt; doesn't become <
};

export function htmlToText(html: string): string {
  let text = html
    .replace(/<script[\s\S]*?<\/script>/gi, "")
    .replace(/<style[\s\S]*?<\/style>/gi, "")
    .replace(/<!--[\s\S]*?-->/g, "");

  text = text
    .replace(new RegExp(`</(${BLOCK_TAGS})>`, "gi"), "\n\n")
    .replace(/<(br|hr)\s*\/?>/gi, "\n")
    .replace(/<[^>]+>/g, " ");

  for (const [entity, char] of Object.entries(ENTITIES)) {
    text = text.replaceAll(entity, char);
  }
  // Numeric entities: &#8212; etc.
  text = text.replace(/&#(\d+);/g, (_, code) => String.fromCodePoint(Number(code)));

  return text
    .replace(/[ \t]+/g, " ")
    .replace(/ ?\n ?/g, "\n")
    .replace(/\n{3,}/g, "\n\n")
    .trim();
}

export async function loadHtml(filePath: string): Promise<RawDocument[]> {
  const html = await fs.readFile(filePath, "utf-8");
  return [
    { source: path.relative(process.cwd(), filePath), text: htmlToText(html) },
  ];
}
