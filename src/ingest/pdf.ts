/**
 * PDF loader.
 *
 * PDF is a page-description format, not a text format — "text" is
 * positioned glyph runs with no inherent reading order or paragraph
 * structure. Extraction is genuinely hard (that's why this one isn't
 * hand-rolled): unpdf wraps Mozilla's pdf.js, which reconstructs
 * reading order from glyph positions.
 *
 * Limitations to know: scanned PDFs yield nothing (they're images —
 * OCR is a different milestone), multi-column layouts can interleave,
 * and tables usually come out scrambled. Inspect extracted text with
 * `rag search` before trusting retrieval over a new PDF corpus.
 */
import fs from "node:fs/promises";
import path from "node:path";
import { extractText, getDocumentProxy } from "unpdf";
import type { RawDocument } from "../types.js";

export async function loadPdf(filePath: string): Promise<RawDocument[]> {
  const buffer = await fs.readFile(filePath);
  const pdf = await getDocumentProxy(new Uint8Array(buffer));
  // mergePages: false → one string per page; join with paragraph breaks
  // so the chunker treats page boundaries as split points.
  const { text: pages } = await extractText(pdf, { mergePages: false });
  const text = pages.join("\n\n").trim();
  return [{ source: path.relative(process.cwd(), filePath), text }];
}
