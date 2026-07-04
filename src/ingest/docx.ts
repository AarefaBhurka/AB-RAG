/**
 * DOCX loader.
 *
 * A .docx file is a ZIP archive of XML parts — the body text lives in
 * word/document.xml as <w:t> runs inside <w:p> paragraphs. mammoth
 * handles the unzipping and XML traversal; extractRawText() joins
 * paragraphs with double newlines, which is exactly the boundary our
 * chunker splits on.
 *
 * (Legacy .doc is a proprietary binary format — not supported; convert
 * to .docx first.)
 */
import path from "node:path";
import mammoth from "mammoth";
import type { RawDocument } from "../types.js";

export async function loadDocx(filePath: string): Promise<RawDocument[]> {
  const { value } = await mammoth.extractRawText({ path: filePath });
  return [{ source: path.relative(process.cwd(), filePath), text: value.trim() }];
}
