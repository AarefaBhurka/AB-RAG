/**
 * CSV loader — from scratch, rows as retrieval units.
 *
 * Two ideas here worth internalizing:
 *
 * 1. RFC 4180 parsing is a tiny state machine, not a split(","). Fields
 *    may be quoted, quoted fields may contain commas and newlines, and
 *    a doubled quote ("") inside quotes is a literal quote. The parser
 *    below walks the input once, character by character.
 *
 * 2. Structured data should NOT go through the prose chunker. A chunk
 *    like "…,42,red\nwidget,17,blu…" is meaningless to embed. Instead,
 *    every row becomes its own document, serialized as "header: value"
 *    lines — so each embedding captures one complete record WITH its
 *    column context:
 *
 *        name: widget
 *        price: 17
 *        color: blue
 *
 *    (For very wide/huge tables, hybrid SQL+vector retrieval beats
 *    embedding rows entirely — a later experiment.)
 */
import fs from "node:fs/promises";
import path from "node:path";
import type { RawDocument } from "../types.js";

/** Minimal RFC 4180 parser: handles quoted fields, escaped quotes, CRLF. */
export function parseCsv(input: string): string[][] {
  const rows: string[][] = [];
  let row: string[] = [];
  let field = "";
  let inQuotes = false;

  for (let i = 0; i < input.length; i++) {
    const char = input[i];
    if (inQuotes) {
      if (char === '"') {
        if (input[i + 1] === '"') {
          field += '"'; // escaped quote
          i++;
        } else {
          inQuotes = false;
        }
      } else {
        field += char;
      }
    } else if (char === '"') {
      inQuotes = true;
    } else if (char === ",") {
      row.push(field);
      field = "";
    } else if (char === "\n" || char === "\r") {
      if (char === "\r" && input[i + 1] === "\n") i++; // CRLF
      row.push(field);
      rows.push(row);
      row = [];
      field = "";
    } else {
      field += char;
    }
  }
  if (field.length > 0 || row.length > 0) {
    row.push(field);
    rows.push(row);
  }
  // Drop fully-empty rows (trailing newlines etc.)
  return rows.filter((r) => r.some((f) => f.trim().length > 0));
}

export async function loadCsv(filePath: string): Promise<RawDocument[]> {
  const source = path.relative(process.cwd(), filePath);
  const rows = parseCsv(await fs.readFile(filePath, "utf-8"));
  if (rows.length < 2) return []; // header only, or empty

  const [header, ...dataRows] = rows;
  return dataRows.map((row, i) => ({
    // :rowN in the source makes citations point at the exact record.
    source: `${source}:row${i + 2}`, // +2: 1-based, after the header line
    text: header.map((col, j) => `${col.trim()}: ${(row[j] ?? "").trim()}`).join("\n"),
  }));
}
