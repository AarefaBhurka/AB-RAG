/**
 * Document loaders — turn files on disk into RawDocuments.
 *
 * Each format gets a loader behind the same signature:
 * `(filePath) => Promise<RawDocument[]>`. Most formats yield one
 * document per file; CSV yields one per ROW (see csv.ts for why).
 * The rest of the pipeline never knows where text came from.
 */
import fs from "node:fs/promises";
import path from "node:path";
import type { RawDocument } from "../types.js";
import { loadPdf } from "./pdf.js";
import { loadDocx } from "./docx.js";
import { loadHtml } from "./html.js";
import { loadCsv } from "./csv.js";

/** Extensions read verbatim as UTF-8 text. */
const TEXT_EXTENSIONS = new Set([
  ".md", ".mdx", ".txt", ".rst",
  ".ts", ".tsx", ".js", ".jsx", ".py", ".go", ".rb", ".java", ".cs", ".php",
  ".json", ".yaml", ".yml", ".toml", ".css", ".sql", ".sh",
]);

async function loadText(filePath: string): Promise<RawDocument[]> {
  const text = await fs.readFile(filePath, "utf-8");
  return [{ source: path.relative(process.cwd(), filePath), text }];
}

/** Format-specific loaders, keyed by extension. */
const LOADERS = new Map<string, (filePath: string) => Promise<RawDocument[]>>([
  [".pdf", loadPdf],
  [".docx", loadDocx],
  [".html", loadHtml],
  [".htm", loadHtml],
  [".csv", loadCsv],
]);

function loaderFor(filePath: string) {
  const ext = path.extname(filePath).toLowerCase();
  if (LOADERS.has(ext)) return LOADERS.get(ext)!;
  if (TEXT_EXTENSIONS.has(ext)) return loadText;
  return undefined;
}

const SKIP_DIRS = new Set(["node_modules", ".git", ".ragdata", "dist", "build"]);

/** Recursively collect loadable files under a path (file or directory). */
export async function discoverFiles(root: string): Promise<string[]> {
  const stat = await fs.stat(root);
  if (stat.isFile()) return [root];

  const files: string[] = [];
  const entries = await fs.readdir(root, { withFileTypes: true });
  for (const entry of entries) {
    if (entry.isDirectory()) {
      if (!SKIP_DIRS.has(entry.name) && !entry.name.startsWith(".")) {
        files.push(...(await discoverFiles(path.join(root, entry.name))));
      }
    } else if (loaderFor(entry.name) !== undefined) {
      files.push(path.join(root, entry.name));
    }
  }
  return files;
}

export async function loadDocuments(filePath: string): Promise<RawDocument[]> {
  const loader = loaderFor(filePath);
  if (!loader) return [];
  try {
    const docs = await loader(filePath);
    return docs.filter((d) => d.text.length > 0);
  } catch (err) {
    // One corrupt file shouldn't sink a whole ingest run.
    console.error(`  ⚠ Failed to load ${filePath}: ${(err as Error).message}`);
    return [];
  }
}
