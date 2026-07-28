import { cpSync, existsSync, mkdirSync } from "node:fs";
import { dirname, resolve } from "node:path";
import { fileURLToPath } from "node:url";

const scriptDir = dirname(fileURLToPath(import.meta.url));
const projectRoot = resolve(scriptDir, "..");
const sourceDir = resolve(projectRoot, "node_modules", "@litertjs", "core", "wasm");
const targetDir = resolve(projectRoot, "public", "wasm");

if (!existsSync(sourceDir)) {
  console.warn("LiteRT.js wasm directory was not found. Run npm install first.");
  process.exit(0);
}

mkdirSync(targetDir, { recursive: true });
cpSync(sourceDir, targetDir, { recursive: true });
console.log(`Copied LiteRT.js wasm assets to ${targetDir}`);
