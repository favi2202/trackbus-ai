import { readFile } from "node:fs/promises";
import { fileURLToPath, pathToFileURL } from "node:url";
import path from "node:path";

const projectRoot = process.env.SITES_PROJECT_ROOT
  ? path.resolve(process.env.SITES_PROJECT_ROOT)
  : path.resolve(fileURLToPath(new URL("..", import.meta.url)));
const workerPath = path.join(projectRoot, "dist", "server", "index.js");
const hostingPath = path.join(projectRoot, "dist", ".openai", "hosting.json");

try {
  JSON.parse(await readFile(hostingPath, "utf8"));
} catch (error) {
  throw new Error(`Missing or invalid packaged Sites manifest: ${hostingPath}`, { cause: error });
}

try {
  await readFile(workerPath);
} catch (error) {
  throw new Error(`Missing Sites Worker entry: ${workerPath}`, { cause: error });
}

const workerUrl = pathToFileURL(workerPath);
workerUrl.searchParams.set("sites-validation", `${process.pid}-${Date.now()}`);
const worker = await import(workerUrl.href);
if (!worker.default || typeof worker.default.fetch !== "function") {
  throw new Error("dist/server/index.js must have an ESM default export with fetch(request, env, ctx)");
}

console.log("Validated Sites artifact: ESM Worker default.fetch and hosting manifest are present.");
