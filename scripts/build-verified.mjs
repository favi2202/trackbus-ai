import { existsSync, mkdirSync } from "node:fs";
import { spawn } from "node:child_process";
import { fileURLToPath } from "node:url";
import path from "node:path";

const projectRoot = path.resolve(fileURLToPath(new URL("..", import.meta.url)));
const runtimeRoot = process.env.SITES_RUNTIME_ROOT
  ? path.resolve(process.env.SITES_RUNTIME_ROOT)
  : path.join(projectRoot, ".sites-runtime");

const runtimePaths = {
  home: path.join(runtimeRoot, "home"),
  npmCache: path.join(runtimeRoot, "npm-cache"),
  xdgConfig: path.join(runtimeRoot, "xdg-config"),
  temporary: path.join(runtimeRoot, "tmp"),
  wranglerLogs: path.join(runtimeRoot, "wrangler", "logs"),
};

for (const directory of Object.values(runtimePaths)) {
  mkdirSync(directory, { recursive: true });
}

const environment = {
  ...process.env,
  SITES_ENV_READY: "1",
  SITES_PROJECT_ROOT: projectRoot,
  HOME: runtimePaths.home,
  XDG_CONFIG_HOME: runtimePaths.xdgConfig,
  TMPDIR: runtimePaths.temporary,
  WRANGLER_WRITE_LOGS: "false",
  WRANGLER_LOG_PATH: runtimePaths.wranglerLogs,
  MINIFLARE_REGISTRY_PATH: path.join(runtimeRoot, "wrangler", "registry"),
  npm_config_cache: runtimePaths.npmCache,
  npm_config_audit: "false",
  npm_config_fund: "false",
  npm_config_update_notifier: "false",
};

for (const name of [
  "NPM_CONFIG_CACHE",
  "npm_config_proxy",
  "npm_config_http_proxy",
  "npm_config_https_proxy",
  "NPM_CONFIG_PROXY",
  "NPM_CONFIG_HTTP_PROXY",
  "NPM_CONFIG_HTTPS_PROXY",
]) {
  delete environment[name];
}

function duration(value, fallback) {
  if (!value) return fallback;
  const match = /^(\d+)(ms|s|m)?$/.exec(value.trim());
  if (!match) throw new Error(`Invalid duration: ${value}`);
  const scale = { ms: 1, s: 1_000, m: 60_000 }[match[2] ?? "ms"];
  return Number(match[1]) * scale;
}

function run(command, args, timeoutMs) {
  return new Promise((resolve, reject) => {
    const child = spawn(command, args, {
      cwd: projectRoot,
      env: environment,
      stdio: "inherit",
      shell: false,
    });
    let timedOut = false;
    const timer = setTimeout(() => {
      timedOut = true;
      child.kill("SIGTERM");
    }, timeoutMs);

    child.once("error", (error) => {
      clearTimeout(timer);
      reject(error);
    });
    child.once("exit", (code, signal) => {
      clearTimeout(timer);
      if (timedOut) {
        reject(new Error(`Command exceeded ${timeoutMs} ms: ${command}`));
      } else if (code !== 0) {
        reject(new Error(`${command} exited with code ${code ?? "unknown"}${signal ? ` (${signal})` : ""}`));
      } else {
        resolve();
      }
    });
  });
}

const vinext = path.join(projectRoot, "node_modules", "vinext", "dist", "cli.js");
if (!existsSync(vinext)) {
  throw new Error("vinext is unavailable. Run npm ci and wait for it to finish before building.");
}

console.log("Running bounded vinext build...");
await run(process.execPath, [vinext, "build"], duration(process.env.SITES_BUILD_TIMEOUT, 180_000));

await run(
  process.execPath,
  [
    "--experimental-loader",
    path.join(projectRoot, "scripts", "cloudflare-workers-loader.mjs"),
    path.join(projectRoot, "scripts", "validate-artifact.mjs"),
  ],
  duration(process.env.SITES_VALIDATE_TIMEOUT, 30_000),
);
