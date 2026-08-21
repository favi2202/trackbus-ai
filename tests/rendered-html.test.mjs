import assert from "node:assert/strict";
import test from "node:test";

async function renderedHome() {
  const workerUrl = new URL("../dist/server/index.js", import.meta.url);
  workerUrl.searchParams.set("test", `${process.pid}-${Date.now()}`);
  const { default: worker } = await import(workerUrl.href);
  return worker.fetch(
    new Request("http://localhost/", { headers: { accept: "text/html" } }),
    { ASSETS: { fetch: async () => new Response("Not found", { status: 404 }) } },
    { waitUntil() {}, passThroughOnException() {} },
  );
}

test("renders finished TrackBus metadata and Live Pilot navigation", async () => {
  const response = await renderedHome();
  assert.equal(response.status, 200);
  assert.match(response.headers.get("content-type") ?? "", /^text\/html\b/i);
  const html = await response.text();
  assert.match(html, /<title>TrackBus Showcase \| Transport intelligence<\/title>/i);
  assert.match(html, /property="og:image" content="https:\/\/trackbus-showcase\.favi\.workers\.dev\/og\.png"/i);
  assert.match(html, /rel="(?:shortcut )?icon" href="https:\/\/trackbus-showcase\.favi\.workers\.dev\/favicon\.svg"/i);
  assert.match(html, />Live pilot</i);
  assert.doesNotMatch(html, /codex-preview/i);
});
