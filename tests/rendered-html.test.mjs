import assert from "node:assert/strict";
import test from "node:test";

async function renderedPlatform() {
  const workerUrl = new URL("../dist/server/index.js", import.meta.url);
  workerUrl.searchParams.set("test", `${process.pid}-${Date.now()}`);
  const { default: worker } = await import(workerUrl.href);
  return worker.fetch(
    new Request("http://localhost/platform", { headers: { accept: "text/html" } }),
    { ASSETS: { fetch: async () => new Response("Not found", { status: 404 }) } },
    { waitUntil() {}, passThroughOnException() {} },
  );
}

test("preserves the working platform metadata and trilingual navigation", async () => {
  const response = await renderedPlatform();
  assert.equal(response.status, 200);
  assert.match(response.headers.get("content-type") ?? "", /^text\/html\b/i);
  const html = (await response.text()).replace(/<!--.*?-->/g, "");
  assert.match(html, /<title>Working Prototype \| TrackBus AI<\/title>/i);
  assert.match(html, /property="og:image" content="https:\/\/trackbus-showcase\.favi-2202\.chatgpt\.site\/pitch-og\.png"/i);
  assert.match(html, /rel="(?:shortcut )?icon" href="https:\/\/trackbus-showcase\.favi-2202\.chatgpt\.site\/favicon\.svg"/i);
  assert.match(html, /href="\/"[^>]*>← TrackBus pitch</i);
  assert.match(html, />Live pilot</i);
  assert.match(html, /role="group" aria-label="Language"/i);
  assert.match(html, /title="English"[^>]*>EN</i);
  assert.match(html, /title="O‘zbekcha"[^>]*>UZ</i);
  assert.match(html, /title="Русский"[^>]*>RU</i);
  assert.match(html, /aria-label="Primary navigation"/i);
  assert.doesNotMatch(html, />18:47</i);
  assert.doesNotMatch(html, /codex-preview/i);
  assert.doesNotMatch(html, /\/workspace\/sites\//i);
});
