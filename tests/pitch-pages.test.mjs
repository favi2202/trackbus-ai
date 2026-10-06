import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import path from "node:path";
import test from "node:test";
import { resolveDemoVideo } from "../lib/demo-video.ts";
import { apiEndpoints, examplePassengerEvent } from "../lib/pitch-api.ts";

const { default: worker } = await import("../dist/server/index.js");
const env = { ASSETS: { fetch: async request => {
  const pathname = new URL(typeof request === "string" ? request : request.url).pathname;
  if (!["/TrackBus-Technical-Guide.pdf", "/pitch-og.png", "/favicon.svg"].includes(pathname)) return new Response("Not found", { status: 404 });
  const bytes = await readFile(new URL(`../dist/client${pathname}`, import.meta.url));
  return new Response(bytes, { headers: { "content-type": pathname.endsWith(".pdf") ? "application/pdf" : pathname.endsWith(".png") ? "image/png" : "image/svg+xml" } });
} } };
const ctx = { waitUntil() {}, passThroughOnException() {} };
const request = (route, init) => worker.fetch(new Request(`http://localhost${route}`, init), env, ctx);

test("landing renders every required submission section and a working product entry", async () => {
  const response = await request("/");
  assert.equal(response.status, 200);
  const html = await response.text();
  assert.match(html, /<title>TrackBus AI — Transport Intelligence<\/title>/);
  for (const id of ["problem", "how-it-works", "product", "team", "why-us", "roadmap", "implementation"]) assert.ok(html.includes(`id="${id}"`), `missing ${id}`);
  assert.match(html, /MVP \/ Research Prototype/);
  assert.match(html, /Synthetic demo/);
  assert.match(html, /No facial recognition/);
  assert.match(html, /href="\/platform"/);
  assert.match(html, /href="\/demo"/);
  assert.match(html, /href="\/TrackBus-Technical-Guide.pdf"/);
  for (const stage of ["Idea", "Prototype", "MVP", "Pilot", "Launched"]) assert.ok(html.includes(stage));
  assert.doesNotMatch(html, /codex-preview/);
});

test("judge demo page includes video placeholder, eight chapters, resources and validation disclosure", async () => {
  const response = await request("/demo");
  assert.equal(response.status, 200);
  const html = await response.text();
  assert.match(html, /Demo video coming soon/);
  assert.match(html, /What you will see/);
  assert.match(html, /Production counting accuracy has not yet been validated/);
  for (const route of ["/platform", "/developers", "/TrackBus-Technical-Guide.pdf"]) assert.ok(html.includes(`href="${route}"`));
  for (const chapter of ["Detect", "Track", "Count", "Update", "Understand", "Forecast", "Evaluate", "Inform"]) assert.match(html, new RegExp(`<h3>${chapter}</h3>`));
  assert.doesNotMatch(html, /<iframe|<video/);
});

test("developer showcase documents existing method/path pairs and protects device credentials", async () => {
  const response = await request("/developers");
  assert.equal(response.status, 200);
  const html = await response.text();
  for (const endpoint of apiEndpoints) {
    const route = path.resolve("app", endpoint.path.split("?")[0].slice(1), "route.ts");
    const source = await readFile(route, "utf8");
    assert.ok(source.includes(`export async function ${endpoint.method}(`), `${endpoint.method} ${endpoint.path} does not exist`);
    assert.ok(html.includes(endpoint.path), `endpoint missing from page: ${endpoint.path}`);
  }
  assert.match(html, /Copy code/);
  assert.match(html, /YOUR_DEVICE_KEY/);
  assert.match(html, /synthetic_demo/);
  assert.match(html, /POST/);
});

test("published event example conforms to required canonical fields and API validation", async () => {
  const schema = JSON.parse(await readFile(new URL("../contracts/passenger-count-event.schema.json", import.meta.url)));
  assert.deepEqual(Object.keys(examplePassengerEvent).sort(), schema.required.toSorted());
  assert.equal(examplePassengerEvent.schemaVersion, "1.0");
  assert.ok(examplePassengerEvent.occupancy <= examplePassengerEvent.capacity);
  const response = await request("/api/v1/events/passenger-counts", { method: "POST", headers: { "content-type": "application/json" }, body: JSON.stringify(examplePassengerEvent) });
  assert.equal(response.status, 503); // Valid event, but no configured device key in this test.
  assert.equal((await response.json()).error, "Camera ingestion is not configured for this deployment");
});

test("the existing platform and dashboard alias remain working routes", async () => {
  for (const route of ["/platform", "/dashboard", "/platform?view=passenger", "/platform?view=forecast"]) {
    const response = await request(route);
    assert.equal(response.status, 200);
    const html = await response.text();
    assert.match(html, /Synthetic Tashkent data/);
    assert.match(html, /Command center/);
    assert.match(html, /Forecast lab/);
    assert.match(html, /Passenger app/);
    assert.match(html, /TrackBus pitch/);
  }
});

test("project Q&A is available without an external AI service", async () => {
  const response = await request("/ask");
  assert.equal(response.status, 200);
  const html = await response.text();
  assert.match(html, /Predefined answers from project documentation/);
  assert.match(html, /What happens if the network disconnects/);
  assert.match(html, /Does TrackBus use facial recognition/);
});

test("the build ships the unchanged technical PDF, social preview and favicon", async () => {
  const original = await readFile(new URL("../output/pdf/TrackBus-Technical-Guide.pdf", import.meta.url));
  const publicCopy = await readFile(new URL("../dist/client/TrackBus-Technical-Guide.pdf", import.meta.url));
  assert.deepEqual(original, publicCopy);
  for (const route of ["/TrackBus-Technical-Guide.pdf", "/pitch-og.png", "/favicon.svg"]) {
    const response = await request(route);
    assert.equal(response.status, 200);
    assert.ok((await response.arrayBuffer()).byteLength > 0);
  }
  const png = await readFile(new URL("../dist/client/pitch-og.png", import.meta.url));
  assert.equal(png.readUInt32BE(16), 1200);
  assert.equal(png.readUInt32BE(20), 630);
});

test("video config accepts supported secure links and rejects unsafe or malformed sources", () => {
  assert.equal(resolveDemoVideo(""), null);
  assert.equal(resolveDemoVideo("javascript:alert(1)"), null);
  assert.equal(resolveDemoVideo("http://example.com/demo.mp4"), null);
  assert.equal(resolveDemoVideo("https://user:secret@example.com/demo.mp4"), null);
  assert.equal(resolveDemoVideo("https://youtube.com.evil.example/watch?v=dQw4w9WgXcQ"), null);
  assert.equal(resolveDemoVideo("https://youtube.com/watch?v=broken"), null);
  for (const source of ["https://youtu.be/dQw4w9WgXcQ", "https://www.youtube.com/watch?v=dQw4w9WgXcQ", "https://www.youtube.com/embed/dQw4w9WgXcQ", "https://www.youtube.com/shorts/dQw4w9WgXcQ"]) {
    assert.deepEqual(resolveDemoVideo(source), { kind: "embed", url: "https://www.youtube-nocookie.com/embed/dQw4w9WgXcQ" });
  }
  assert.deepEqual(resolveDemoVideo("https://vimeo.com/123456"), { kind: "embed", url: "https://player.vimeo.com/video/123456" });
  assert.deepEqual(resolveDemoVideo("https://example.com/demo.mp4?token=public"), { kind: "video", url: "https://example.com/demo.mp4?token=public" });
  assert.deepEqual(resolveDemoVideo("https://example.com/demo.webm"), { kind: "video", url: "https://example.com/demo.webm" });
});
