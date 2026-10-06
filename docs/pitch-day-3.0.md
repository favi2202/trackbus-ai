# Pitch Day 3.0 submission

TrackBus remains an MVP / research prototype. The pitch site introduces the
transport intelligence product; the existing software stays available at
`/platform` (and `/dashboard`). English pitch copy is complete. The platform
retains English, Uzbek and Russian controls.

## Routes

| Path | Purpose |
| --- | --- |
| `/` | Problem → solution, product, team, why us, roadmap, implementation and AI |
| `/demo` | Judge walkthrough, video placeholder, eight chapters and resources |
| `/platform` | Existing interactive operator dashboard and passenger beta |
| `/dashboard` | Alias for the same dashboard |
| `/developers` | Real API routes, canonical JSON, copyable examples and forecast input |
| `/ask` | Predefined project Q&A from verified documentation; no paid AI API |
| `/TrackBus-Technical-Guide.pdf` | Unchanged technical guide, served as a static asset |

`/platform?view=command`, `pilot`, `forecast`, `passenger` and `system` open the
corresponding existing view after hydration. The pitch link in the dashboard
returns to `/`.

## Add the video

Record and upload a **1–5 minute** walkthrough. In `lib/pitch-config.ts`, set:

```ts
export const DEMO_VIDEO_URL = "https://www.youtube.com/watch?v=YOUR_VIDEO_ID";
```

Supported sources: HTTPS YouTube watch/share/embed/shorts links, public Vimeo
links, or direct HTTPS `.mp4` / `.webm` URLs. YouTube uses the privacy-enhanced
embed host. Invalid or empty configuration keeps the placeholder. No autoplay.
For direct files, include narration or captions in the recording and host it
with a compatible video content type. Rebuild and redeploy after editing.

`pitchConfig` in the same file contains team details, roles, skills, source and
documentation links, the prototype route and `siteUrl`. Set `siteUrl` to the
actual public origin if deploying to a different host; sharing metadata and
developer examples use it. Team members are never generated automatically.

## Data and claim boundaries

- Fleet preview reuses the real `FleetMap` component and existing sample data.
- Command center, scenario effects and passenger beta are synthetic demos.
- Live Pilot reads D1 events; an empty/unavailable source stays empty/unavailable.
- Forecasting is the existing transparent baseline. Confidence values and
  intervals are heuristic, not calibrated production accuracy.
- APC, CCTV, GPS and historical data are source pathways, not claimed official
  integrations. Payment/demand data is a future option subject to permission.
- No completed live pilot, paying customer, transport partnership, production
  accuracy or real fleet deployment is claimed.
- Vision uses temporary anonymous tracks. No facial recognition or persistent
  passenger identity tracking is introduced.
- Two buses is a planned pilot scope, not a completed result.

## Validation

```bash
npm ci
npm run lint
npx tsc --noEmit
npm test
npm run cloudflare:dry-run
```

`npm test` includes the production build and existing API / passenger tests,
plus rendered pitch pages, required sections, preserved dashboard routes,
video URL handling, example-event validation and static asset packaging.
`npm run build` validates the ESM Worker artifact and packaged Sites manifest.
CI runs on pushes to `pitch-day-3.0` as well as main and pull requests.

Responsive layouts cover phone widths, tablet and desktop. Before final
submission, inspect at **375, 390, 430, 768 and 1440 px**: no page overflow;
mobile navigation opens; fleet selections and preview-mode buttons work;
pipeline stages, question buttons, copy controls, dashboard language changes,
scenario controls, passenger view and video playback work. CSS respects
`prefers-reduced-motion`. No browser screenshot/viewport pass is claimed by the
rendered HTML tests.

The fonts are served from `public/fonts`, using existing bundled Geist assets;
builds do not depend on a Google Fonts request or checkout-specific font URLs.
The sharing image is `public/pitch-og.png` (1200 × 630).

## Deployment

The existing `.openai/hosting.json`, Worker entry, D1 migrations and Wrangler
bindings are retained. The existing public Sites origin is:

`https://trackbus-showcase.favi-2202.chatgpt.site`

Prepared URLs **after publishing this branch**:

- `https://trackbus-showcase.favi-2202.chatgpt.site/`
- `https://trackbus-showcase.favi-2202.chatgpt.site/demo`
- `https://trackbus-showcase.favi-2202.chatgpt.site/platform`
- `https://trackbus-showcase.favi-2202.chatgpt.site/developers`
- `https://trackbus-showcase.favi-2202.chatgpt.site/ask`

These paths do not attest that the new branch has already been published.
For a direct Cloudflare deployment, follow `docs/cloudflare-deployment.md`.
The existing deployment must retain `DB` and `TRACKBUS_INGEST_KEY` for camera
ingestion. Public pitch, synthetic dashboard and forecast do not require a
device key. Never place ingestion credentials in browser code or examples.

Production hosting is not switched automatically by a GitHub branch push.
Publish a saved, verified build to update the existing site.

## Final submission check

- Required sections and `/demo` are implemented.
- Working prototype and technical guide remain linked.
- API showcase uses actual methods (the forecast endpoint is **POST**).
- Prototype status and synthetic scenarios are visibly disclosed.
- Team details are editable in one configuration file.
- **Still required:** add the real demo video URL, then rebuild and republish.
- **Final manual QA:** mobile viewport and interaction checks on the public build.
