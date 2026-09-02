# Cloudflare Workers deployment

TrackBus can run on Cloudflare Workers with static assets and a D1 database on
the Workers Free plan. The `.chatgpt.site` deployment can remain online as a
separate demo while the `workers.dev` deployment is used for camera ingestion.

## Product choice

Use this Cloudflare stack for the current MVP:

- **Workers** for the Vinext/Next application and API routes;
- **Workers Static Assets** for the compiled browser bundle;
- **D1** for passenger-count events and operational snapshots;
- **Workers Secrets** for `TRACKBUS_INGEST_KEY`;
- **Workers Logs and source maps** for production diagnosis.

Do not add Pages, KV, R2, Queues, Workers AI, or Cloudflare Images yet. The
current application is already a full-stack Worker, stores structured rows in
D1, does not upload files, and does not use Next's image component. The Python
YOLO/ByteTrack pipeline remains on the bus or another suitable compute host; a
Worker receives its small JSON events rather than running the vision model.

## First deployment

Run these commands from the repository root. PowerShell does not require Git
Bash; the build and artifact checks run through Node.js on every platform:

```powershell
npx.cmd wrangler login
npx.cmd wrangler whoami
npm.cmd run cloudflare:types
npm.cmd run cloudflare:dry-run
npx.cmd wrangler secret put TRACKBUS_INGEST_KEY
npm.cmd run cloudflare:migrations:list
npm.cmd run cloudflare:migrate
npm.cmd run cloudflare:deploy
```

The required secret must exist before the final deploy. On an existing Worker,
Wrangler preserves the secret across later code deployments. The migration
scripts intentionally use the stable D1 database name instead of the binding
name, which helps prevent applying a migration to the wrong environment if a
binding is renamed later.

Wrangler prints the public `https://trackbus-showcase.<account>.workers.dev`
URL after deployment. Enter the camera ingest key only at Wrangler's secret
prompt. Never commit the key to Git, `wrangler.jsonc`, or an `.env` file.

The D1 binding is named `DB` and points to `trackbus-showcase-db`. The migration
command applies the SQL files in `drizzle/` and records them in D1's migration
table. Local development uses a local D1 database unless you deliberately opt
into a remote binding.

## Inspect an existing deployment

```powershell
npx.cmd wrangler whoami
npx.cmd wrangler deployments list
npx.cmd wrangler d1 info trackbus-showcase-db
npm.cmd run cloudflare:migrations:list
npx.cmd wrangler secret list
npx.cmd wrangler tail trackbus-showcase
```

`wrangler tail` stays open and streams live requests and exceptions; stop it
with `Ctrl+C` after the inspection window.

## Verify production

Replace `<worker-url>` with the exact URL printed by Wrangler:

```powershell
curl.exe <worker-url>/api/v1/health
curl.exe "<worker-url>/api/v1/events/passenger-counts?limit=1"
curl.exe <worker-url>/api/v1/operations/pilot
```

Expected initial state:

- health reports `status: ok`, `dataMode: connected`, and
  `storageConfigured: true`;
- the event feed reports `dataMode: live-empty` until the camera sends an
  event;
- the pilot endpoint reports `status: waiting-for-camera`.

## Connect Vision

```powershell
$env:TRACKBUS_API_KEY="<same key entered in Wrangler>"
python showcase.py --camera 0 --api-url <worker-url>/api
```

The camera sends normalized crossing events, not video frames. The public event
feed and Live Pilot view then refresh from D1.
