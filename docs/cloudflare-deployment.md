# Cloudflare Workers deployment

TrackBus can run on Cloudflare Workers with static assets and a D1 database on
the Workers Free plan. The `.chatgpt.site` deployment can remain online as a
separate demo while the `workers.dev` deployment is used for camera ingestion.

## First deployment

Run these commands from the repository root. PowerShell does not require Git
Bash; the build and artifact checks run through Node.js on every platform:

```powershell
npx.cmd wrangler login
npm.cmd run cloudflare:types
npm.cmd run cloudflare:dry-run
npm.cmd run cloudflare:deploy
npm.cmd run cloudflare:migrate
npx.cmd wrangler secret put TRACKBUS_INGEST_KEY
```

Wrangler prints the public `https://trackbus-showcase.<account>.workers.dev`
URL after deployment. Enter the camera ingest key only at Wrangler's secret
prompt. Never commit the key to Git, `wrangler.jsonc`, or an `.env` file.

The D1 binding is named `DB`. Wrangler provisions `trackbus-showcase-db` from
the draft binding on the first authenticated deployment. The migration command
applies the SQL files in `drizzle/` and records them in D1's migration table.

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
