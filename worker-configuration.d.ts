declare namespace Cloudflare {
  interface Env {
    ASSETS: Fetcher;
    DB: D1Database;
    TRACKBUS_INGEST_KEY?: string;
  }
}
