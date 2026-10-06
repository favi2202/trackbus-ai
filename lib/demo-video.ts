export type DemoVideo = { kind: "embed" | "video"; url: string } | null;

/** Only supported HTTPS sources are embedded; invalid config keeps the placeholder. */
export function resolveDemoVideo(value: string): DemoVideo {
  try {
    const url = new URL(value.trim());
    if (url.protocol !== "https:" || url.username || url.password) return null;
    let youtubeId: string | null = null;
    if (url.hostname === "youtu.be") youtubeId = url.pathname.slice(1).split("/")[0];
    if (["youtube.com", "www.youtube.com", "m.youtube.com", "www.youtube-nocookie.com"].includes(url.hostname)) {
      youtubeId = url.searchParams.get("v") ?? (/^\/(?:embed|shorts)\/([^/]+)/.exec(url.pathname)?.[1] ?? null);
    }
    if (youtubeId && /^[a-zA-Z0-9_-]{11}$/.test(youtubeId)) {
      return { kind: "embed", url: `https://www.youtube-nocookie.com/embed/${youtubeId}` };
    }
    if (["vimeo.com", "www.vimeo.com", "player.vimeo.com"].includes(url.hostname)) {
      const id = /^\/(?:video\/)?(\d+)\/?$/.exec(url.pathname)?.[1];
      if (id) return { kind: "embed", url: `https://player.vimeo.com/video/${id}` };
    }
    if (/\.(mp4|webm)$/i.test(url.pathname)) return { kind: "video", url: url.href };
    return null;
  } catch {
    return null;
  }
}
