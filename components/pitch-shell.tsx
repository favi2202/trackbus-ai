import Link from "next/link";
import { pitchConfig } from "@/lib/pitch-config";

export function Arrow({ diagonal = false }: { diagonal?: boolean }) {
  return <svg width="18" height="18" viewBox="0 0 24 24" fill="none" aria-hidden="true"><path d={diagonal ? "M6 18 18 6M6 6h12v12" : "M4 12h16m-6-6 6 6-6 6"} stroke="currentColor" strokeWidth="1.7" strokeLinecap="round" strokeLinejoin="round" /></svg>;
}

export function PitchButton({ href, children, secondary = false, external = false }: { href: string; children: React.ReactNode; secondary?: boolean; external?: boolean }) {
  const className = `pitch-button${secondary ? " pitch-button-secondary" : ""}`;
  return external
    ? <a className={className} href={href} target="_blank" rel="noopener noreferrer">{children}<Arrow diagonal /></a>
    : <Link className={className} href={href}>{children}<Arrow /></Link>;
}

export function SectionHeading({ number, eyebrow, title, children }: { number: string; eyebrow: string; title: string; children?: React.ReactNode }) {
  return <div className="pitch-section-heading"><div className="pitch-eyebrow"><span>{number}</span>{eyebrow}</div><h2>{title}</h2>{children && <p>{children}</p>}</div>;
}

export function PitchShell({ children, active }: { children: React.ReactNode; active?: "home" | "demo" | "developers" | "ask" }) {
  const links = [
    { href: "/#how-it-works", label: "How it works" },
    { href: "/#roadmap", label: "Roadmap" },
    { href: "/#team", label: "Team" },
    { href: "/demo", label: "Demo", id: "demo" },
    { href: "/developers", label: "Developers", id: "developers" },
  ];
  const navigation = links.map(link => <Link key={link.href} href={link.href} aria-current={active === link.id ? "page" : undefined}>{link.label}</Link>);
  return <div className="pitch-root">
    <a className="pitch-skip" href="#pitch-main">Skip to content</a>
    <header className="pitch-header"><div className="pitch-container pitch-header-inner">
      <Link href="/" className="pitch-brand" aria-label="TrackBus AI home"><span className="pitch-brand-mark" aria-hidden="true"><svg viewBox="0 0 32 32" fill="none"><path d="M8 7h16v12H8zM8 13h16M12 7v6M20 7v6M10 19v6m12-6v6M10 23h12" stroke="currentColor" strokeWidth="2" strokeLinecap="round"/><circle cx="12" cy="17" r="1" fill="currentColor"/><circle cx="20" cy="17" r="1" fill="currentColor"/></svg></span><strong>TrackBus<span> AI</span></strong></Link>
      <nav className="pitch-nav" aria-label="Pitch navigation">{navigation}</nav>
      <Link className="pitch-header-cta" href={pitchConfig.prototypeUrl}>Open Live Demo<Arrow /></Link>
      <details className="pitch-mobile-menu"><summary aria-label="Open navigation"><svg width="22" height="22" viewBox="0 0 24 24" fill="none" aria-hidden="true"><path d="M4 6h16M4 12h16M4 18h16" stroke="currentColor" strokeWidth="1.7" strokeLinecap="round"/></svg></summary><nav aria-label="Mobile pitch navigation">{navigation}<Link href="/ask">Ask TrackBus</Link></nav></details>
    </div></header>
    <main id="pitch-main">{children}</main>
    <footer className="pitch-footer"><div className="pitch-container"><div className="pitch-footer-top"><Link href="/" className="pitch-brand"><span className="pitch-footer-wordmark">TB</span><strong>TrackBus<span> AI</span></strong></Link><p>Passenger flow. Better decisions.</p><span className="pitch-status"><i />{pitchConfig.status}</span></div><div className="pitch-footer-bottom"><span>Built for transport intelligence · AIFU Pitch Day 3.0</span><nav aria-label="Footer navigation"><Link href="/demo">Demo</Link><Link href="/ask">Ask TrackBus</Link><Link href="/developers">API</Link><a href={pitchConfig.guideUrl}>Technical guide</a><a href={pitchConfig.repositoryUrl} target="_blank" rel="noopener noreferrer">GitHub ↗</a></nav></div></div></footer>
  </div>;
}
