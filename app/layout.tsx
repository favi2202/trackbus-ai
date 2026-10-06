import type { Metadata } from "next";
import "./globals.css";
import "./pitch.css";
import { pitchConfig } from "@/lib/pitch-config";

export const metadata: Metadata = {
  metadataBase: new URL(pitchConfig.siteUrl),
  title: "TrackBus AI — Transport Intelligence",
  description:
    "AI-powered passenger-flow intelligence for smarter public transport operations. Explore the working MVP and planned pilot.",
  openGraph: {
    title: "TrackBus AI — Transport Intelligence",
    description: "Turn passenger flow into better transport decisions. Working MVP / research prototype.",
    type: "website",
    images: [{ url: "/pitch-og.png", width: 1200, height: 630, alt: "TrackBus AI: passenger flow, better transport decisions. MVP / Research Prototype." }],
  },
  twitter: {
    card: "summary_large_image",
    title: "TrackBus AI — Transport Intelligence",
    description: "AI-powered passenger-flow intelligence for smarter public transport operations.",
    images: ["/pitch-og.png"],
  },
  icons: {
    icon: "/favicon.svg",
    shortcut: "/favicon.svg",
    apple: "/favicon.svg",
  },
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en">
      <body className="antialiased">
        {children}
      </body>
    </html>
  );
}
