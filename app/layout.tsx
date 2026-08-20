import type { Metadata } from "next";
import { Geist, Geist_Mono } from "next/font/google";
import "./globals.css";

const geistSans = Geist({
  variable: "--font-geist-sans",
  subsets: ["latin"],
});

const geistMono = Geist_Mono({
  variable: "--font-geist-mono",
  subsets: ["latin"],
});

export const metadata: Metadata = {
  metadataBase: new URL("https://trackbus-showcase.favi-2202.chatgpt.site"),
  title: "TrackBus Showcase | Transport intelligence",
  description:
    "Pilot-ready passenger counting, occupancy forecasting, and operator decision intelligence for Tashkent transport.",
  openGraph: {
    title: "TrackBus Showcase | Transport intelligence",
    description: "Anonymous counts become auditable transport decisions.",
    type: "website",
    images: [{ url: "/og.png", width: 1672, height: 941, alt: "TrackBus transport intelligence showcase" }],
  },
  twitter: {
    card: "summary_large_image",
    title: "TrackBus Showcase | Transport intelligence",
    description: "Anonymous counts become auditable transport decisions.",
    images: ["/og.png"],
  },
  icons: {
    icon: "/favicon.svg",
    shortcut: "/favicon.svg",
  },
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en">
      <body
        className={`${geistSans.variable} ${geistMono.variable} antialiased`}
      >
        {children}
      </body>
    </html>
  );
}
