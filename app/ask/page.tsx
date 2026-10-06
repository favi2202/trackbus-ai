import type { Metadata } from "next";
import { PitchShell } from "@/components/pitch-shell";
import { ProjectQuestions } from "@/components/pitch-interactions";

export const metadata: Metadata = { title: "Ask TrackBus | TrackBus AI", description: "Verified project answers about TrackBus privacy, hardware, forecasting, pilot plans and offline event delivery." };

export default function Ask() {
  return <PitchShell active="ask"><div className="pitch-container"><section className="pitch-page-hero"><div className="pitch-eyebrow">ASK TRACKBUS / PROJECT Q&A</div><h1>Good questions.<br /><em>Clear answers.</em></h1><p>Explore verified project information about the prototype, privacy, forecasting and the planned pilot.</p></section><section className="pitch-ask-section" aria-label="TrackBus project assistant"><ProjectQuestions /></section></div></PitchShell>;
}
