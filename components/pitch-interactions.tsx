"use client";

import { useState } from "react";
import Link from "next/link";
import { FleetMap } from "@/components/fleet-map";
import { occupancyPercent } from "@/lib/contracts";
import { forecastOccupancy } from "@/lib/forecast";
import { translate, type TranslationKey } from "@/lib/i18n";
import { route22Buses } from "@/lib/sample-data";
import { pipeline, projectQuestions } from "@/lib/pitch-config";

const t = (key: TranslationKey, values?: Record<string, string | number>) => translate("en", key, values);

export function FleetPreview() {
  const [selected, setSelected] = useState(3);
  const [mode, setMode] = useState<"occupancy" | "forecast">("occupancy");
  const values = route22Buses.map(bus => mode === "occupancy" ? bus.occupancy : forecastOccupancy({ recentOccupancy: [bus.occupancy - 8, bus.occupancy - 5, bus.occupancy - 4, bus.occupancy - 2, bus.occupancy - 1, bus.occupancy], capacity: bus.capacity, hour: 19, dayType: "weekday" }).expectedOccupancy);
  const bus = route22Buses[selected];
  const percent = occupancyPercent(values[selected], bus.capacity);
  return <div className="pitch-fleet-preview">
    <div className="pitch-preview-header"><div><span className="pitch-eyebrow">OPERATOR VIEW</span><strong>A clearer picture of the fleet.</strong></div><span className="pitch-sample-tag">Synthetic demo</span></div>
    <div className="pitch-preview-toolbar"><span>Route <b>22</b> / Tashkent scenario</span><div role="group" aria-label="Preview mode"><button aria-pressed={mode === "occupancy"} onClick={() => setMode("occupancy")}>Occupancy</button><button aria-pressed={mode === "forecast"} onClick={() => setMode("forecast")}>Forecast</button></div></div>
    <FleetMap values={values} selected={selected} onSelect={setSelected} live={false} t={t} />
    <div className="pitch-preview-detail" aria-live="polite"><div><small>SELECTED BUS</small><strong>{bus.label}</strong></div><div><small>{mode === "occupancy" ? "CURRENT ESTIMATE" : "BASELINE FORECAST"}</small><strong className={percent >= 72 ? "pitch-coral" : "pitch-mint"}>{percent}% <span>full</span></strong></div><Link href={`/platform?view=${mode === "forecast" ? "forecast" : "command"}`}>Explore dashboard <span aria-hidden="true">↗</span></Link></div>
    <p className="pitch-preview-note">Select a bus to inspect it. Illustrative values from the working prototype; forecast confidence is heuristic.</p>
  </div>;
}

export function SignalPipeline() {
  const [selected, setSelected] = useState(0);
  return <div className="pitch-pipeline"><div className="pitch-pipeline-steps" role="group" aria-label="Explore the TrackBus signal pipeline">{pipeline.map((step, index) => <button key={step.title} onClick={() => setSelected(index)} aria-pressed={index === selected}><span className="pitch-pipeline-number">{String(index + 1).padStart(2, "0")}</span><span><strong>{step.title}</strong><small>{step.label}</small></span><span aria-hidden="true">↗</span></button>)}</div><div className="pitch-pipeline-detail" aria-live="polite"><span className="pitch-eyebrow">SIGNAL {String(selected + 1).padStart(2, "0")} / 06</span><div className="pitch-signal-graphic" aria-hidden="true"><i /><i /><i /><span>{String(selected + 1).padStart(2, "0")}</span></div><h3>{pipeline[selected].title}</h3><p>{pipeline[selected].detail}</p><div className="pitch-pipeline-progress" aria-hidden="true">{pipeline.map((step, i) => <i key={step.title} className={i <= selected ? "active" : ""} />)}</div><span className="pitch-muted">Explore each step in the data journey.</span></div></div>;
}

export function CopyCode({ code, label }: { code: string; label: string }) {
  const [status, setStatus] = useState<"idle" | "copied" | "failed">("idle");
  async function copy() {
    try { await navigator.clipboard.writeText(code); setStatus("copied"); }
    catch { setStatus("failed"); }
  }
  return <div className="pitch-code"><div className="pitch-code-header"><span>{label}</span><button onClick={copy} aria-label={`Copy ${label}`}>{status === "copied" ? "Copied ✓" : "Copy code"}</button></div><pre tabIndex={0}><code>{code}</code></pre><span className="pitch-copy-status" role="status">{status === "failed" ? "Copy unavailable. Select the code to copy it manually." : status === "copied" ? "Code copied to clipboard." : ""}</span></div>;
}

export function ProjectQuestions({ compact = false }: { compact?: boolean }) {
  const [selected, setSelected] = useState(0);
  const questions = compact ? projectQuestions.slice(0, 4) : projectQuestions;
  return <div className="pitch-questions"><div className="pitch-question-list" role="group" aria-label="Verified project questions">{questions.map((item, index) => <button key={item.question} onClick={() => setSelected(index)} aria-pressed={index === selected}>{item.question}<span aria-hidden="true">↗</span></button>)}</div><div className="pitch-answer" aria-live="polite"><span className="pitch-eyebrow">PROJECT KNOWLEDGE</span><h3>{questions[selected].question}</h3><p>{questions[selected].answer}</p><small>Predefined answers from project documentation. No external AI service.</small><Link href="/demo">Explore the demo →</Link></div></div>;
}
