"use client";

import { useEffect, useMemo, useState } from "react";
import { occupancyLevel, occupancyPercent } from "@/lib/contracts";
import { forecastOccupancy } from "@/lib/forecast";
import { PassengerBeta } from "@/components/passenger-beta";
import {
  languageLocales,
  languageOptions,
  translate,
  type Language,
  type TranslationKey,
} from "@/lib/i18n";
import { hourlyForecast, passengerEvents, route22Buses } from "@/lib/sample-data";

type View = "command" | "pilot" | "forecast" | "passenger" | "system";
type Translator = (key: TranslationKey, values?: Record<string, string | number>) => string;

const mapPositions = [[17, 19], [31, 37], [49, 53], [66, 69], [81, 82]];

function clamp(value: number, low: number, high: number) {
  return Math.min(high, Math.max(low, value));
}

function Icon({ children }: { children: React.ReactNode }) {
  return <span className="icon-box" aria-hidden="true">{children}</span>;
}

function LoadBadge({ value, capacity = 72, t }: { value: number; capacity?: number; t: Translator }) {
  const level = occupancyLevel(value, capacity);
  return <span className={`load-badge ${level}`}><i />{t("percentFull", { percent: occupancyPercent(value, capacity) })}</span>;
}

function Sparkline({ values, accent = false, t }: { values: number[]; accent?: boolean; t: Translator }) {
  const max = Math.max(...values);
  const min = Math.min(...values);
  const points = values.map((value, index) => {
    const x = 8 + index * (244 / Math.max(1, values.length - 1));
    const y = 65 - ((value - min) / Math.max(1, max - min)) * 52;
    return `${x},${y}`;
  }).join(" ");
  return <svg className={`sparkline ${accent ? "accent" : ""}`} viewBox="0 0 260 74" role="img" aria-label={t("demandTrend")}>
    <defs><linearGradient id={`area-${accent}`} x1="0" y1="0" x2="0" y2="1"><stop offset="0" stopColor={accent ? "#ffb04a" : "#4ce1b6"} stopOpacity=".32"/><stop offset="1" stopColor={accent ? "#ffb04a" : "#4ce1b6"} stopOpacity="0"/></linearGradient></defs>
    <polyline className="spark-area" points={`8,70 ${points} 252,70`} fill={`url(#area-${accent})`} />
    <polyline className="spark-line" points={points} />
  </svg>;
}

function StoryRail({ active, onSelect, t }: { active: number; onSelect: (step: number) => void; t: Translator }) {
  const storySteps = [
    { kicker: t("detect"), title: t("detectTitle"), detail: t("detectDetail") },
    { kicker: t("predict"), title: t("predictTitle"), detail: t("predictDetail") },
    { kicker: t("act"), title: t("actTitle"), detail: t("actDetail") },
    { kicker: t("inform"), title: t("informTitle"), detail: t("informDetail") },
  ];
  return <section className="story-rail" aria-label={t("liveScenario")}>
    <div className="story-heading"><span>{t("liveScenario")}</span><strong>{t("eveningPeak")}</strong></div>
    <div className="story-steps">
      {storySteps.map((step, index) => <button key={step.kicker} className={index === active ? "active" : index < active ? "done" : ""} onClick={() => onSelect(index)}>
        <span className="step-number">{index < active ? "✓" : index + 1}</span>
        <span><small>{step.kicker}</small><strong>{step.title}</strong><em>{step.detail}</em></span>
      </button>)}
    </div>
  </section>;
}

function FleetMap({ values, selected, onSelect, live, t }: { values: number[]; selected: number; onSelect: (index: number) => void; live: boolean; t: Translator }) {
  return <div className="fleet-map">
    <div className="map-noise" />
    <div className="district d1">SHAYXONTOHUR</div><div className="district d2">YAKKASAROY</div><div className="district d3">MIRZO ULUG&apos;BEK</div>
    <svg className="map-network" viewBox="0 0 900 560" aria-label={t("routeMap")}>
      <path className="map-road major" d="M-30 120 C 180 150, 260 50, 440 135 S 690 205, 960 110" />
      <path className="map-road" d="M60 470 C 180 350, 300 430, 420 300 S 680 160, 910 250" />
      <path className="map-road" d="M210 -30 C 250 150, 180 250, 320 590" />
      <path className="map-road" d="M680 -20 C 580 130, 730 300, 620 590" />
      <path className="route-glow" d="M125 102 C 215 148, 250 180, 310 245 S 430 355, 515 390 S 680 430, 760 490" />
      <path className="route-core" d="M125 102 C 215 148, 250 180, 310 245 S 430 355, 515 390 S 680 430, 760 490" />
      {[[125,102,"Chorsu"],[310,245,"Navoiy"],[515,390,"Mustaqillik"],[760,490,"Do'stlik"]].map(([x,y,label]) => <g key={String(label)}><circle className="map-stop-ring" cx={Number(x)} cy={Number(y)} r="9"/><circle className="map-stop" cx={Number(x)} cy={Number(y)} r="4"/><text x={Number(x)+14} y={Number(y)-12}>{label}</text></g>)}
    </svg>
    {route22Buses.map((bus, index) => {
      const level = occupancyLevel(values[index], bus.capacity);
      const [left, top] = mapPositions[index];
      return <button key={bus.id} className={`vehicle ${level} ${selected === index ? "selected" : ""} ${live ? "moving" : ""}`} style={{ left: `${left}%`, top: `${top}%`, animationDelay: `${index * 180}ms` }} onClick={() => onSelect(index)} aria-label={`${bus.label}, ${t("percentFull", { percent: occupancyPercent(values[index], bus.capacity) })}`}>
        <span>22</span><strong>{occupancyPercent(values[index], bus.capacity)}%</strong>
      </button>;
    })}
    <div className="map-key"><span><i className="low"/>{t("space")}</span><span><i className="medium"/>{t("filling")}</span><span><i className="high"/>{t("crowded")}</span></div>
    <div className="map-status"><i/>{t("vehiclesReporting")}</div>
  </div>;
}

function CommandCenter({ tick, live, applied, setApplied, jump, t, displayTime }: { tick: number; live: boolean; applied: boolean; setApplied: (value: boolean) => void; jump: (step: number) => void; t: Translator; displayTime: string }) {
  const [selected, setSelected] = useState(3);
  const occupancies = route22Buses.map((bus, index) => {
    const pulse = live ? ((tick + index * 2) % 5) - 2 : 0;
    if (index === 3) return clamp(bus.occupancy + (applied ? -17 : Math.min(3, Math.floor(tick / 3))) + pulse, 0, bus.capacity);
    return clamp(bus.occupancy + pulse, 0, bus.capacity);
  });
  const selectedBus = route22Buses[selected];
  const selectedLoad = occupancies[selected];

  return <>
    <section className="metric-row">
      <article><Icon>▦</Icon><div><span>{t("vehiclesOnline")}</span><strong>128 <small>/ 132</small></strong><em className="positive">{t("reporting")}</em></div><Sparkline values={[112,118,121,119,124,126,128]} t={t}/></article>
      <article><Icon>◎</Icon><div><span>{t("passengersOnboard")}</span><strong>8,420</strong><em>{t("activeNetwork")}</em></div><Sparkline values={[48,52,55,61,65,72,78]} accent t={t}/></article>
      <article><Icon>↗</Icon><div><span>{t("peakDemand")}</span><strong>19:30</strong><em className="warning-text">{t("routes")}</em></div><div className="mini-bars"><i/><i/><i/><i/><i/><i/></div></article>
      <article className={applied ? "resolved" : "danger-card"}><Icon>{applied ? "✓" : "!"}</Icon><div><span>{applied ? t("interventionActive") : t("overloadRisk")}</span><strong>{applied ? "−23%" : t("overloadRoutes")}</strong><em>{applied ? t("crowdingReduction") : t("actionNeeded")}</em></div></article>
    </section>

    <section className="command-grid">
      <article className="surface map-surface">
        <header className="surface-head"><div><span className="section-kicker">{t("liveNetwork")}</span><h1>{t("fleetMovement")}</h1></div><div className="map-tabs"><button className="active">{t("vehicles")}</button><button>{t("heat")}</button><button>{t("stops")}</button></div></header>
        <FleetMap values={occupancies} selected={selected} onSelect={setSelected} live={live} t={t}/>
        <div className="vehicle-detail">
          <div className="route-chip">22</div><div><span>{t("selectedVehicle")}</span><strong>{selectedBus.label}</strong></div><LoadBadge value={selectedLoad} capacity={selectedBus.capacity} t={t}/><div><span>{t("nextStop")}</span><strong>{t("minutesShort", { minutes: selectedBus.etaMinutes })}</strong></div><div><span>{t("trend")}</span><strong>{applied && selected === 3 ? t("relieving") : selectedBus.trend === "up" ? t("rising") : selectedBus.trend === "down" ? t("falling") : t("stable")}</strong></div>
        </div>
      </article>

      <aside className="decision-stack">
        <article className={`surface incident-card ${applied ? "success" : ""}`}>
          <div className="incident-top"><span className="incident-code">{applied ? t("actionConfirmed") : t("priority")}</span><span className="incident-time">{displayTime}</span></div>
          <div className="incident-visual"><div className="capacity-ring" style={{ "--load": `${applied ? 68 : 91}%` } as React.CSSProperties}><strong>{applied ? 68 : 91}%</strong><span>{t("full")}</span></div><div><small>{t("bus", { bus: "22-04" })}</small><h2>{applied ? t("crowdingFalling") : t("capacityRisk")}</h2><p>{applied ? t("crowdingFallingDetail") : t("capacityRiskDetail")}</p></div></div>
          <div className="prediction"><span>{t("twelveMinutes")}</span><div><strong>{applied ? t("estimatedStabilization") : t("untilFullCapacity")}</strong><small>{t("forecastConfidence")}</small></div></div>
          {!applied ? <button className="approve-action" onClick={() => { setApplied(true); jump(3); }}><span>{t("dispatchReserve")}</span><b>{t("approve")}</b></button> : <button className="approve-action applied" onClick={() => setApplied(false)}><span>{t("reserveDispatched")}</span><b>{t("undoDemo")}</b></button>}
          <p className="decision-note">{t("operatorControl")}</p>
        </article>

        <article className="surface impact-card"><div className="surface-head compact"><div><span className="section-kicker">{t("predictedImpact")}</span><h2>{t("ifApproved")}</h2></div><span className="model-chip">AI · 86%</span></div><div className="impact-grid"><div><strong>{t("minutesShort", { minutes: "−14" })}</strong><span>{t("crowdedTravel")}</span></div><div><strong>+18%</strong><span>{t("routeCapacity")}</span></div><div><strong>620</strong><span>{t("ridersHelped")}</span></div></div><div className="impact-scale"><span>{t("current")}</span><i><b className={applied ? "after" : "before"}/></i><strong>{applied ? "68%" : "91%"}</strong></div></article>
      </aside>
    </section>

    <section className="surface event-ticker"><div><span className="live-dot"/>{t("liveEventStream")}</div>{passengerEvents.map((event) => <span key={event.eventId}><b>{event.observedAt.slice(11,19)}</b> {t("bus", { bus: event.busId.slice(-4) })} · {event.stopId} <em>+{event.boardings} / −{event.alightings}</em></span>)}<button onClick={() => jump(0)}>{t("inspectPipeline")}</button></section>
  </>;
}

function ForecastLab({ t }: { t: Translator }) {
  const [rain, setRain] = useState(false);
  const [eventNearby, setEventNearby] = useState(false);
  const forecast = useMemo(() => forecastOccupancy({ recentOccupancy: [42,47,51,56,61,66], capacity: 72, hour: 19, dayType: "weekday", weather: rain ? "rain" : "clear", eventNearby }), [rain, eventNearby]);
  const values = hourlyForecast.map(value => clamp(Math.round(value * (rain ? 1.07 : 1) * (eventNearby ? 1.1 : 1)), 0, 100));
  return <section className="feature-view">
    <header className="feature-hero"><div><span className="section-kicker">{t("scenarioEngine")}</span><h1>{t("forecastHero")}</h1><p>{t("forecastHeroDetail")}</p></div><div className="scenario-switches"><button className={!rain ? "active" : ""} onClick={() => setRain(false)}>☀ {t("clear")}</button><button className={rain ? "active" : ""} onClick={() => setRain(true)}>☂ {t("rain")}</button><button className={eventNearby ? "active" : ""} onClick={() => setEventNearby(!eventNearby)}>◉ {t("stadiumEvent")}</button></div></header>
    <div className="forecast-main">
      <article className="surface forecast-chart-card"><div className="surface-head"><div><span className="section-kicker">{t("routeFriday")}</span><h2>{t("expectedLoad")}</h2></div><span className="model-chip">{t("confidence86")}</span></div><svg className="forecast-chart" viewBox="0 0 900 360" role="img" aria-label={t("forecastChart")}><defs><linearGradient id="forecast-fill" x1="0" y1="0" x2="0" y2="1"><stop offset="0" stopColor="#4ce1b6" stopOpacity=".35"/><stop offset="1" stopColor="#4ce1b6" stopOpacity="0"/></linearGradient></defs>{[70,140,210,280].map(y => <line key={y} x1="45" y1={y} x2="860" y2={y}/>)}<path className="forecast-area" d={`M45 320 ${values.map((v,i)=>`L ${45+i*(815/(values.length-1))} ${320-v*2.7}`).join(" ")} L860 320 Z`}/><polyline points={values.map((v,i)=>`${45+i*(815/(values.length-1))},${320-v*2.7}`).join(" ")}/><line className="peak-line" x1="670" y1="50" x2="670" y2="320"/><circle cx="670" cy={320-values[13]*2.7} r="7"/><text x="630" y="38">{t("peak1930")}</text><text x="45" y="345">06:00</text><text x="420" y="345">14:00</text><text x="820" y="345">22:00</text></svg><div className="chart-caption"><span><i className="history"/>{t("historicalPattern")}</span><span><i className="prediction-line"/>{t("aiForecast")}</span><span><i className="confidence-band"/>{t("confidenceRange")}</span></div></article>
      <aside className="surface forecast-result"><span className="section-kicker">{t("forecast1900")}</span><strong className="forecast-big">{forecast.expectedPercent}<small>%</small></strong><LoadBadge value={forecast.expectedOccupancy} t={t}/><div className="forecast-facts"><div><span>{t("passengers")}</span><strong>{forecast.expectedOccupancy} / 72</strong></div><div><span>{t("likelyRange")}</span><strong>{forecast.lowerBound}–{forecast.upperBound}</strong></div><div><span>{t("drivers")}</span><strong>{rain ? t("rain") : t("normal")}{eventNearby ? ` + ${t("event")}` : ""}</strong></div></div><button className="approve-action"><span>{t("generateDispatch")}</span><b>→</b></button></aside>
    </div>
  </section>;
}

type PilotSource = {
  source: string;
  status: string;
  coverage?: string;
  latest?: string;
  eventCount?: number;
  busCount?: number;
  latestObservedAt?: string;
};

type PilotSnapshot = {
  dataMode: "live-empty" | "unavailable" | "live";
  summary: { status: string; eventCount: number; busCount: number; routeCount: number; flaggedEventCount: number; staleBusCount: number };
  sources: PilotSource[];
  reconciliation: { routeId: string; sensorBoardings: number; paymentBoardings: number | null; boardingGap: number | null; status: string } | null;
  buses: { busId: string; routeId: string; occupancy: number; capacity: number; source: string; observedAt: string }[];
  recentEvents: { eventId: string; observedAt: string; receivedAt: string; source: string; busId: string; routeId: string; stopId: string; doorId: string; boardings: number; alightings: number; occupancy: number; capacity: number; confidence: number; qualityFlags: string[] }[];
  refreshedAt: string;
};

const emptyPilot: PilotSnapshot = {
  dataMode: "live-empty",
  summary: { status: "waiting-for-camera", eventCount: 0, busCount: 0, routeCount: 0, flaggedEventCount: 0, staleBusCount: 0 },
  sources: [],
  reconciliation: null,
  buses: [],
  recentEvents: [],
  refreshedAt: new Date(0).toISOString(),
};

function PilotProof({ t, language }: { t: Translator; language: Language }) {
  const [snapshot, setSnapshot] = useState<PilotSnapshot>(emptyPilot);

  useEffect(() => {
    const controller = new AbortController();
    const refresh = () => {
      fetch("/api/v1/operations/pilot", { signal: controller.signal, cache: "no-store" })
        .then(async response => response.ok ? await response.json() as PilotSnapshot : Promise.reject(new Error("pilot endpoint offline")))
        .then(payload => setSnapshot(payload))
        .catch((error: unknown) => {
          if (error instanceof DOMException && error.name === "AbortError") return;
          setSnapshot(current => ({ ...current, dataMode: "unavailable" }));
        });
    };
    refresh();
    const interval = window.setInterval(refresh, 2000);
    return () => {
      controller.abort();
      window.clearInterval(interval);
    };
  }, []);

  const mode = snapshot.dataMode === "live" ? t("liveCameraData") : snapshot.dataMode === "live-empty" ? t("waitingCamera") : t("storageUnavailable");
  const proofTargets = [
    [t("accuracyTitle"), t("accuracyDetail"), t("accuracyMetric")],
    [t("telemetryTitle"), t("telemetryDetail"), t("telemetryMetric")],
    [t("historyTitle"), t("historyDetail"), t("historyMetric")],
    [t("failuresTitle"), t("failuresDetail"), t("failuresMetric")],
  ];
  const stages = [
    ["01", t("install"), t("installDetail")],
    ["02", t("collectLabel"), t("collectDetail")],
    ["03", t("tuneApi"), t("tuneDetail")],
    ["04", t("pilotReport"), t("reportDetail")],
  ];
  const statusKeys: Record<string, TranslationKey> = {
    online: "statusOnline",
    live: "statusLive",
    pilot: "statusPilot",
    degraded: "statusDegraded",
    stale: "statusStale",
    offline: "statusOffline",
  };

  return <section className="feature-view pilot-proof">
    <header className="feature-hero pilot-hero"><div><span className="section-kicker">{t("pilotProof")}</span><h1>{t("pilotHero")}</h1><p>{t("pilotHeroDetail")}</p></div><span className={`synthetic-label ${snapshot.dataMode === "live" ? "live-data" : ""}`}>{mode}</span></header>

    <div className="proof-targets">{proofTargets.map(([title, detail, metric]) => <article className="surface" key={title}><span>{t("proofTarget")}</span><h2>{title}</h2><p>{detail}</p><strong>{metric}</strong></article>)}</div>

    <div className="pilot-grid">
      <article className="surface source-health"><div className="surface-head"><div><span className="section-kicker">{t("sourceHealth")}</span><h2>{t("sourceContract")}</h2></div><span className="model-chip">{t("eventsCount", { count: snapshot.summary.eventCount.toLocaleString(languageLocales[language]) })}</span></div>
        <div className="source-list">{snapshot.sources.map(source => <div key={source.source}><span className={`source-dot ${source.status}`}/><strong>{source.source.toUpperCase()}</strong><em>{source.coverage ?? t("busesCount", { count: source.busCount ?? 0 })}</em><small>{source.latest ?? source.latestObservedAt ?? t("noEvent")}</small><b>{statusKeys[source.status] ? t(statusKeys[source.status]) : source.status}</b></div>)}</div>
        <div className="health-foot"><span>{t("validationEvents", { count: snapshot.summary.flaggedEventCount })}</span><span>{t("staleSources", { count: snapshot.summary.staleBusCount })}</span><span>{t("rawVideoLocal")}</span></div>
      </article>

      <aside className="surface reconcile-card"><span className="section-kicker">{t("reconciliation")}</span><h2>{t("route", { route: snapshot.reconciliation?.routeId ?? "—" })}</h2><div className="reconcile-values"><div><span>{t("sensorBoardings")}</span><strong>{snapshot.reconciliation?.sensorBoardings ?? "—"}</strong></div><div><span>{t("paymentTaps")}</span><strong>{snapshot.reconciliation?.paymentBoardings ?? "—"}</strong></div><div className="gap"><span>{t("gapInvestigate")}</span><strong>{snapshot.reconciliation?.boardingGap == null ? "—" : `${snapshot.reconciliation.boardingGap > 0 ? "+" : ""}${snapshot.reconciliation.boardingGap}`}</strong></div></div><p>{t("reconciliationDetail")}</p><button className="approve-action"><span>{t("openValidation")}</span><b>{t("operatorReview")}</b></button></aside>
    </div>

    <div className="live-data-grid">
      <article className="surface live-fleet"><div className="surface-head"><div><span className="section-kicker">{t("currentBuses")}</span><h2>{t("latestOccupancy")}</h2></div><span className="live-chip"><i/>{t("connectedCount", { count: snapshot.summary.busCount })}</span></div>
        {snapshot.buses.length ? <div className="live-bus-list">{snapshot.buses.map(bus => <div key={bus.busId}><strong>{bus.busId}</strong><span>{t("route", { route: bus.routeId })}</span><b>{bus.occupancy} / {bus.capacity}</b><em>{t("percentFull", { percent: Math.round((bus.occupancy / Math.max(1, bus.capacity)) * 100) })}</em><small>{new Date(bus.observedAt).toLocaleTimeString(languageLocales[language])}</small></div>)}</div> : <p className="empty-live">{t("startVision")}</p>}
      </article>
      <article className="surface live-events"><div className="surface-head"><div><span className="section-kicker">{t("liveJsonLog")}</span><h2>{t("latestEvents")}</h2></div><a href="/api/v1/events/passenger-counts?limit=50" target="_blank" rel="noreferrer">{t("openRawJson")}</a></div>
        {snapshot.recentEvents.length ? <div className="live-event-list">{snapshot.recentEvents.map(event => <div key={event.eventId}><time>{new Date(event.observedAt).toLocaleTimeString(languageLocales[language])}</time><strong>{event.busId}</strong><span>{event.stopId}</span><b>+{event.boardings} / −{event.alightings}</b><em>{t("onboard", { count: event.occupancy })}</em><small>{Math.round(event.confidence * 100)}% · {event.qualityFlags.length ? event.qualityFlags.join(", ") : "OK"}</small></div>)}</div> : <p className="empty-live">{t("noEvents")}</p>}
      </article>
    </div>

    <div className="validation-grid">
      <article className="surface"><span className="section-kicker">{t("validationGaps")}</span><h2>{t("evidenceRequired")}</h2><ul><li>{t("validationNight")}</li><li>{t("validationChildren")}</li><li>{t("validationOpposite")}</li><li>{t("validationDrift")}</li></ul></article>
      <article className="surface why-trackbus"><span className="section-kicker">{t("whyTrackbus")}</span><h2>{t("operationsHardware")}</h2><p>{t("operationsDetail")}</p><strong>{t("vendorNeutral")}</strong></article>
      <article className="surface privacy-boundary"><span className="section-kicker">{t("edgePrivacy")}</span><h2>{t("privacyHero")}</h2><ul><li>{t("temporaryIds")}</li><li>{t("noFaceRecognition")}</li><li>{t("localFrames")}</li><li>{t("noVehicleControl")}</li></ul></article>
    </div>

    <section className="pilot-stages"><header><span className="section-kicker">{t("pilotSequence")}</span><h2>{t("sequenceTitle")}</h2></header><div>{stages.map(([number, title, detail]) => <article key={number}><span>{number}</span><h3>{title}</h3><p>{detail}</p></article>)}</div></section>
  </section>;
}

function SystemView({ t }: { t: Translator }) {
  const stages = [
    { n: "01", icon: "◉", title: t("count"), text: t("countDetail"), meta: t("noFaceStorage") },
    { n: "02", icon: "⇄", title: t("validate"), text: t("validateDetail"), meta: t("dataQuality") },
    { n: "03", icon: "▦", title: t("forecast"), text: t("forecastDetail"), meta: t("confidence86") },
    { n: "04", icon: "↗", title: t("coordinate"), text: t("coordinateDetail"), meta: t("humanApproval") },
  ];
  return <section className="feature-view"><header className="feature-hero"><div><span className="section-kicker">{t("systemStory")}</span><h1>{t("systemHero")}</h1><p>{t("systemHeroDetail")}</p></div><span className="synthetic-label">{t("syntheticDemo")}</span></header><div className="system-flow">{stages.map((stage,index)=><article className="surface" key={stage.n}><span className="stage-number">{stage.n}</span><Icon>{stage.icon}</Icon><h2>{stage.title}</h2><p>{stage.text}</p><strong>{stage.meta}</strong>{index < stages.length-1 && <i>→</i>}</article>)}</div><div className="system-bottom"><article className="surface event-console"><div className="surface-head"><div><span className="section-kicker">{t("canonicalContract")}</span><h2>{t("passengerEvents")}</h2></div><span className="live-chip"><i/>{t("syntheticReplay")}</span></div>{passengerEvents.map(event=><div className="console-row" key={event.eventId}><span>{event.observedAt.slice(11,19)}</span><strong>{event.busId}</strong><span>{event.source}</span><b>+{event.boardings} / −{event.alightings}</b><LoadBadge value={event.occupancy} t={t}/><em>{t("confidence", { percent: Math.round(event.confidence*100) })}</em></div>)}</article><aside className="surface privacy-card"><Icon>◇</Icon><h2>{t("publicTrust")}</h2><p>{t("publicTrustDetail")}</p><ul><li>{t("noPassengerNames")}</li><li>{t("videoAtEdge")}</li><li>{t("noPersistentIdentity")}</li><li>{t("reviewableRecommendations")}</li></ul></aside></div></section>;
}

export function TrackBusApp() {
  const [view, setView] = useState<View>("command");
  const [live, setLive] = useState(false);
  const [tick, setTick] = useState(0);
  const [applied, setApplied] = useState(false);
  const [api, setApi] = useState<"checking" | "connected" | "offline">("checking");
  const [language, setLanguage] = useState<Language>("en");
  const [languageLoaded, setLanguageLoaded] = useState(false);
  const [now, setNow] = useState<Date | null>(null);
  const t = useMemo<Translator>(() => (key, values) => translate(language, key, values), [language]);
  const navigation: { id: View; label: string }[] = [
    { id: "command", label: t("commandCenter") },
    { id: "pilot", label: t("livePilot") },
    { id: "forecast", label: t("forecastLab") },
    { id: "passenger", label: t("passengerApp") },
    { id: "system", label: t("howItWorks") },
  ];
  const activeStory = applied ? 3 : Math.min(2, Math.floor((tick % 12) / 4));
  const tashkentTime = now
    ? new Intl.DateTimeFormat(languageLocales[language], {
      timeZone: "Asia/Tashkent",
      hour: "2-digit",
      minute: "2-digit",
      hour12: false,
    }).format(now)
    : "--:--";
  const tashkentDate = now
    ? new Intl.DateTimeFormat(languageLocales[language], {
      timeZone: "Asia/Tashkent",
      weekday: "short",
      day: "2-digit",
      month: "short",
    }).format(now)
    : t("tashkent");

  useEffect(() => {
    const timer = window.setTimeout(() => {
      const saved = window.localStorage.getItem("trackbus-language");
      if (saved === "en" || saved === "uz" || saved === "ru") setLanguage(saved);
      setLanguageLoaded(true);
    }, 0);
    return () => window.clearTimeout(timer);
  }, []);

  useEffect(() => {
    if (!languageLoaded) return;
    window.localStorage.setItem("trackbus-language", language);
    document.documentElement.lang = language;
  }, [language, languageLoaded]);

  useEffect(() => {
    const controller = new AbortController();
    fetch("/api/v1/health", { signal: controller.signal }).then(response => {
      if (!response.ok) throw new Error("offline");
      setApi("connected");
    }).catch((error: unknown) => {
      if (error instanceof DOMException && error.name === "AbortError") return;
      setApi("offline");
    });
    return () => controller.abort();
  }, []);

  useEffect(() => {
    if (!live) return;
    const timer = window.setInterval(() => setTick(value => value + 1), 1600);
    return () => window.clearInterval(timer);
  }, [live]);

  useEffect(() => {
    const firstUpdate = window.setTimeout(() => setNow(new Date()), 0);
    const timer = window.setInterval(() => setNow(new Date()), 30_000);
    return () => {
      window.clearTimeout(firstUpdate);
      window.clearInterval(timer);
    };
  }, []);

  function startDemo() {
    if (!live) { setTick(0); setApplied(false); setView("command"); }
    setLive(value => !value);
  }

  function jumpStory(step: number) {
    setTick(step * 4);
    setApplied(step === 3);
    setView("command");
  }

  return <main className="app-shell">
    <div className="demo-disclaimer"><span>{t("showcaseMode")}</span><p>{t("showcaseDisclaimer")}</p><button onClick={() => setView("system")}>{t("realVsSimulated")}</button></div>
    <div className="site-header">
      <header className="topbar">
        <button className="brand" onClick={() => setView("command")}><span>TB</span><div><strong>TrackBus</strong><small>{t("urbanIntelligence")}</small></div></button>
        <div className="header-status">
          <div className="language-switcher" role="group" aria-label={t("language")}>
            {languageOptions.map(option => <button key={option.code} type="button" className={language === option.code ? "active" : ""} aria-pressed={language === option.code} title={option.label} onClick={() => setLanguage(option.code)}>{option.short}</button>)}
          </div>
          <span className={`api-status ${api}`}><i/>{api === "connected" ? t("systemOnline") : api === "offline" ? t("offline") : t("connecting")}</span>
          <div className="city-time" aria-label={`${t("tashkent")} ${tashkentTime}`}><strong>{tashkentTime}</strong><small>{tashkentDate}</small></div>
          <button className={`demo-control ${live?"playing":""}`} onClick={startDemo}>{live ? `Ⅱ ${t("pauseScenario")}` : `▶ ${t("runScenario")}`}</button>
        </div>
      </header>
      <nav className="primary-nav" aria-label={t("primaryNavigation")}>{navigation.map(item=><button key={item.id} className={view===item.id?"active":""} onClick={()=>setView(item.id)}>{item.label}</button>)}</nav>
    </div>
    {view === "command" && <StoryRail active={activeStory} onSelect={jumpStory} t={t}/>}
    <div className="content-frame">{view === "command" && <CommandCenter tick={tick} live={live} applied={applied} setApplied={setApplied} jump={jumpStory} t={t} displayTime={tashkentTime}/>} {view === "pilot" && <PilotProof t={t} language={language}/>} {view === "forecast" && <ForecastLab t={t}/>} {view === "passenger" && <PassengerBeta t={t}/>} {view === "system" && <SystemView t={t}/>}</div>
    <footer><span>{t("footerBrand")}</span><span>{t("footerData")}</span></footer>
  </main>;
}
