"use client";

import { useEffect, useMemo, useState } from "react";
import { occupancyLevel, occupancyPercent } from "@/lib/contracts";
import { forecastOccupancy } from "@/lib/forecast";
import { hourlyForecast, passengerEvents, route22Buses } from "@/lib/sample-data";

type View = "command" | "forecast" | "passenger" | "system";

const navigation: { id: View; label: string }[] = [
  { id: "command", label: "Command center" },
  { id: "forecast", label: "Forecast lab" },
  { id: "passenger", label: "Passenger app" },
  { id: "system", label: "How it works" },
];

const storySteps = [
  { kicker: "Detect", title: "Door counter reports +12", detail: "Bus 22-04 reaches 91% capacity" },
  { kicker: "Predict", title: "Overload in 12 minutes", detail: "Model confidence: 86%" },
  { kicker: "Act", title: "Reserve bus recommended", detail: "Move one low-load vehicle to Route 22" },
  { kicker: "Inform", title: "Passenger ETA updated", detail: "Crowding falls from 91% to 68%" },
];

const mapPositions = [[17, 19], [31, 37], [49, 53], [66, 69], [81, 82]];

function clamp(value: number, low: number, high: number) {
  return Math.min(high, Math.max(low, value));
}

function Icon({ children }: { children: React.ReactNode }) {
  return <span className="icon-box" aria-hidden="true">{children}</span>;
}

function LoadBadge({ value, capacity = 72 }: { value: number; capacity?: number }) {
  const level = occupancyLevel(value, capacity);
  return <span className={`load-badge ${level}`}><i />{occupancyPercent(value, capacity)}% full</span>;
}

function Sparkline({ values, accent = false }: { values: number[]; accent?: boolean }) {
  const max = Math.max(...values);
  const min = Math.min(...values);
  const points = values.map((value, index) => {
    const x = 8 + index * (244 / Math.max(1, values.length - 1));
    const y = 65 - ((value - min) / Math.max(1, max - min)) * 52;
    return `${x},${y}`;
  }).join(" ");
  return <svg className={`sparkline ${accent ? "accent" : ""}`} viewBox="0 0 260 74" role="img" aria-label="Demand trend">
    <defs><linearGradient id={`area-${accent}`} x1="0" y1="0" x2="0" y2="1"><stop offset="0" stopColor={accent ? "#ffb04a" : "#4ce1b6"} stopOpacity=".32"/><stop offset="1" stopColor={accent ? "#ffb04a" : "#4ce1b6"} stopOpacity="0"/></linearGradient></defs>
    <polyline className="spark-area" points={`8,70 ${points} 252,70`} fill={`url(#area-${accent})`} />
    <polyline className="spark-line" points={points} />
  </svg>;
}

function StoryRail({ active, onSelect }: { active: number; onSelect: (step: number) => void }) {
  return <section className="story-rail" aria-label="Guided demo story">
    <div className="story-heading"><span>LIVE SCENARIO</span><strong>Route 22 evening peak</strong></div>
    <div className="story-steps">
      {storySteps.map((step, index) => <button key={step.kicker} className={index === active ? "active" : index < active ? "done" : ""} onClick={() => onSelect(index)}>
        <span className="step-number">{index < active ? "✓" : index + 1}</span>
        <span><small>{step.kicker}</small><strong>{step.title}</strong><em>{step.detail}</em></span>
      </button>)}
    </div>
  </section>;
}

function FleetMap({ values, selected, onSelect, live }: { values: number[]; selected: number; onSelect: (index: number) => void; live: boolean }) {
  return <div className="fleet-map">
    <div className="map-noise" />
    <div className="district d1">SHAYXONTOHUR</div><div className="district d2">YAKKASAROY</div><div className="district d3">MIRZO ULUG&apos;BEK</div>
    <svg className="map-network" viewBox="0 0 900 560" aria-label="Stylized map of Route 22 in Tashkent">
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
      return <button key={bus.id} className={`vehicle ${level} ${selected === index ? "selected" : ""} ${live ? "moving" : ""}`} style={{ left: `${left}%`, top: `${top}%`, animationDelay: `${index * 180}ms` }} onClick={() => onSelect(index)} aria-label={`${bus.label}, ${occupancyPercent(values[index], bus.capacity)} percent full`}>
        <span>22</span><strong>{occupancyPercent(values[index], bus.capacity)}%</strong>
      </button>;
    })}
    <div className="map-key"><span><i className="low"/>Space</span><span><i className="medium"/>Filling</span><span><i className="high"/>Crowded</span></div>
    <div className="map-status"><i/>5 vehicles reporting · 4 sec ago</div>
  </div>;
}

function CommandCenter({ tick, live, applied, setApplied, jump }: { tick: number; live: boolean; applied: boolean; setApplied: (value: boolean) => void; jump: (step: number) => void }) {
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
      <article><Icon>▦</Icon><div><span>Vehicles online</span><strong>128 <small>/ 132</small></strong><em className="positive">96.9% reporting</em></div><Sparkline values={[112,118,121,119,124,126,128]}/></article>
      <article><Icon>◎</Icon><div><span>Passengers onboard</span><strong>8,420</strong><em>Across the active network</em></div><Sparkline values={[48,52,55,61,65,72,78]} accent/></article>
      <article><Icon>↗</Icon><div><span>Peak demand</span><strong>19:30</strong><em className="warning-text">Routes 22 · 67 · 93</em></div><div className="mini-bars"><i/><i/><i/><i/><i/><i/></div></article>
      <article className={applied ? "resolved" : "danger-card"}><Icon>{applied ? "✓" : "!"}</Icon><div><span>{applied ? "Intervention active" : "Overload risk"}</span><strong>{applied ? "−23%" : "3 routes"}</strong><em>{applied ? "Expected crowding reduction" : "Action needed in 12 min"}</em></div></article>
    </section>

    <section className="command-grid">
      <article className="surface map-surface">
        <header className="surface-head"><div><span className="section-kicker">LIVE NETWORK</span><h1>Tashkent fleet movement</h1></div><div className="map-tabs"><button className="active">Vehicles</button><button>Heat</button><button>Stops</button></div></header>
        <FleetMap values={occupancies} selected={selected} onSelect={setSelected} live={live}/>
        <div className="vehicle-detail">
          <div className="route-chip">22</div><div><span>Selected vehicle</span><strong>{selectedBus.label}</strong></div><LoadBadge value={selectedLoad} capacity={selectedBus.capacity}/><div><span>Next stop</span><strong>{selectedBus.etaMinutes} min</strong></div><div><span>Trend</span><strong>{applied && selected === 3 ? "Relieving ↓" : selectedBus.trend === "up" ? "Rising ↑" : selectedBus.trend === "down" ? "Falling ↓" : "Stable →"}</strong></div>
        </div>
      </article>

      <aside className="decision-stack">
        <article className={`surface incident-card ${applied ? "success" : ""}`}>
          <div className="incident-top"><span className="incident-code">{applied ? "ACTION CONFIRMED" : "PRIORITY 01"}</span><span className="incident-time">18:47</span></div>
          <div className="incident-visual"><div className="capacity-ring" style={{ "--load": `${applied ? 68 : 91}%` } as React.CSSProperties}><strong>{applied ? 68 : 91}%</strong><span>full</span></div><div><small>BUS 22-04</small><h2>{applied ? "Crowding is falling" : "Capacity risk detected"}</h2><p>{applied ? "Reserve vehicle 17-08 is entering Route 22. Passenger arrival times have been updated." : "Demand near Chorsu is rising faster than the current timetable can absorb."}</p></div></div>
          <div className="prediction"><span>12 min</span><div><strong>{applied ? "Estimated stabilization" : "until full capacity"}</strong><small>Forecast confidence 86%</small></div></div>
          {!applied ? <button className="approve-action" onClick={() => { setApplied(true); jump(3); }}><span>Dispatch reserve bus</span><b>Approve →</b></button> : <button className="approve-action applied" onClick={() => setApplied(false)}><span>Reserve bus dispatched</span><b>Undo demo</b></button>}
          <p className="decision-note">Recommendation only · an operator remains in control</p>
        </article>

        <article className="surface impact-card"><div className="surface-head compact"><div><span className="section-kicker">PREDICTED IMPACT</span><h2>If action is approved</h2></div><span className="model-chip">AI · 86%</span></div><div className="impact-grid"><div><strong>−14 min</strong><span>crowded travel</span></div><div><strong>+18%</strong><span>route capacity</span></div><div><strong>620</strong><span>riders helped</span></div></div><div className="impact-scale"><span>Current</span><i><b className={applied ? "after" : "before"}/></i><strong>{applied ? "68%" : "91%"}</strong></div></article>
      </aside>
    </section>

    <section className="surface event-ticker"><div><span className="live-dot"/>LIVE EVENT STREAM</div>{passengerEvents.map((event) => <span key={event.eventId}><b>{event.observedAt.slice(11,19)}</b> BUS {event.busId.slice(-4)} · {event.stopId} <em>+{event.boardings} / −{event.alightings}</em></span>)}<button onClick={() => jump(0)}>Inspect pipeline →</button></section>
  </>;
}

function ForecastLab() {
  const [rain, setRain] = useState(false);
  const [eventNearby, setEventNearby] = useState(false);
  const forecast = useMemo(() => forecastOccupancy({ recentOccupancy: [42,47,51,56,61,66], capacity: 72, hour: 19, dayType: "weekday", weather: rain ? "rain" : "clear", eventNearby }), [rain, eventNearby]);
  const values = hourlyForecast.map(value => clamp(Math.round(value * (rain ? 1.07 : 1) * (eventNearby ? 1.1 : 1)), 0, 100));
  return <section className="feature-view">
    <header className="feature-hero"><div><span className="section-kicker">SCENARIO ENGINE</span><h1>See tomorrow&apos;s pressure before it arrives.</h1><p>Change the conditions. TrackBus recalculates expected occupancy and exposes exactly why the prediction changed.</p></div><div className="scenario-switches"><button className={!rain ? "active" : ""} onClick={() => setRain(false)}>☀ Clear</button><button className={rain ? "active" : ""} onClick={() => setRain(true)}>☂ Rain</button><button className={eventNearby ? "active" : ""} onClick={() => setEventNearby(!eventNearby)}>◉ Stadium event</button></div></header>
    <div className="forecast-main">
      <article className="surface forecast-chart-card"><div className="surface-head"><div><span className="section-kicker">ROUTE 22 · FRIDAY</span><h2>Expected load by hour</h2></div><span className="model-chip">86% confidence</span></div><svg className="forecast-chart" viewBox="0 0 900 360" role="img" aria-label="Forecast chart showing an evening demand peak"><defs><linearGradient id="forecast-fill" x1="0" y1="0" x2="0" y2="1"><stop offset="0" stopColor="#4ce1b6" stopOpacity=".35"/><stop offset="1" stopColor="#4ce1b6" stopOpacity="0"/></linearGradient></defs>{[70,140,210,280].map(y => <line key={y} x1="45" y1={y} x2="860" y2={y}/>)}<path className="forecast-area" d={`M45 320 ${values.map((v,i)=>`L ${45+i*(815/(values.length-1))} ${320-v*2.7}`).join(" ")} L860 320 Z`}/><polyline points={values.map((v,i)=>`${45+i*(815/(values.length-1))},${320-v*2.7}`).join(" ")}/><line className="peak-line" x1="670" y1="50" x2="670" y2="320"/><circle cx="670" cy={320-values[13]*2.7} r="7"/><text x="630" y="38">19:30 peak</text><text x="45" y="345">06:00</text><text x="420" y="345">14:00</text><text x="820" y="345">22:00</text></svg><div className="chart-caption"><span><i className="history"/>Historical pattern</span><span><i className="prediction-line"/>AI forecast</span><span><i className="confidence-band"/>Confidence range</span></div></article>
      <aside className="surface forecast-result"><span className="section-kicker">19:00 FORECAST</span><strong className="forecast-big">{forecast.expectedPercent}<small>%</small></strong><LoadBadge value={forecast.expectedOccupancy}/><div className="forecast-facts"><div><span>Passengers</span><strong>{forecast.expectedOccupancy} / 72</strong></div><div><span>Likely range</span><strong>{forecast.lowerBound}–{forecast.upperBound}</strong></div><div><span>Drivers</span><strong>{rain ? "Rain" : "Normal"}{eventNearby ? " + event" : ""}</strong></div></div><button className="approve-action"><span>Generate dispatch plan</span><b>→</b></button></aside>
    </div>
  </section>;
}

function PassengerApp() {
  return <section className="feature-view passenger-view"><header className="feature-hero"><div><span className="section-kicker">PUBLIC EXPERIENCE</span><h1>Turn operational intelligence into a calmer trip.</h1><p>Passengers see useful crowding estimates—not camera feeds, faces, or personal data.</p></div></header><div className="passenger-stage">
    <div className="phone"><div className="phone-top"><span>9:41</span><span>●●●</span></div><div className="app-brand"><div>TB</div><span><strong>TrackBus</strong><small>Tashkent</small></span><button>⌁</button></div><div className="app-search">⌕ <span>Where are you going?</span></div><div className="public-map"><div className="public-route-line"/><i className="public-stop s1"/><i className="public-stop s2"/><i className="public-stop s3"/><span className="public-bus b1">22</span><span className="public-bus b2">22</span><b>You are here</b></div><div className="nearby-title"><div><strong>Route 22</strong><small>Chorsu → Do&apos;stlik</small></div><span>See route</span></div><div className="arrival-card"><strong>2 min</strong><div><b>Bus 22-01</b><small>31 of 72 seats used</small></div><LoadBadge value={31}/></div><div className="arrival-card"><strong>7 min</strong><div><b>Bus 22-02</b><small>More space expected</small></div><LoadBadge value={27}/></div></div>
    <aside className="passenger-copy"><span className="section-kicker">ONE DATASET · TWO USERS</span><h2>Operators act.<br/>Passengers plan.</h2><p>When a reserve bus is dispatched, arrival time and crowding estimates update automatically.</p><div className="benefit-list"><div><Icon>✓</Icon><span><strong>Simple choices</strong><small>Wait 5 minutes for a less crowded bus</small></span></div><div><Icon>◎</Icon><span><strong>Privacy by design</strong><small>Only anonymous counts reach the public app</small></span></div><div><Icon>↗</Icon><span><strong>More trust</strong><small>Confidence and update time stay visible</small></span></div></div><div className="public-result"><span>After dispatch</span><strong>91% → 68%</strong><small>Expected crowding on the next departure</small></div></aside>
  </div></section>;
}

function SystemView() {
  const stages = [
    { n: "01", icon: "◉", title: "Count", text: "Existing door sensors report anonymous boardings and alightings.", meta: "No face storage" },
    { n: "02", icon: "⇄", title: "Validate", text: "TrackBus checks time, location, confidence, and count consistency.", meta: "96% data quality" },
    { n: "03", icon: "▦", title: "Forecast", text: "Route history and live events predict pressure before capacity is reached.", meta: "86% confidence" },
    { n: "04", icon: "↗", title: "Coordinate", text: "Operators receive an auditable action; passengers receive a simpler choice.", meta: "Human approval" },
  ];
  return <section className="feature-view"><header className="feature-hero"><div><span className="section-kicker">SYSTEM STORY</span><h1>Camera counts become transport decisions.</h1><p>The AI camera project continues in the background. TrackBus can start with any approved counter that sends the shared event format.</p></div><span className="synthetic-label">SYNTHETIC DEMONSTRATION</span></header><div className="system-flow">{stages.map((stage,index)=><article className="surface" key={stage.n}><span className="stage-number">{stage.n}</span><Icon>{stage.icon}</Icon><h2>{stage.title}</h2><p>{stage.text}</p><strong>{stage.meta}</strong>{index < stages.length-1 && <i>→</i>}</article>)}</div><div className="system-bottom"><article className="surface event-console"><div className="surface-head"><div><span className="section-kicker">LIVE CONTRACT</span><h2>Passenger-count events</h2></div><span className="live-chip"><i/>Streaming</span></div>{passengerEvents.map(event=><div className="console-row" key={event.eventId}><span>{event.observedAt.slice(11,19)}</span><strong>{event.busId}</strong><span>{event.stopId}</span><b>+{event.boardings} / −{event.alightings}</b><LoadBadge value={event.occupancy}/><em>{Math.round(event.qualityScore*100)}% quality</em></div>)}</article><aside className="surface privacy-card"><Icon>◇</Icon><h2>Built for public trust</h2><p>The operating contract contains counts, vehicle IDs, route IDs, timestamps, and sensor confidence.</p><ul><li>No passenger names</li><li>No public camera feeds</li><li>No facial recognition required</li><li>Every recommendation is reviewable</li></ul></aside></div></section>;
}

export function TrackBusApp() {
  const [view, setView] = useState<View>("command");
  const [live, setLive] = useState(false);
  const [tick, setTick] = useState(0);
  const [applied, setApplied] = useState(false);
  const [api, setApi] = useState<"checking" | "connected" | "offline">("checking");
  const activeStory = applied ? 3 : Math.min(2, Math.floor((tick % 12) / 4));

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
    <div className="demo-disclaimer"><span>SHOWCASE MODE</span><p>Synthetic Tashkent data · designed to demonstrate the product workflow</p><button onClick={() => setView("system")}>What is real vs simulated?</button></div>
    <header className="topbar"><button className="brand" onClick={() => setView("command")}><span>TB</span><div><strong>TrackBus</strong><small>Urban intelligence</small></div></button><nav>{navigation.map(item=><button key={item.id} className={view===item.id?"active":""} onClick={()=>setView(item.id)}>{item.label}</button>)}</nav><div className="header-status"><span className={`api-status ${api}`}><i/>{api === "connected" ? "SYSTEM ONLINE" : api === "offline" ? "OFFLINE" : "CONNECTING"}</span><div className="city-time"><strong>18:{47 + (tick % 10)}</strong><small>TASHKENT · FRI</small></div><button className={`demo-control ${live?"playing":""}`} onClick={startDemo}>{live ? "Ⅱ Pause scenario" : "▶ Run 60s scenario"}</button></div></header>
    <div className="mobile-nav">{navigation.map(item=><button key={item.id} className={view===item.id?"active":""} onClick={()=>setView(item.id)}>{item.label}</button>)}</div>
    {view === "command" && <StoryRail active={activeStory} onSelect={jumpStory}/>} 
    <div className="content-frame">{view === "command" && <CommandCenter tick={tick} live={live} applied={applied} setApplied={setApplied} jump={jumpStory}/>} {view === "forecast" && <ForecastLab/>} {view === "passenger" && <PassengerApp/>} {view === "system" && <SystemView/>}</div>
    <footer><span>TrackBus · Tashkent transport intelligence</span><span>Pilot foundation · all values in this showcase are synthetic</span></footer>
  </main>;
}
