"use client";

import { useEffect, useMemo, useState } from "react";
import { occupancyLevel, occupancyPercent } from "@/lib/contracts";
import { forecastOccupancy } from "@/lib/forecast";
import { hourlyForecast, networkRoutes, passengerEvents, route22Buses } from "@/lib/sample-data";

type View = "operations" | "forecast" | "pipeline" | "passenger";

const viewLabels: Record<View, string> = {
  operations: "Live operations",
  forecast: "Forecast",
  pipeline: "Data pipeline",
  passenger: "Passenger view",
};

const busPoints = [
  [20, 18],
  [72, 72],
  [56, 78],
  [28, 14],
  [39, 50],
];

function OccupancyPill({ value, capacity = 72 }: { value: number; capacity?: number }) {
  const level = occupancyLevel(value, capacity);
  return <span className={`occupancy-pill ${level}`}>{occupancyPercent(value, capacity)}% full</span>;
}

function MiniTrend({ values = hourlyForecast }: { values?: number[] }) {
  const max = Math.max(...values);
  const min = Math.min(...values);
  const points = values
    .map((value, index) => {
      const x = 12 + (index / (values.length - 1)) * 376;
      const y = 126 - ((value - min) / (max - min || 1)) * 92;
      return `${x},${y}`;
    })
    .join(" ");
  return (
    <svg className="trend-chart" viewBox="0 0 400 145" role="img" aria-label="Demand forecast rises toward the evening peak">
      <line x1="12" y1="34" x2="388" y2="34" />
      <line x1="12" y1="80" x2="388" y2="80" />
      <line x1="12" y1="126" x2="388" y2="126" />
      <polyline points={points} />
      <circle cx="300" cy="65" r="5" />
      <text x="12" y="141">06:00</text><text x="180" y="141">14:00</text><text x="350" y="141">22:00</text>
    </svg>
  );
}

function RouteMap({ tick, selectedBus, onSelect }: { tick: number; selectedBus: number; onSelect: (index: number) => void }) {
  return (
    <div className="map-canvas" aria-label="Synthetic Route 22 map">
      <div className="map-grid" />
      <svg viewBox="0 0 800 520" className="route-map" role="img" aria-label="Route 22 through Chorsu, Alisher Navoiy, Mustaqillik and Oybek">
        <path className="river" d="M520 -20 C 430 95, 610 150, 520 260 S 490 420, 660 560" />
        <path className="route-line" d="M160 80 C 255 95, 280 170, 300 230 C 322 300, 360 345, 440 390 C 520 435, 570 445, 650 470" />
        {[[160,80,"Chorsu"],[300,230,"Alisher Navoiy"],[440,390,"Mustaqillik"],[650,470,"Oybek"]].map(([x,y,label]) => (
          <g key={String(label)}><circle className="stop" cx={Number(x)} cy={Number(y)} r="10" /><text className="stop-label" x={Number(x)+14} y={Number(y)+5}>{label}</text></g>
        ))}
      </svg>
      {route22Buses.map((bus, index) => {
        const [x, y] = busPoints[index];
        const drift = tick % 2 === 0 ? index * 0.35 : index * -0.2;
        return (
          <button
            type="button"
            key={bus.id}
            className={`bus-marker ${occupancyLevel(bus.occupancy, bus.capacity)} ${selectedBus === index ? "selected" : ""}`}
            style={{ left: `${x + drift}%`, top: `${y}%` }}
            onClick={() => onSelect(index)}
            aria-label={`${bus.label}, ${occupancyPercent(bus.occupancy, bus.capacity)} percent full`}
          >
            <span className="bus-symbol">▰</span>
            <strong>{bus.label}</strong>
            <small>{occupancyPercent(bus.occupancy, bus.capacity)}%</small>
          </button>
        );
      })}
      <div className="map-city">Tashkent</div>
      <div className="map-controls"><button aria-label="Zoom in">+</button><button aria-label="Zoom out">−</button></div>
      <div className="map-legend"><span className="dot low" /> low <span className="dot medium" /> medium <span className="dot high" /> high</div>
    </div>
  );
}

function OperationsView({ tick }: { tick: number }) {
  const [selectedBus, setSelectedBus] = useState(3);
  const bus = route22Buses[selectedBus];
  const average = Math.round(route22Buses.reduce((sum, item) => sum + occupancyPercent(item.occupancy, item.capacity), 0) / route22Buses.length);
  return (
    <>
      <section className="kpi-grid" aria-label="Network summary">
        <article className="kpi route-kpi"><span className="kpi-icon">22</span><div><small>Selected route</small><strong>Chorsu — Do&apos;stlik</strong></div></article>
        <article className="kpi"><span className="signal-bars">▥</span><div><small>Buses online</small><strong>8 / 8</strong><em>All devices reporting</em></div></article>
        <article className="kpi"><span className="people-icon">●●●</span><div><small>Average occupancy</small><strong>{average}%</strong><em>+6% in the last hour</em></div></article>
        <article className="kpi warning"><span className="people-icon">●●</span><div><small>High-load buses</small><strong>2</strong><em>Action recommended</em></div></article>
        <article className="kpi"><span className="clock-icon">◷</span><div><small>On-time service</small><strong>91%</strong><em>Target ≥ 90%</em></div></article>
      </section>

      <section className="operations-grid">
        <article className="panel map-panel">
          <div className="panel-heading"><div><span className="eyebrow">Live fleet</span><h2>Route 22 operations</h2></div><div className="segmented"><button className="active">Map</button><button>Stops</button></div></div>
          <RouteMap tick={tick} selectedBus={selectedBus} onSelect={setSelectedBus} />
          <div className="selected-bus"><div><span>Selected bus</span><strong>{bus.label}</strong></div><OccupancyPill value={bus.occupancy} /><div><span>Next stop</span><strong>{bus.etaMinutes} min</strong></div><div><span>Trend</span><strong>{bus.trend === "up" ? "Increasing ↗" : bus.trend === "down" ? "Falling ↘" : "Stable →"}</strong></div></div>
        </article>

        <aside className="side-stack">
          <article className="panel alerts-panel">
            <div className="panel-heading"><div><span className="eyebrow">Decision support</span><h2>Route alerts</h2></div><span className="count-badge">3</span></div>
            <button className="alert critical"><span>!</span><div><strong>High occupancy</strong><small>Bus 22-04 · 91% full</small><em>Capacity expected in 12 min</em></div><b>›</b></button>
            <button className="alert warning"><span>△</span><div><strong>Service gap</strong><small>18 min near Chorsu</small><em>Above the 12 min target</em></div><b>›</b></button>
            <button className="alert recommendation"><span>↗</span><div><strong>Recommended action</strong><small>Dispatch one reserve bus</small><em>Estimated relief: 23%</em></div><b>›</b></button>
          </article>
          <article className="panel forecast-panel"><div className="panel-heading"><div><span className="eyebrow">Next 4 hours</span><h2>Demand forecast</h2></div><span className="confidence">86% confidence</span></div><MiniTrend /><div className="forecast-summary"><strong>Peak at 19:30</strong><span>Routes 22 and 93 need attention</span></div></article>
        </aside>
      </section>
    </>
  );
}

function ForecastView() {
  const [weather, setWeather] = useState<"clear" | "rain">("clear");
  const [eventNearby, setEventNearby] = useState(false);
  const result = useMemo(() => forecastOccupancy({ recentOccupancy: [42, 47, 51, 56, 61, 66], capacity: 72, hour: 19, dayType: "weekday", weather, eventNearby }), [weather, eventNearby]);
  return (
    <section className="workspace-view">
      <div className="view-intro"><div><span className="eyebrow">Planning workspace</span><h1>Forecast demand before buses overload</h1><p>Change the scenario to see how a transparent baseline responds. Synthetic data is used throughout this showcase.</p></div><div className="scenario-controls"><button className={weather === "clear" ? "active" : ""} onClick={() => setWeather("clear")}>Clear weather</button><button className={weather === "rain" ? "active" : ""} onClick={() => setWeather("rain")}>Rain</button><button className={eventNearby ? "active" : ""} onClick={() => setEventNearby(!eventNearby)}>Nearby event</button></div></div>
      <div className="forecast-layout">
        <article className="panel large-chart"><div className="panel-heading"><div><span className="eyebrow">Route 22 · Friday</span><h2>Expected occupancy</h2></div><span className="confidence">86% model confidence</span></div><MiniTrend values={hourlyForecast.map((v) => Math.min(100, Math.round(v * (weather === "rain" ? 1.07 : 1) * (eventNearby ? 1.1 : 1))))} /><div className="chart-legend"><span><i className="line actual" /> Historical pattern</span><span><i className="line predicted" /> Forecast</span><span><i className="band" /> Confidence interval</span></div></article>
        <aside className="panel forecast-decision"><span className="decision-label">19:00 prediction</span><strong className="giant-number">{result.expectedPercent}%</strong><OccupancyPill value={result.expectedOccupancy} /><dl><div><dt>Expected passengers</dt><dd>{result.expectedOccupancy} / 72</dd></div><div><dt>Likely range</dt><dd>{result.lowerBound}–{result.upperBound}</dd></div><div><dt>Confidence</dt><dd>{Math.round(result.confidence * 100)}%</dd></div></dl><button className="primary-action">Create dispatch plan</button></aside>
      </div>
      <div className="route-table panel"><div className="panel-heading"><div><span className="eyebrow">Network view</span><h2>Routes ranked by expected load</h2></div><button className="text-button">Export forecast CSV</button></div>{networkRoutes.map((route) => <div className="route-row" key={route.id}><span className="route-number">{route.id}</span><div><strong>{route.name}</strong><small>{route.buses} buses active</small></div><div className="load-bar"><i style={{ width: `${route.average}%` }} /></div><strong>{route.average}%</strong><span className={`status ${route.status}`}>{route.status}</span></div>)}</div>
    </section>
  );
}

function PipelineView() {
  return (
    <section className="workspace-view">
      <div className="view-intro"><div><span className="eyebrow">System foundation</span><h1>From bus sensors to useful decisions</h1><p>The platform accepts counts, validates quality, builds history, and produces predictions without sending passenger images to public-facing systems.</p></div><span className="demo-label">SHOWCASE PIPELINE · SYNTHETIC EVENTS</span></div>
      <div className="pipeline-flow" aria-label="TrackBus data pipeline">
        {[{n:"01",t:"On-bus counter",d:"Door camera or APC reports boardings and alightings",m:"3 events/s"},{n:"02",t:"Edge gateway",d:"Adds bus, route, stop, time and quality metadata",m:"96% quality"},{n:"03",t:"Data platform",d:"Validates, stores and aggregates occupancy history",m:"184k events"},{n:"04",t:"Forecast service",d:"Predicts load and suggests operator actions",m:"86% confidence"}].map((step, index) => <article className="pipeline-step" key={step.n}><span>{step.n}</span><div className="pipeline-icon">{index === 0 ? "◉" : index === 1 ? "⇄" : index === 2 ? "▤" : "↗"}</div><h2>{step.t}</h2><p>{step.d}</p><strong>{step.m}</strong>{index < 3 && <i>→</i>}</article>)}
      </div>
      <div className="pipeline-grid">
        <article className="panel event-stream"><div className="panel-heading"><div><span className="eyebrow">Validated input</span><h2>Latest passenger-count events</h2></div><span className="live-badge"><i /> live</span></div><div className="event-head"><span>Time</span><span>Bus</span><span>Stop</span><span>In / out</span><span>Occupancy</span><span>Quality</span></div>{passengerEvents.map((event) => <div className="event-row" key={event.eventId}><span>{event.observedAt.slice(11,19)}</span><strong>{event.busId.replace("BUS-", "")}</strong><span>{event.stopId}</span><span className="in-out"><b>+{event.boardings}</b> / −{event.alightings}</span><OccupancyPill value={event.occupancy} capacity={event.capacity} /><strong>{Math.round(event.qualityScore * 100)}%</strong></div>)}</article>
        <aside className="panel quality-card"><div className="panel-heading"><div><span className="eyebrow">Data readiness</span><h2>Quality gates</h2></div></div>{[["Sensor connectivity",98],["Count consistency",94],["Location matched",99],["Freshness",96]].map(([label, value]) => <div className="quality-meter" key={String(label)}><div><span>{label}</span><strong>{value}%</strong></div><i><b style={{width:`${value}%`}} /></i></div>)}<div className="privacy-note"><strong>Privacy by design</strong><p>The core contract stores anonymous counts and operational metadata—not faces or passenger identities.</p></div></aside>
      </div>
    </section>
  );
}

function PassengerView() {
  return (
    <section className="passenger-shell">
      <div className="passenger-intro"><span className="eyebrow">Public information concept</span><h1>Choose a less crowded bus</h1><p>Estimated crowding helps passengers plan while giving operators a shared view of demand.</p><div className="search-box"><span>⌕</span><input aria-label="Search route or stop" placeholder="Search route or stop" defaultValue="Route 22" /><button>Search</button></div></div>
      <div className="passenger-grid">
        <article className="public-route-card"><div className="public-route-head"><span className="route-number large">22</span><div><h2>Chorsu → Do&apos;stlik</h2><p>8 buses operating · service every 9–12 min</p></div><span className="service-good">On time</span></div><div className="arrival-list">{route22Buses.slice(0,4).map((bus) => <div className="arrival" key={bus.id}><span className="bus-public">▰</span><div><strong>Bus {bus.label}</strong><small>{bus.etaMinutes === 2 ? "Arriving soon" : `${bus.etaMinutes} minutes away`}</small></div><OccupancyPill value={bus.occupancy} /><button aria-label={`Open details for bus ${bus.label}`}>›</button></div>)}</div></article>
        <aside className="public-info"><div className="phone-map"><div className="mini-route"><i /><i /><i /><i /></div><span className="mini-bus one">22</span><span className="mini-bus two">22</span><div className="you-are-here">You are here</div></div><div className="legend-card"><h3>How crowding estimates work</h3><p>Anonymous passenger counts are combined with route history and live vehicle data.</p><div><OccupancyPill value={20} /><OccupancyPill value={42} /><OccupancyPill value={66} /></div></div></aside>
      </div>
    </section>
  );
}

export function TrackBusApp() {
  const [view, setView] = useState<View>("operations");
  const [live, setLive] = useState(false);
  const [tick, setTick] = useState(0);
  useEffect(() => {
    if (!live) return;
    const timer = window.setInterval(() => setTick((value) => value + 1), 1800);
    return () => window.clearInterval(timer);
  }, [live]);

  return (
    <main className="app-shell">
      <div className="showcase-ribbon"><strong>DEMO FOR SHOWCASING</strong><span>Synthetic Tashkent data · not connected to live government systems</span></div>
      <header className="topbar"><button className="brand" onClick={() => setView("operations")}><span>TB</span><div><strong>TrackBus</strong><small>Transport intelligence</small></div></button><nav aria-label="Main navigation">{(Object.keys(viewLabels) as View[]).map((item) => <button key={item} className={view === item ? "active" : ""} onClick={() => setView(item)}>{viewLabels[item]}</button>)}</nav><div className="top-actions"><span className="updated"><i /> Updated 17:{32 + (tick % 9)}</span><button className={`live-button ${live ? "running" : ""}`} onClick={() => setLive(!live)}>{live ? "■ Pause demo" : "▶ Start live demo"}</button></div></header>
      <div className="mobile-tabs" aria-label="Mobile navigation">{(Object.keys(viewLabels) as View[]).map((item) => <button key={item} className={view === item ? "active" : ""} onClick={() => setView(item)}>{viewLabels[item]}</button>)}</div>
      <div className="page-frame">{view === "operations" && <OperationsView tick={tick} />}{view === "forecast" && <ForecastView />}{view === "pipeline" && <PipelineView />}{view === "passenger" && <PassengerView />}</div>
      <footer><span>TrackBus foundation · Pilot milestone 01</span><span>All dashboard values are synthetic showcase data</span></footer>
    </main>
  );
}
