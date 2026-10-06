"use client";

import { occupancyLevel, occupancyPercent } from "@/lib/contracts";
import { route22Buses } from "@/lib/sample-data";
import type { TranslationKey } from "@/lib/i18n";

type Translator = (key: TranslationKey, values?: Record<string, string | number>) => string;
const mapPositions = [[17, 19], [31, 37], [49, 53], [66, 69], [81, 82]];

export function FleetMap({ values, selected, onSelect, live, t }: { values: number[]; selected: number; onSelect: (index: number) => void; live: boolean; t: Translator }) {
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

