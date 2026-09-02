"use client";

import { useEffect, useMemo, useState } from "react";

import {
  recommendPassengerChoice,
  type CrowdLevelDefinition,
  type CrowdLevelId,
  type PassengerCrowdReport,
  type SeatsAvailable,
} from "@/lib/crowding";
import {
  createDemoPassengerReports,
  estimatePassengerBus,
  passengerDemoBuses,
} from "@/lib/passenger-demo";
import type { TranslationKey } from "@/lib/i18n";

type Translator = (key: TranslationKey, values?: Record<string, string | number>) => string;

const INITIAL_NOW = new Date("2000-01-01T00:00:00.000Z");

const crowdLabelKeys: Record<CrowdLevelId, TranslationKey> = {
  plenty: "crowdPlenty",
  seats_available: "crowdSeatsAvailable",
  moderate: "crowdModerate",
  crowded: "crowdCrowded",
  full: "crowdFull",
};

const reportLabelKeys: Record<"plenty" | "moderate" | "crowded" | "full", TranslationKey> = {
  plenty: "reportPlenty",
  moderate: "reportModerate",
  crowded: "reportCrowded",
  full: "reportPacked",
};

function CrowdStatus({ level, t, compact = false }: { level: CrowdLevelDefinition; t: Translator; compact?: boolean }) {
  return <span className={`passenger-crowd crowd-${level.tone} ${compact ? "compact" : ""}`}><i aria-hidden="true">{level.icon}</i><strong>{t(crowdLabelKeys[level.id])}</strong></span>;
}

export function PassengerBeta({ t }: { t: Translator }) {
  const [now, setNow] = useState(INITIAL_NOW);
  const [reports, setReports] = useState<PassengerCrowdReport[]>([]);
  const [reportBusId, setReportBusId] = useState("22-04");
  const [seatsAvailable, setSeatsAvailable] = useState<SeatsAvailable>("few");
  const [lastReport, setLastReport] = useState<PassengerCrowdReport | null>(null);

  useEffect(() => {
    const firstUpdate = window.setTimeout(() => {
      const current = new Date();
      setNow(current);
      setReports(createDemoPassengerReports(current));
    }, 0);
    const timer = window.setInterval(() => setNow(new Date()), 60_000);
    return () => {
      window.clearTimeout(firstUpdate);
      window.clearInterval(timer);
    };
  }, []);

  const estimates = useMemo(
    () => passengerDemoBuses.map(bus => estimatePassengerBus(bus, reports, now)),
    [reports, now],
  );
  const recommendation = recommendPassengerChoice(
    {
      busId: estimates[0].busId,
      etaMinutes: estimates[0].etaMinutes,
      currentLevel: estimates[0].current.level,
      predictedLevel: estimates[0].predictedLevel,
      predictedConfidence: estimates[0].predictedConfidence,
    },
    {
      busId: estimates[1].busId,
      etaMinutes: estimates[1].etaMinutes,
      currentLevel: estimates[1].current.level,
      predictedLevel: estimates[1].predictedLevel,
      predictedConfidence: estimates[1].predictedConfidence,
    },
  );
  const selectedEstimate = estimates.find(bus => bus.busId === reportBusId) ?? estimates[0];

  function submitReport(crowdLevel: "plenty" | "moderate" | "crowded" | "full") {
    const timestamp = new Date();
    const event: PassengerCrowdReport = {
      reportId: `rpt-${timestamp.getTime()}`,
      busId: reportBusId,
      routeId: "22",
      timestamp: timestamp.toISOString(),
      crowdLevel,
      seatsAvailable,
      source: "passenger_report",
    };
    setReports(current => [...current, event]);
    setLastReport(event);
    setNow(timestamp);
  }

  return <section className="feature-view passenger-view">
    <header className="feature-hero passenger-hero"><div><span className="section-kicker">{t("publicExperience")}</span><h1>{t("passengerHero")}</h1><p>{t("passengerHeroDetail")}</p></div><span className="synthetic-label">{t("passengerBetaDemo")}</span></header>
    <div className="passenger-stage">
      <div className="passenger-phone" aria-label={t("passengerBetaAria")}>
        <div className="phone-speaker"/>
        <div className="passenger-screen">
          <div className="phone-top"><span>9:41</span><span>●●●</span></div>
          <div className="app-brand"><div>TB</div><span><strong>TrackBus</strong><small>{t("passengerBeta")}</small></span><button aria-label={t("seeRoute")}>⌁</button></div>
          <button className="app-search" type="button"><span>⌕</span><b>{t("whereGoing")}</b></button>
          <div className="passenger-route-head"><div><span>{t("route22")}</span><strong>Mustaqillik → Chilonzor</strong></div><em>{t("syntheticScenario")}</em></div>

          <div className="passenger-bus-list">
            {estimates.map((bus, index) => <article className={`passenger-bus-card ${index === 1 ? "recommended" : ""}`} key={bus.busId}>
              <header><div><span>{t("busName", { bus: bus.busId })}</span><small>{index === 0 ? t("nextBus") : t("alternativeBus")}</small></div><strong>{t("minutesShort", { minutes: bus.etaMinutes })}</strong></header>
              <div className="bus-current"><div><small>{t("currentStatus")}</small><CrowdStatus level={bus.current.level} t={t}/></div><span>{t("estimatedOccupancy", { percent: bus.current.estimatedPercent })}</span></div>
              <div className="stop-prediction"><div><small>{t("atYourStop")}</small><CrowdStatus level={bus.predictedLevel} t={t} compact/></div><div><b>{t("predicted")}</b><span>{t("confidenceValue", { percent: Math.round(bus.predictedConfidence * 100) })}</span></div></div>
              <footer><span>{t("recentReports", { count: bus.current.recentReportCount })}</span><span>{t("updatedRecently")}</span></footer>
            </article>)}
          </div>

          <div className={`passenger-recommendation ${recommendation.kind}`}><span aria-hidden="true">↳</span><div><strong>{recommendation.kind === "wait" ? t("waitRecommendation", { minutes: recommendation.waitMinutes }) : recommendation.kind === "insufficient_data" ? t("notEnoughData") : t("nextBusRecommendation")}</strong><small>{recommendation.kind === "wait" ? t("waitRecommendationDetail", { bus: estimates[1].busId }) : t("recommendationCaution")}</small></div></div>

          <div className="historical-note"><span>{t("passengerHistorical")}</span><p>{t("usualCrowdingPattern")}</p></div>

          <section className="community-report">
            <div className="community-heading"><div><span>{t("communityBeta")}</span><h3>{t("areYouOnBus")}</h3></div><b>{t("oneTapAnonymous")}</b></div>
            <div className="report-bus-choice">{estimates.map(bus => <button type="button" className={reportBusId === bus.busId ? "active" : ""} onClick={() => setReportBusId(bus.busId)} key={bus.busId}>{bus.busId}</button>)}</div>
            <p>{t("howCrowded")}</p>
            <div className="crowd-report-grid">{(["plenty", "moderate", "crowded", "full"] as const).map(level => <button type="button" className={`report-${level}`} onClick={() => submitReport(level)} key={level}><i>●</i><span>{t(reportLabelKeys[level])}</span></button>)}</div>
            <div className="seat-question"><span>{t("seatsAvailableQuestion")}</span><div>{(["yes", "few", "no"] as const).map(value => <button type="button" className={seatsAvailable === value ? "active" : ""} onClick={() => setSeatsAvailable(value)} key={value}>{t(value === "yes" ? "seatsYes" : value === "few" ? "seatsFew" : "seatsNo")}</button>)}</div></div>
            {lastReport && <div className="report-thanks" role="status"><span>✓</span><div><strong>{t("reportThanks")}</strong><small>{t("reportSignalUpdated", { bus: lastReport.busId, count: selectedEstimate.current.recentReportCount })}</small></div></div>}
          </section>

          <details className="estimate-details"><summary>{t("estimateDetails")}</summary><div><strong>{t("highConfidence")}</strong><ul><li>{t("recentVehicleData")}</li><li>{t("passengerReportsSource")}</li><li>{t("routeHistory")}</li></ul><small>{t("communityDoesNotOverride")}</small></div></details>
        </div>
      </div>

      <aside className="passenger-copy">
        <span className="section-kicker">{t("passengerValue")}</span>
        <h2>{t("knowBeforeArrival")}</h2>
        <p>{t("knowBeforeArrivalDetail")}</p>
        <div className="passenger-value-list">
          <article><span>01</span><div><small>{t("liveLabel")}</small><strong>{t("compareApproaching")}</strong><p>{t("compareApproachingDetail")}</p></div></article>
          <article><span>02</span><div><small>{t("predictedLabel")}</small><strong>{t("forecastAtStop")}</strong><p>{t("forecastAtStopDetail")}</p></div></article>
          <article><span>03</span><div><small>{t("communityLabel")}</small><strong>{t("communitySignal")}</strong><p>{t("communitySignalDetail")}</p></div></article>
        </div>
        <div className="signal-blend"><span>{t("trackbusEstimate")}</span><div><b>{t("sensorData")}</b><i>+</i><b>{t("communityReports")}</b><i>+</i><b>{t("passengerHistoricalPattern")}</b><i>→</i><strong>{t("simpleCrowdStatus")}</strong></div><small>{t("privacyNoIdentity")}</small></div>
      </aside>
    </div>
  </section>;
}
