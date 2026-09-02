export type CrowdLevelId =
  | "plenty"
  | "seats_available"
  | "moderate"
  | "crowded"
  | "full";

export type CrowdTone = "green" | "lime" | "yellow" | "orange" | "red";

export type CrowdLevelDefinition = {
  id: CrowdLevelId;
  minPercent: number;
  maxPercent: number;
  severity: number;
  tone: CrowdTone;
  icon: string;
};

export type CrowdThresholdConfig = {
  plentyMax: number;
  seatsAvailableMax: number;
  moderateMax: number;
  crowdedMax: number;
};

export const defaultCrowdThresholds: CrowdThresholdConfig = {
  plentyMax: 35,
  seatsAvailableMax: 60,
  moderateMax: 80,
  crowdedMax: 95,
};

export function crowdLevels(
  config: CrowdThresholdConfig = defaultCrowdThresholds,
): CrowdLevelDefinition[] {
  return [
    { id: "plenty", minPercent: 0, maxPercent: config.plentyMax, severity: 0, tone: "green", icon: "●" },
    { id: "seats_available", minPercent: config.plentyMax, maxPercent: config.seatsAvailableMax, severity: 1, tone: "lime", icon: "●" },
    { id: "moderate", minPercent: config.seatsAvailableMax, maxPercent: config.moderateMax, severity: 2, tone: "yellow", icon: "●" },
    { id: "crowded", minPercent: config.moderateMax, maxPercent: config.crowdedMax, severity: 3, tone: "orange", icon: "●" },
    { id: "full", minPercent: config.crowdedMax, maxPercent: 101, severity: 4, tone: "red", icon: "●" },
  ];
}

export function crowdLevelForPercent(
  percent: number,
  config: CrowdThresholdConfig = defaultCrowdThresholds,
) {
  const safePercent = Math.min(100, Math.max(0, percent));
  return crowdLevels(config).find(level => safePercent < level.maxPercent) ?? crowdLevels(config)[4];
}

export type SeatsAvailable = "yes" | "few" | "no" | "unknown";

export type PassengerCrowdReport = {
  reportId: string;
  busId: string;
  routeId: string;
  timestamp: string;
  crowdLevel: CrowdLevelId;
  seatsAvailable: SeatsAvailable;
  source: "passenger_report";
};

export type ReportDecayConfig = {
  strongUntilMinutes: number;
  mediumUntilMinutes: number;
  weakUntilMinutes: number;
  ignoreAfterMinutes: number;
  strongWeight: number;
  mediumWeight: number;
  weakWeight: number;
  fadingWeight: number;
};

export const defaultReportDecay: ReportDecayConfig = {
  strongUntilMinutes: 3,
  mediumUntilMinutes: 10,
  weakUntilMinutes: 20,
  ignoreAfterMinutes: 30,
  strongWeight: 1,
  mediumWeight: 0.55,
  weakWeight: 0.2,
  fadingWeight: 0.05,
};

export function reportTimeWeight(
  timestamp: string,
  now: Date,
  config: ReportDecayConfig = defaultReportDecay,
) {
  const ageMinutes = Math.max(0, (now.getTime() - new Date(timestamp).getTime()) / 60_000);
  if (!Number.isFinite(ageMinutes) || ageMinutes >= config.ignoreAfterMinutes) return 0;
  if (ageMinutes <= config.strongUntilMinutes) return config.strongWeight;
  if (ageMinutes <= config.mediumUntilMinutes) return config.mediumWeight;
  if (ageMinutes <= config.weakUntilMinutes) return config.weakWeight;
  return config.fadingWeight;
}

const representativePercent: Record<CrowdLevelId, number> = {
  plenty: 20,
  seats_available: 48,
  moderate: 70,
  crowded: 87,
  full: 98,
};

export type CrowdAggregationInput = {
  sensorPercent: number;
  sensorConfidence: number;
  reports: PassengerCrowdReport[];
  now: Date;
  decay?: ReportDecayConfig;
  maxCommunityInfluence?: number;
};

export type CrowdAggregationResult = {
  estimatedPercent: number;
  level: CrowdLevelDefinition;
  confidence: number;
  recentReportCount: number;
  communityInfluence: number;
  communityDirection: "higher" | "lower" | "aligned" | "none";
};

export function aggregateCrowdEstimate(input: CrowdAggregationInput): CrowdAggregationResult {
  const sensorPercent = Math.min(100, Math.max(0, input.sensorPercent));
  const weightedReports = input.reports
    .map(report => ({ report, weight: reportTimeWeight(report.timestamp, input.now, input.decay) }))
    .filter(item => item.weight > 0);
  const totalWeight = weightedReports.reduce((sum, item) => sum + item.weight, 0);
  const reportPercent = totalWeight
    ? weightedReports.reduce(
      (sum, item) => sum + representativePercent[item.report.crowdLevel] * item.weight,
      0,
    ) / totalWeight
    : sensorPercent;

  // A single fresh report contributes only 5%. Several agreeing recent reports
  // can contribute up to 18%, so community input informs but never replaces APC.
  const communityInfluence = Math.min(input.maxCommunityInfluence ?? 0.18, totalWeight * 0.05);
  const estimatedPercent = Math.round(
    sensorPercent * (1 - communityInfluence) + reportPercent * communityInfluence,
  );
  const agreement = 1 - Math.min(1, Math.abs(reportPercent - sensorPercent) / 50);
  const confidenceLift = totalWeight ? communityInfluence * agreement * 0.35 : 0;
  const confidence = Math.min(0.98, Math.max(0, input.sensorConfidence + confidenceLift));
  const difference = reportPercent - sensorPercent;

  return {
    estimatedPercent,
    level: crowdLevelForPercent(estimatedPercent),
    confidence,
    recentReportCount: weightedReports.length,
    communityInfluence,
    communityDirection: !totalWeight
      ? "none"
      : Math.abs(difference) < 5
        ? "aligned"
        : difference > 0
          ? "higher"
          : "lower",
  };
}

export type PassengerBusEstimate = {
  busId: string;
  etaMinutes: number;
  currentLevel: CrowdLevelDefinition;
  predictedLevel: CrowdLevelDefinition;
  predictedConfidence: number;
};

export type BusRecommendation =
  | { kind: "wait"; waitMinutes: number }
  | { kind: "take_next" }
  | { kind: "insufficient_data" }
  | { kind: "no_change" };

export function recommendPassengerChoice(
  first: PassengerBusEstimate,
  second: PassengerBusEstimate,
  minimumConfidence = 0.72,
): BusRecommendation {
  if (Math.min(first.predictedConfidence, second.predictedConfidence) < minimumConfidence) {
    return { kind: "insufficient_data" };
  }
  const improvement = first.predictedLevel.severity - second.predictedLevel.severity;
  if (improvement >= 1 && second.etaMinutes > first.etaMinutes) {
    return { kind: "wait", waitMinutes: second.etaMinutes - first.etaMinutes };
  }
  if (first.currentLevel.severity <= second.currentLevel.severity) return { kind: "take_next" };
  return { kind: "no_change" };
}
