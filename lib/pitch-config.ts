/** Edit pitch content here. Set DEMO_VIDEO_URL after recording a 1–5 minute demo. */
export const DEMO_VIDEO_URL = "";

export const pitchConfig = {
  name: "TrackBus AI",
  status: "MVP / Research Prototype",
  siteUrl: "https://trackbus-showcase.favi-2202.chatgpt.site",
  repositoryUrl: "https://github.com/favi2202/trackbus-ai",
  documentationUrl: "https://github.com/favi2202/trackbus-ai/tree/pitch-day-3.0/docs",
  guideUrl: "/TrackBus-Technical-Guide.pdf",
  prototypeUrl: "/platform",
  team: [
    {
      name: "Favi",
      role: "Founder / Product & Engineering",
      initials: "F",
      description: "Building the path from passenger-flow signals to decisions operators can inspect.",
      responsibilities: ["Product direction", "Vision experiments", "Backend & API", "Dashboard & passenger UX", "Deployment", "Evaluation"],
      skills: ["Python", "Computer vision", "YOLO", "ByteTrack", "FastAPI", "Next.js / React", "Cloudflare", "Data analysis", "Cybersecurity & systems"],
      links: [{ label: "GitHub", href: "https://github.com/favi2202" }],
    },
  ],
  technologies: ["Python", "YOLO", "ByteTrack", "OpenCV", "Next.js / Vinext", "React", "TypeScript", "Cloudflare Workers", "D1", "GitHub Actions"],
} as const;

export const pipeline = [
  { title: "Collect signals", label: "Sensors · CCTV · Vision · GPS", detail: "Start with approved APC sensors where available. Existing CCTV or TrackBus Vision can generate counts; GPS and historical operations provide context. These are supported or planned source pathways, not claimed transport integrations." },
  { title: "Normalize events", label: "Anonymous IN / OUT events", detail: "Detection and temporary tracks produce directional crossing events. The canonical JSON contract carries counts, time, operational IDs, confidence and quality flags—no faces or passenger identities." },
  { title: "Connect the API", label: "Validated · authenticated · retryable", detail: "The event API validates the contract and authenticates devices. Stable event IDs prevent duplicate ingestion, while the Vision client queues events during a connection loss." },
  { title: "Understand occupancy", label: "Load estimates · quality checks", detail: "Boarding and alighting events inform occupancy estimates. Source freshness and quality checks help operators distinguish a useful signal from missing or unreliable data." },
  { title: "Anticipate demand", label: "Transparent forecasting baseline", detail: "The current forecast uses recent occupancy and explicit time, weather and event factors. Its confidence and intervals are heuristic; future models must be trained and evaluated against representative history." },
  { title: "Support decisions", label: "Operator insight · passenger information", detail: "Investigate repeated crowding, capacity imbalance and possible interventions. Passenger views show simple crowding bands. Recommendations remain advisory; operators approve changes." },
] as const;

export const roadmap = [
  { name: "Idea", state: "Completed", description: "Define the transport signal.", items: ["Crowding problem defined", "System architecture", "Anonymous counting approach"] },
  { name: "Prototype", state: "Completed", description: "Connect the working pieces.", items: ["YOLO + ByteTrack", "IN / OUT event pipeline", "Operator dashboard", "Passenger beta"] },
  { name: "MVP", state: "Current stage", description: "Make the workflow reliable.", items: ["Improve counting reliability", "Integrate multiple sources", "Refine operator workflow", "Evaluate forecast baseline", "Improve mobile UX"] },
  { name: "Pilot", state: "Next", description: "Measure in the real world.", items: ["Privacy-approved footage", "Manual ground truth", "Approximately two buses", "Network & hardware measurements", "Compare APC where available"] },
  { name: "Launched", state: "Future", description: "Scale after validation.", items: ["Operator integration", "Production monitoring", "Larger fleet rollout", "Trained demand models", "Passenger integrations"] },
] as const;

export const implementation = [
  { title: "Data", body: "Obtain representative footage or APC records with documented permission, access and retention.", tools: "Approved footage · APC · GPS" },
  { title: "Validation", body: "Label boarding and alighting events manually. Split representative ground truth for tuning and evaluation.", tools: "Annotation tools · evaluation harness" },
  { title: "Vision", body: "Calibrate doorway geometry; improve detection, track continuity and directional crossing logic.", tools: "YOLO · ByteTrack · OpenCV · Python" },
  { title: "Edge / cloud", body: "Compare on-bus inference with server inference using measured hardware cost, latency and connectivity.", tools: "Edge compute · network measurements" },
  { title: "Platform", body: "Normalize sources, calculate occupancy, surface quality checks and evaluate forecasts in the operator workflow.", tools: "API · Workers · D1 · React" },
  { title: "Pilot", body: "Plan a controlled pilot on approximately two buses with manual counts and an auditable failure log.", tools: "Ground truth · reliability measurements" },
  { title: "Scale", body: "Add vehicles and data sources after validation; train demand models on real operational history.", tools: "Monitoring · integrations · model evaluation" },
] as const;

export const demoChapters = [
  ["Detect", "TrackBus Vision detects people near a bus doorway."],
  ["Track", "Temporary anonymous tracks establish movement direction."],
  ["Count", "Confirmed boarding and alighting events reach the API."],
  ["Update", "Occupancy estimates update from accepted count events."],
  ["Understand", "The operator dashboard visualizes fleet conditions."],
  ["Forecast", "The baseline illustrates future crowding risk."],
  ["Evaluate", "An operator explores a possible intervention in the synthetic scenario."],
  ["Inform", "The passenger beta communicates crowding in simple bands."],
] as const;

export const projectQuestions = [
  { question: "What problem does TrackBus solve?", answer: "Transport capacity and demand can be poorly matched. TrackBus brings passenger-flow signals into one picture so operators can investigate crowding, spare capacity and fleet imbalance, while passengers get clearer crowding information." },
  { question: "Is TrackBus just a passenger-counting camera?", answer: "Counting is one input. The platform connects anonymous events to occupancy, quality checks, forecasting and transport decisions. Approved existing APC or CCTV can be used where available; TrackBus Vision is an optional counting path." },
  { question: "Does TrackBus use facial recognition?", answer: "No. Vision uses temporary process-local track IDs. There is no facial recognition, biometric matching or persistent passenger identity tracking. The event API receives counts and operational metadata, not video frames." },
  { question: "What stage is the project at?", answer: "MVP / research prototype. The software foundation is working, but production counting accuracy, operator value and real-world reliability still require representative validation. No completed live pilot or official Tashkent transport integration is claimed." },
  { question: "How would a pilot work?", answer: "Obtain approved data, establish manual ground truth, calibrate the doorway model and plan a controlled deployment on approximately two buses. Measure counting errors, network availability and hardware constraints, then compare with existing APC where available." },
  { question: "What hardware is required?", answer: "An approved APC feed may already provide counts. Vision needs a suitable doorway camera plus edge or server compute. The final hardware and inference location will be chosen from pilot measurements; Cloudflare receives events rather than running YOLO." },
  { question: "How does forecasting work?", answer: "The current transparent baseline weights recent occupancy and applies explicit rush-hour, weather and nearby-event factors. Confidence values and ranges are heuristic, not calibrated production accuracy. Later models must beat this baseline on held-out operational history." },
  { question: "What happens if the network disconnects?", answer: "The Vision client has an offline event queue and retries delivery. Stable event IDs make ingestion idempotent. Operators should inspect source freshness; a delayed signal must not be presented as current data." },
] as const;
