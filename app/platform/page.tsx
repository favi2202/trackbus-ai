import type { Metadata } from "next";
import { TrackBusApp } from "@/components/trackbus-app";

export const metadata: Metadata = { title: "Working Prototype | TrackBus AI", description: "Explore the TrackBus operator dashboard, passenger beta and forecasting baseline. Showcase scenarios use synthetic data." };

export default function Platform() { return <TrackBusApp />; }
