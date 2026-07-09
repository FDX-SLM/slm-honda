// Schema chung §10 — cả SLM và Gemini chuẩn hoá về đây trước khi render.

export type OutputChannel = "customer" | "internal";
export type SignalStatus = "ok" | "warning" | "error" | "unknown";

export interface SystemSignal {
  name: string;
  value: string;
  status: SignalStatus;
}

export interface DemoCase {
  id: string;
  label: string;
  complaint: string;
  expectedRootCause: string | null;
  outputChannel: OutputChannel;
  vehicleContext: Record<string, string>;
  systemSignals: SystemSignal[];
}

export interface NormalizedDiagnosis {
  model: "slm" | "gemini";
  status: "success" | "abstain" | "error";
  outputChannel: OutputChannel;
  rootCause: string | null;
  confidence: number | null;
  summary: string;
  evidence: string[];
  customerSteps: string[];
  affectedSystem: string | null;
  owner: string | null;
  escalation: string | null;
  runbook: string | null;
  similarIncident: string | null;
  nextActions: string[];
  missingEvidence: string[];
  latencyMs: number | null;
  isMock: boolean;
  errorMessage: string | null;
}

export type DiagnosisState = "idle" | "running" | "complete" | "partial" | "error";
export type ActiveMainTab = "customer" | "internal";

export interface Health {
  modelLoaded: boolean;
  base: string;
  adapter: string;
  mode: string;
  loadError: string | null;
}
