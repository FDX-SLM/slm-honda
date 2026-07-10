// Schema chung §10 — cả SLM và Gemini chuẩn hoá về đây trước khi render.

export type OutputChannel = "customer" | "internal";
export type SignalStatus = "ok" | "warning" | "error" | "unknown";

export interface SystemSignal {
  name: string;
  value: string;
  status: SignalStatus;
}

export interface Customer {
  name: string;
  email: string;
  phone: string;
}

export interface DemoCase {
  id: string;
  label: string;
  complaint: string;
  expectedRootCause: string | null;
  outputChannel: OutputChannel;
  customer: Customer;
  vehicleContext: Record<string, string>;
  systemSignals: SystemSignal[];
}

export interface SlmArtifacts {
  rcaMd: string | null;
  workOrderMd: string | null;
  customerEmail: string | null;
  diagramMermaid: string | null;
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
  severity: string | null;
  priority: string | null;
  artifacts: SlmArtifacts | null;
  latencyMs: number | null;
  isMock: boolean;
  errorMessage: string | null;
  // Nguyên văn output của SLM (chỉ endpoint /api/diagnose/slm trả về) để in raw ở UI.
  rawText?: string;
  rawResolution?: Record<string, unknown> | null;
  think?: string;
  // Tốc độ sinh: số token + token/s (đo từ vLLM usage / generate).
  genTokens?: number | null;
  tokensPerSec?: number | null;
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
