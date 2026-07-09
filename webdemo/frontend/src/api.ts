// Gọi backend. Diagnose chạy SLM + Gemini SONG SONG bằng Promise.allSettled → một model lỗi
// không chặn model kia (§5, §8).

import type { DemoCase, Health, NormalizedDiagnosis } from "./types";

async function postJson<T>(url: string, body: unknown): Promise<T> {
  const resp = await fetch(url, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  if (!resp.ok) throw new Error(`${url} → HTTP ${resp.status}`);
  return (await resp.json()) as T;
}

export async function getCases(): Promise<DemoCase[]> {
  const resp = await fetch("/api/cases");
  if (!resp.ok) throw new Error(`/api/cases → HTTP ${resp.status}`);
  return (await resp.json()) as DemoCase[];
}

export async function getHealth(): Promise<Health> {
  const resp = await fetch("/api/health");
  if (!resp.ok) throw new Error(`/api/health → HTTP ${resp.status}`);
  return (await resp.json()) as Health;
}

export interface DiagnoseReq {
  complaint: string;
  caseId: string | null;
  requestedOutput: string;
}

function errorResult(model: "slm" | "gemini", channel: string, msg: string): NormalizedDiagnosis {
  return {
    model,
    status: "error",
    outputChannel: channel as NormalizedDiagnosis["outputChannel"],
    rootCause: null,
    confidence: null,
    summary: "",
    evidence: [],
    customerSteps: [],
    affectedSystem: null,
    owner: null,
    escalation: null,
    runbook: null,
    similarIncident: null,
    nextActions: [],
    missingEvidence: [],
    latencyMs: null,
    isMock: false,
    errorMessage: msg,
  };
}

// Chạy 2 model song song; trả cả 2 kết quả (một cái lỗi vẫn giữ cái kia).
export async function diagnoseBoth(
  req: DiagnoseReq,
  channel: string,
): Promise<{ slm: NormalizedDiagnosis; gemini: NormalizedDiagnosis }> {
  const [slmR, dsR] = await Promise.allSettled([
    postJson<NormalizedDiagnosis>("/api/diagnose/slm", req),
    postJson<NormalizedDiagnosis>("/api/diagnose/gemini", req),
  ]);
  return {
    slm:
      slmR.status === "fulfilled"
        ? slmR.value
        : errorResult("slm", channel, `SLM lỗi: ${String(slmR.reason)}`),
    gemini:
      dsR.status === "fulfilled"
        ? dsR.value
        : errorResult("gemini", channel, `Gemini lỗi: ${String(dsR.reason)}`),
  };
}
