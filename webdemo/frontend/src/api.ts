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
    severity: null,
    priority: null,
    artifacts: null,
    latencyMs: null,
    isMock: false,
    errorMessage: msg,
  };
}

// Streaming SLM: gọi SSE endpoint, đẩy từng token qua onToken, kết quả cuối qua onDone.
export async function diagnoseSlmStream(
  req: DiagnoseReq,
  onToken: (t: string) => void,
  onDone: (d: NormalizedDiagnosis) => void,
  onError: (msg: string) => void,
): Promise<void> {
  const resp = await fetch("/api/diagnose/slm/stream", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(req),
  });
  if (!resp.ok || !resp.body) {
    onError(`stream → HTTP ${resp.status}`);
    return;
  }
  const reader = resp.body.getReader();
  const decoder = new TextDecoder();
  let buf = "";
  for (;;) {
    const { done, value } = await reader.read();
    if (done) break;
    buf += decoder.decode(value, { stream: true });
    // Tách theo frame SSE ("\n\n"); mỗi frame là một dòng "data: {...}".
    let idx: number;
    while ((idx = buf.indexOf("\n\n")) !== -1) {
      const frame = buf.slice(0, idx).trim();
      buf = buf.slice(idx + 2);
      if (!frame.startsWith("data:")) continue;
      const payload = frame.slice(5).trim();
      let msg: { type: string; text?: string; result?: NormalizedDiagnosis; message?: string };
      try {
        msg = JSON.parse(payload);
      } catch {
        continue;
      }
      if (msg.type === "token" && msg.text) onToken(msg.text);
      else if (msg.type === "done" && msg.result) onDone(msg.result);
      else if (msg.type === "error") onError(msg.message || "stream error");
    }
  }
}

// Chỉ chạy SLM (đã bỏ so sánh Gemini) → trả nguyên văn output để UI parse & in raw.
export async function diagnoseSlm(req: DiagnoseReq): Promise<NormalizedDiagnosis> {
  return postJson<NormalizedDiagnosis>("/api/diagnose/slm", req);
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
