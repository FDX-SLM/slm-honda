import { useEffect, useMemo, useRef, useState, type ReactNode } from "react";
import { diagnoseSlmStream, getCases } from "./api";
import type {
  ActiveMainTab,
  DemoCase,
  DiagnosisState,
  NormalizedDiagnosis,
  ResolutionPlan,
} from "./types";

const TAB_LABEL: Record<ActiveMainTab, string> = {
  customer: "Customer guidance",
  internal: "Internal diagnosis",
};

/* -------------------------- Tiny markdown render -------------------------- */
/* SLM artifacts (RCA / work order) là markdown: `## heading`, `**bold**`, `- bullet`. */

function renderInline(text: string): ReactNode[] {
  return text.split(/\*\*/).map((part, i) =>
    i % 2 === 1 ? <strong key={i}>{part}</strong> : <span key={i}>{part}</span>,
  );
}

function Markdown({ text }: { text: string }) {
  const blocks: ReactNode[] = [];
  let bullets: string[] = [];
  const flush = (key: string) => {
    if (bullets.length) {
      const items = bullets;
      blocks.push(
        <ul className="md-ul" key={key}>
          {items.map((b, i) => (
            <li key={i}>{renderInline(b)}</li>
          ))}
        </ul>,
      );
      bullets = [];
    }
  };
  text.split("\n").forEach((raw, idx) => {
    const line = raw.trimEnd();
    if (/^#{1,6}\s/.test(line)) {
      flush(`f${idx}`);
      blocks.push(
        <div className="md-h" key={idx}>
          {renderInline(line.replace(/^#{1,6}\s/, ""))}
        </div>,
      );
    } else if (/^[-*]\s/.test(line)) {
      bullets.push(line.replace(/^[-*]\s/, ""));
    } else if (line.trim() === "") {
      flush(`f${idx}`);
    } else {
      flush(`f${idx}`);
      blocks.push(
        <p className="md-p" key={idx}>
          {renderInline(line)}
        </p>,
      );
    }
  });
  flush("end");
  return <div className="md">{blocks}</div>;
}

function fmtLatency(ms: number | null): string {
  if (ms == null) return "—";
  return ms >= 1000 ? `${(ms / 1000).toFixed(1)}s` : `${ms}ms`;
}

function fmtConfidence(c: number | null): string {
  return c == null ? "—" : c.toFixed(2);
}

// Kênh hiển thị = do chính SLM quyết: TCU_OFFLINE → hướng dẫn khách; còn lại → chẩn đoán nội bộ.
function channelForRc(rc: string | null): ActiveMainTab {
  return rc === "TCU_OFFLINE" ? "customer" : "internal";
}

/* ------------------------------ Input card ------------------------------ */

function CustomerInfo({ c }: { c: DemoCase }) {
  const rows: [string, string][] = [
    ["Customer", c.customer.name],
    ["Email", c.customer.email],
    ["Phone", c.customer.phone],
    ["Vehicle", c.vehicleContext.model],
    ["Region", c.vehicleContext.region],
    ["VIN", c.vehicleContext.vin],
  ];
  return (
    <div className="cust-grid">
      {rows.map(([label, value]) => (
        <div className="field" key={label}>
          <span className="field-label">{label}</span>
          <span className="field-value">{value || "—"}</span>
        </div>
      ))}
    </div>
  );
}

/* ---------------------------- Result views ------------------------------ */
/* Chỉ render đúng những field SLM trả ra — không thêm thắt. */

function CustomerView({ d }: { d: NormalizedDiagnosis }) {
  return (
    <>
      {d.summary && <p className="summary">{d.summary}</p>}
      {d.customerSteps.length > 0 && (
        <ol className="steps">
          {d.customerSteps.map((s, i) => (
            <li key={i}>{s}</li>
          ))}
        </ol>
      )}
    </>
  );
}

function Field({ label, value }: { label: string; value: string | null }) {
  if (!value) return null;
  return (
    <div className="field">
      <span className="field-label">{label}</span>
      <span className="field-value">{value}</span>
    </div>
  );
}

function InternalView({ d }: { d: NormalizedDiagnosis }) {
  if (d.status === "abstain") {
    return (
      <>
        <div className="verdict">
          <span className="rc abstain">{d.rootCause || "INSUFFICIENT_EVIDENCE"}</span>
          <span className="conf">confidence {fmtConfidence(d.confidence)}</span>
        </div>
        {d.summary && <p className="summary">{d.summary}</p>}
        {d.missingEvidence.length > 0 && (
          <>
            <div className="block-label">To confirm</div>
            <ul className="bullets todo">
              {d.missingEvidence.map((m, i) => (
                <li key={i}>{m}</li>
              ))}
            </ul>
          </>
        )}
      </>
    );
  }
  const isCache = d.rootCause === "ENTITLEMENT_CACHE_STALE";
  return (
    <>
      <div className="verdict">
        <span className="rc">{d.rootCause || "—"}</span>
        <span className="conf">confidence {fmtConfidence(d.confidence)}</span>
        {d.runbook && <span className="tag">{d.runbook}</span>}
      </div>
      {d.summary && <p className="summary muted">{d.summary}</p>}
      {d.evidence.length > 0 && (
        <>
          <div className="block-label">Evidence</div>
          <ul className="bullets">
            {d.evidence.map((e, i) => (
              <li key={i}>{e}</li>
            ))}
          </ul>
        </>
      )}
      <div className="fields">
        <Field label="Affected system" value={d.affectedSystem} />
        <Field label="Owner" value={d.owner} />
        <Field label="Escalation" value={d.escalation} />
        {isCache && <Field label="SLA" value={d.resolutionPlan?.eta ?? null} />}
        <Field label="Similar incident" value={d.similarIncident} />
        <Field label="Severity" value={d.severity} />
        <Field label="Priority" value={d.priority} />
      </div>
      {/* Ca eligibility (non-cache): Next actions chuyển xuống NGAY SAU khối RCA (slot afterRca). */}
      <Artifacts
        a={d.artifacts}
        afterRca={
          !isCache && d.nextActions.length > 0 ? (
            <>
              <div className="block-label">Next actions</div>
              <ol className="steps">
                {d.nextActions.map((a, i) => (
                  <li key={i}>{a}</li>
                ))}
              </ol>
            </>
          ) : null
        }
      />
      {/* Resolvement Planning (L2) + de-dup CHỈ áp cho ca cache-stale; các ca khác giữ nguyên như cũ. */}
      {isCache && <ResolutionPlanPanel p={d.resolutionPlan} />}
    </>
  );
}

function ResolutionPlanPanel({ p }: { p?: ResolutionPlan | null }) {
  if (!p || p.steps.length === 0) return null;
  return (
    <div className="plan">
      <div className="plan-title">Resolvement Planning</div>
      {p.preconditions.length > 0 && (
        <div className="plan-pre">
          <span className="field-label">Preconditions</span>
          <ul className="bullets">
            {p.preconditions.map((s, i) => (
              <li key={i}>{s}</li>
            ))}
          </ul>
        </div>
      )}
      <ol className="steps plan-steps">
        {p.steps.map((s, i) => (
          <li key={i}>{s}</li>
        ))}
      </ol>
    </div>
  );
}

function Artifacts({
  a,
  afterRca = null,
}: {
  a: NormalizedDiagnosis["artifacts"];
  afterRca?: ReactNode;
}) {
  if (!a) return null;
  return (
    <>
      {a.rcaMd && (
        <>
          <div className="block-label">RCA</div>
          <div className="artifact-md">
            <Markdown text={a.rcaMd} />
          </div>
        </>
      )}
      {afterRca}
      {a.customerEmail && (
        <>
          <div className="block-label">Customer email</div>
          <div className="artifact-md email">
            <Markdown text={a.customerEmail} />
          </div>
        </>
      )}
      {a.diagramMermaid && (
        <>
          <div className="block-label">Diagram (mermaid)</div>
          <pre className="artifact mermaid">{a.diagramMermaid}</pre>
        </>
      )}
    </>
  );
}

/* --------------------------- Raw SLM output ----------------------------- */

function RawBlock({ d }: { d: NormalizedDiagnosis }) {
  // Khi xong: in <think> (nếu có) + JSON đã pretty cho dễ đọc. Parse fail → fallback rawText thô.
  const pretty = d.rawResolution ? JSON.stringify(d.rawResolution, null, 2) : null;
  if (!pretty && !d.rawText) return null;
  return (
    <section className="card raw-card">
      <div className="card-sub">Raw SLM output</div>
      {d.think && <pre className="raw think">{`<think>\n${d.think}\n</think>`}</pre>}
      <pre className="raw">{pretty ?? d.rawText}</pre>
    </section>
  );
}

/* --------------------------------- App --------------------------------- */

export default function App() {
  const [cases, setCases] = useState<DemoCase[]>([]);
  const [selectedId, setSelectedId] = useState<string>("tcu-offline");
  const [complaint, setComplaint] = useState<string>("");
  const [state, setState] = useState<DiagnosisState>("idle");
  const [result, setResult] = useState<NormalizedDiagnosis | null>(null);
  const [streamText, setStreamText] = useState<string>("");
  // Gom token vào ref, flush ra state tối đa ~mỗi 80ms → tránh re-render cả khối text 1000+ lần (lag).
  const streamBufRef = useRef<string>("");
  const flushTimerRef = useRef<number | null>(null);

  useEffect(() => {
    getCases()
      .then((cs) => {
        setCases(cs);
        const first = cs.find((c) => c.id === "tcu-offline") ?? cs[0];
        if (first) {
          setSelectedId(first.id);
          setComplaint(first.complaint);
        }
      })
      .catch(() => setCases([]));
  }, []);

  const selected = useMemo(() => cases.find((c) => c.id === selectedId), [cases, selectedId]);

  function selectCase(c: DemoCase) {
    setSelectedId(c.id);
    setComplaint(c.complaint);
    setResult(null); // xoá kết quả cũ khi đổi ca
    setStreamText("");
    setState("idle");
  }

  function stopFlush() {
    if (flushTimerRef.current != null) {
      window.clearTimeout(flushTimerRef.current);
      flushTimerRef.current = null;
    }
  }

  async function diagnose() {
    if (!complaint.trim() || state === "running") return;
    setState("running");
    setResult(null);
    setStreamText("");
    streamBufRef.current = "";
    stopFlush();
    try {
      // Stream full text trước (từng token), khi xong mới có NormalizedDiagnosis để render + JSON.
      await diagnoseSlmStream(
        { complaint, caseId: selectedId, requestedOutput: "internal_diagnosis" },
        (t) => {
          // Gom vào ref; chỉ đẩy ra state theo nhịp (throttle) để trình duyệt không re-render mỗi token.
          streamBufRef.current += t;
          if (flushTimerRef.current == null) {
            flushTimerRef.current = window.setTimeout(() => {
              flushTimerRef.current = null;
              setStreamText(streamBufRef.current);
            }, 80);
          }
        },
        (d) => {
          stopFlush();
          setStreamText(streamBufRef.current); // flush nốt phần còn lại
          setResult(d);
          setState(d.status === "error" ? "error" : "complete");
        },
        () => {
          stopFlush();
          setState("error");
        },
      );
    } catch {
      stopFlush();
      setState("error");
    }
  }

  const running = state === "running";
  const channel = result ? channelForRc(result.rootCause) : null;

  return (
    <div className="wrap">
      <header className="app-head">
        <div className="brand">
          <span className="brand-honda">Honda</span> Entitlement Resolver
        </div>
      </header>

      <section className="card input-card">
        <div className="card-sub">Incoming complaint &amp; customer</div>
        <div className="case-row">
          {cases.map((c) => (
            <button
              key={c.id}
              className={`case-btn ${c.id === selectedId ? "active" : ""}`}
              onClick={() => selectCase(c)}
              disabled={running}
            >
              {c.label}
            </button>
          ))}
        </div>

        <textarea
          className="complaint"
          value={complaint}
          onChange={(e) => setComplaint(e.target.value)}
          rows={4}
          placeholder="Customer complaint (raw — no error code, no logs)…"
        />

        {selected && <CustomerInfo c={selected} />}

        <button className="diagnose" onClick={diagnose} disabled={running || !complaint.trim()}>
          {running ? "Diagnosing…" : "Diagnose"}
        </button>
      </section>

      {/* Stream full text ra trước (chạy thẳng xuống), khi có kết quả thì dựng bản parse + JSON. */}
      {running && !result && (
        <section className="card stream-card">
          <div className="card-sub">
            SLM generating… {streamText ? "" : "(waiting for first token)"}
            <span className="caret" />
          </div>
          {streamText && <pre className="raw stream-live">{streamText}</pre>}
        </section>
      )}

      {result && channel ? (
        <>
          <section className="card result-card">
            <header className="result-head">
              <span className="result-title">{TAB_LABEL[channel]}</span>
              <span className="result-meta">
                <span className="latency">{fmtLatency(result.latencyMs)}</span>
                {result.tokensPerSec != null && (
                  <span className="tps">{result.tokensPerSec} tok/s</span>
                )}
                {result.isMock && <span className="mock">Mock · model not loaded</span>}
              </span>
            </header>
            <div className="result-body">
              {result.status === "error" ? (
                <div className="err">{result.errorMessage || "SLM error."}</div>
              ) : channel === "customer" ? (
                <CustomerView d={result} />
              ) : (
                <InternalView d={result} />
              )}
            </div>
          </section>

          <RawBlock d={result} />
        </>
      ) : (
        !running && <div className="empty">Select a case and click Diagnose.</div>
      )}

      <footer className="foot">
        The SLM reasons only from cues in the complaint — it never fabricates telemetry; with no cue it
        abstains and asks to confirm.
      </footer>
    </div>
  );
}
