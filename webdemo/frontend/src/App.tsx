import { useEffect, useMemo, useState } from "react";
import { diagnoseBoth, getCases, getHealth } from "./api";
import type {
  ActiveMainTab,
  DemoCase,
  DiagnosisState,
  Health,
  NormalizedDiagnosis,
  SystemSignal,
} from "./types";

const TAB_LABEL: Record<ActiveMainTab, string> = {
  customer: "Hướng dẫn khách hàng",
  internal: "Chẩn đoán nội bộ",
};

function fmtLatency(ms: number | null): string {
  if (ms == null) return "—";
  return ms >= 1000 ? `${(ms / 1000).toFixed(1)}s` : `${ms}ms`;
}

function fmtConfidence(c: number | null): string {
  return c == null ? "—" : c.toFixed(2);
}

/* ------------------------------ Input card ------------------------------ */

function SignalChips({ signals }: { signals: SystemSignal[] }) {
  return (
    <div className="signals">
      {signals.map((s, i) => (
        <span className={`signal signal-${s.status}`} key={i}>
          <span className="dot" />
          <span className="signal-name">{s.name}</span>
          <span className="signal-val">{s.value}</span>
        </span>
      ))}
    </div>
  );
}

/* ---------------------------- Result columns ---------------------------- */

function EvalChecks() {
  return (
    <ul className="checks">
      <li>Actionable</li>
      <li>An toàn cho khách</li>
      <li>Không lộ dữ liệu nội bộ</li>
    </ul>
  );
}

function CustomerView({ d }: { d: NormalizedDiagnosis }) {
  return (
    <>
      <p className="summary">{d.summary || "—"}</p>
      {d.customerSteps.length > 0 && (
        <ol className="steps">
          {d.customerSteps.map((s, i) => (
            <li key={i}>{s}</li>
          ))}
        </ol>
      )}
      <EvalChecks />
    </>
  );
}

function Field({ label, value }: { label: string; value: string | null }) {
  return (
    <div className="field">
      <span className="field-label">{label}</span>
      <span className="field-value">{value || "—"}</span>
    </div>
  );
}

function InternalView({ d }: { d: NormalizedDiagnosis }) {
  if (d.status === "abstain") {
    return (
      <>
        <div className="verdict">
          <span className="rc abstain">NEEDS_MORE_EVIDENCE</span>
          <span className="conf">confidence {fmtConfidence(d.confidence)}</span>
        </div>
        <p className="summary">{d.summary || "Chưa đủ bằng chứng để kết luận root cause."}</p>
        {d.missingEvidence.length > 0 && (
          <>
            <div className="block-label">Cần kiểm tra thêm</div>
            <ul className="bullets todo">
              {d.missingEvidence.map((m, i) => (
                <li key={i}>{m}</li>
              ))}
            </ul>
          </>
        )}
        <p className="note">Giữ ticket ở L2 triage — không route xuống owner khi chưa đủ bằng chứng.</p>
      </>
    );
  }
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
          <div className="block-label">Bằng chứng</div>
          <ul className="bullets">
            {d.evidence.map((e, i) => (
              <li key={i}>{e}</li>
            ))}
          </ul>
        </>
      )}
      <div className="fields">
        <Field label="Hệ thống ảnh hưởng" value={d.affectedSystem} />
        <Field label="Nhóm phụ trách" value={d.owner} />
        <Field label="Escalation" value={d.escalation} />
        <Field label="Similar incident" value={d.similarIncident} />
      </div>
      {d.nextActions.length > 0 && (
        <>
          <div className="block-label">Bước xử lý tiếp theo</div>
          <ol className="steps">
            {d.nextActions.map((a, i) => (
              <li key={i}>{a}</li>
            ))}
          </ol>
        </>
      )}
    </>
  );
}

function ModelColumn({
  title,
  badge,
  d,
  channel,
}: {
  title: string;
  badge: string;
  d: NormalizedDiagnosis;
  channel: ActiveMainTab;
}) {
  return (
    <section className="col">
      <header className="col-head">
        <div>
          <div className="col-title">{title}</div>
          <div className="col-badge">{badge}</div>
        </div>
        <div className="col-meta">
          <span className="latency">{fmtLatency(d.latencyMs)}</span>
          {d.isMock && <span className="mock">Mock · chưa cấu hình API key</span>}
        </div>
      </header>
      <div className="col-body">
        {d.status === "error" ? (
          <div className="err">{d.errorMessage || "Model lỗi."}</div>
        ) : channel === "customer" ? (
          <CustomerView d={d} />
        ) : (
          <InternalView d={d} />
        )}
      </div>
    </section>
  );
}

/* --------------------------------- App --------------------------------- */

export default function App() {
  const [cases, setCases] = useState<DemoCase[]>([]);
  const [selectedId, setSelectedId] = useState<string>("tcu-offline");
  const [complaint, setComplaint] = useState<string>("");
  const [activeTab, setActiveTab] = useState<ActiveMainTab>("customer");
  const [state, setState] = useState<DiagnosisState>("idle");
  const [results, setResults] = useState<{ slm: NormalizedDiagnosis; gemini: NormalizedDiagnosis } | null>(
    null,
  );
  const [health, setHealth] = useState<Health | null>(null);

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
    getHealth().then(setHealth).catch(() => setHealth(null));
  }, []);

  const selected = useMemo(() => cases.find((c) => c.id === selectedId), [cases, selectedId]);
  const routedTab: ActiveMainTab = (selected?.outputChannel ?? "internal") as ActiveMainTab;

  function selectCase(c: DemoCase) {
    setSelectedId(c.id);
    setComplaint(c.complaint);
    setResults(null); // xoá kết quả cũ khi đổi ca
    setState("idle");
  }

  async function diagnose() {
    if (!complaint.trim() || state === "running") return;
    setState("running");
    setResults(null);
    const channel = routedTab;
    try {
      const res = await diagnoseBoth(
        {
          complaint,
          caseId: selectedId,
          requestedOutput: channel === "customer" ? "customer_guidance" : "internal_diagnosis",
        },
        channel,
      );
      setResults(res);
      setActiveTab(channel); // tự chuyển sang tab đúng
      const errs = [res.slm, res.gemini].filter((r) => r.status === "error").length;
      setState(errs === 0 ? "complete" : errs === 1 ? "partial" : "error");
    } catch {
      setState("error");
    }
  }

  const running = state === "running";

  return (
    <div className="wrap">
      <header className="app-head">
        <div className="brand">
          <span className="brand-honda">Honda</span> Entitlement Resolver
        </div>
        <div className="sub">
          Closed-book resolver · so sánh <b>SLM chuyên biệt</b> vs <b>LLM tổng quát (Gemini)</b>
        </div>
        {health && (
          <div className={`health ${health.modelLoaded ? "ok" : "warn"}`}>
            {health.modelLoaded ? `SLM: ${health.adapter.split("/").pop()}` : "SLM: ground-truth (chưa nạp model)"}
          </div>
        )}
      </header>

      <section className="card input-card">
        <div className="card-sub">Complaint đến &amp; tín hiệu hệ thống</div>
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
          placeholder="Lời than của khách (thô — không mã lỗi, không log)…"
        />

        {selected && <SignalChips signals={selected.systemSignals} />}

        <button className="diagnose" onClick={diagnose} disabled={running || !complaint.trim()}>
          {running ? "Đang chẩn đoán…" : "Diagnose"}
        </button>
        {running && (
          <div className="hint">SLM chạy trên GPU (có thể ~30–60s) và Gemini chạy song song.</div>
        )}
      </section>

      <div className="tabs">
        {(["customer", "internal"] as ActiveMainTab[]).map((t) => (
          <button
            key={t}
            className={`tab ${activeTab === t ? "active" : ""}`}
            onClick={() => setActiveTab(t)}
          >
            {TAB_LABEL[t]}
            {routedTab === t && results && <span className="tab-dot" />}
          </button>
        ))}
      </div>

      {results ? (
        activeTab === routedTab ? (
          <>
            {state === "partial" && (
              <div className="banner warn">Một model gặp lỗi — kết quả model còn lại vẫn hiển thị.</div>
            )}
            <div className="cols">
              <ModelColumn
                title="Domain SLM"
                badge="Local · chuyên biệt"
                d={results.slm}
                channel={activeTab}
              />
              <ModelColumn
                title="LLM · Gemini"
                badge="External API · tổng quát"
                d={results.gemini}
                channel={activeTab}
              />
            </div>
          </>
        ) : (
          <div className="banner">
            Kết quả ca này được định tuyến sang tab <b>{TAB_LABEL[routedTab]}</b>.
            <button className="link" onClick={() => setActiveTab(routedTab)}>
              Mở tab đó
            </button>
          </div>
        )
      ) : (
        <div className="empty">
          {running ? "Đang chạy 2 model trên cùng một complaint…" : "Chọn một ca và bấm Diagnose."}
        </div>
      )}

      <footer className="foot">
        SLM chỉ suy luận từ cue trong lời than — không bịa telemetry; thiếu cue thì abstain và
        chuyển người. Gemini là baseline LLM tổng quát nhận cùng dữ kiện, không phải RAG.
      </footer>
    </div>
  );
}
