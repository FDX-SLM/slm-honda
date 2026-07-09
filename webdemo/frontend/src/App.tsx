import { useEffect, useMemo, useRef, useState } from "react";
import {
  ABSTAIN_RC,
  defaultCustomer,
  layerBadge,
  rcToLayer,
  SAMPLES,
  seedQueue,
  sevForRc,
  ticketFromSLM,
  triage,
  type Customer,
  type Layer,
  type Ticket,
} from "./data";

const sleep = (ms: number) => new Promise((r) => setTimeout(r, ms));

/** Gọi SLM thật ở backend (đường L1 triage). Trả NormalizedDiagnosis. */
async function slmTriage(complaint: string): Promise<any> {
  const r = await fetch("/api/diagnose/slm", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ complaint, requestedOutput: "internal_diagnosis" }),
  });
  if (!r.ok) throw new Error("slm http " + r.status);
  return r.json();
}
const REDUCE = matchMedia("(prefers-reduced-motion:reduce)").matches;
const SP = REDUCE ? 0.25 : 1;
const wait = (ms: number) => sleep(ms * SP);

/* ---------------- reveal state cho resolver ---------------- */
interface Reveal {
  shown: number; // số step đã hiện
  running: number; // step đang chạy (-1 = none)
  panel: boolean;
  diff: number; // số dòng diff đã hiện
  test: "" | "fail" | "pass";
  reeval: boolean;
  exec: number;
  artifacts: boolean;
  resolved: boolean;
}
const INIT: Reveal = {
  shown: 0,
  running: -1,
  panel: false,
  diff: 0,
  test: "",
  reeval: false,
  exec: 0,
  artifacts: false,
  resolved: false,
};
// Trạng thái "đã hiện hết" — dùng khi xem lại ticket đã resolved.
const FULL: Reveal = {
  shown: 99,
  running: -1,
  panel: true,
  diff: 99,
  test: "pass",
  reeval: true,
  exec: 99,
  artifacts: true,
  resolved: true,
};

function toSec(s: string) {
  const m = s.match(/(?:(\d+)m)?\s*(\d+)s/);
  return (Number(m?.[1] || 0)) * 60 + Number(m?.[2] || 0);
}
function fmt(sec: number) {
  const m = Math.floor(sec / 60);
  const s = Math.round(sec % 60);
  return `${m}m ${String(s).padStart(2, "0")}s`;
}

const H = (s: string) => ({ dangerouslySetInnerHTML: { __html: s } });

/* ======================================================================= */
export default function App() {
  const [tab, setTab] = useState<"l1" | "resolve">("l1");
  const [queue, setQueue] = useState<Ticket[]>(seedQueue);
  const [activeId, setActiveId] = useState<string | null>(null);
  const [reveal, setReveal] = useState<Reveal>(INIT);
  const [viewId, setViewId] = useState<string | null>(null); // ticket đang xem lại (null = theo live)
  const runningRef = useRef(false);
  const idRef = useRef(4480);

  const queued = queue.filter((t) => t.status === "queued").length;
  const processing = queue.filter((t) => t.status === "processing").length;
  const resolved = queue.filter((t) => t.status === "resolved");

  /* ---------- runner: tự xử lý tuần tự khi ở tab resolve ---------- */
  useEffect(() => {
    if (tab !== "resolve" || runningRef.current) return;
    const next = queue.find((t) => t.status === "queued");
    if (!next) return;
    runningRef.current = true;
    (async () => {
      setActiveId(next.id);
      setReveal(INIT);
      setQueue((q) => q.map((t) => (t.id === next.id ? { ...t, status: "processing" } : t)));
      await playTicket(next);
      await wait(900);
      // reset cờ TRƯỚC khi đổi queue → thay đổi 'resolved' sẽ re-run effect và bốc ticket kế
      runningRef.current = false;
      setQueue((q) => q.map((t) => (t.id === next.id ? { ...t, status: "resolved" } : t)));
    })();
  }, [tab, queue]);

  async function playTicket(t: Ticket) {
    for (let i = 0; i < t.steps.length; i++) {
      setReveal((r) => ({ ...r, shown: i + 1, running: i }));
      await wait(t.steps[i].ev ? 1000 : 760);
      setReveal((r) => ({ ...r, running: -1 }));
    }
    await wait(160);
    setReveal((r) => ({ ...r, panel: true }));
    await wait(400);
    if (t.layer === "l3" && t.diff) {
      for (let n = 1; n <= t.diff.length; n++) {
        setReveal((r) => ({ ...r, diff: n }));
        await wait(210);
      }
      setReveal((r) => ({ ...r, test: "fail" }));
      await wait(650);
      setReveal((r) => ({ ...r, test: "pass" }));
      await wait(550);
      setReveal((r) => ({ ...r, reeval: true }));
    } else if (t.layer === "l2" && t.exec) {
      for (let n = 1; n <= t.exec.length; n++) {
        setReveal((r) => ({ ...r, exec: n }));
        await wait(340);
      }
    } else {
      await wait(900);
    }
    await wait(320);
    setReveal((r) => ({ ...r, artifacts: true }));
    await wait(500);
    setReveal((r) => ({ ...r, resolved: true }));
    await wait(200);
  }

  /* ---------- L1: SLM triage tạo ticket (nếu không abstain) ---------- */
  function addFromSLM(customer: Customer, complaint: string, d: any) {
    const id = "INC-" + idRef.current++;
    const nt = ticketFromSLM(id, customer, complaint, d);
    // Chen lên ĐẦU nhóm queued → resolver xử lý ticket của người dùng trước (thấy RCA ngay).
    setQueue((q) => {
      const i = q.findIndex((t) => t.status === "queued");
      return i < 0 ? [...q, nt] : [...q.slice(0, i), nt, ...q.slice(i)];
    });
    setViewId(null); // theo live để thấy nó chạy
  }

  /* ---------- KPI ---------- */
  const kpi = useMemo(() => {
    const n = resolved.length;
    const secs = resolved.map((t) => toSec(t.mttr));
    const avg = secs.length ? secs.reduce((a, b) => a + b, 0) / secs.length : 0;
    return { n, total: queue.length, avg: n ? fmt(avg) : "—", auto: n ? "100%" : "—" };
  }, [queue]);

  const activeTicket = queue.find((t) => t.id === activeId) || null;
  // Ticket đang hiển thị: ưu tiên cái người dùng bấm xem lại, mặc định theo live.
  const shownTicket = (viewId ? queue.find((t) => t.id === viewId) : null) || activeTicket;
  const shownReveal: Reveal = !shownTicket
    ? INIT
    : shownTicket.status === "resolved"
      ? FULL
      : shownTicket.id === activeId
        ? reveal
        : INIT; // queued & không phải ticket đang chạy → preview (chờ)

  return (
    <div className="wrap">
      <header className="app-head">
        <div className="brand">
          <span className="mark" />
          <div>
            <h1>
              <b>Honda</b> AMS · SLM Resolver
            </h1>
            <div className="sub">L1 intake → auto-resolve at L2 / L3</div>
          </div>
        </div>
        <div className="head-right">
          <span className="live">
            <span className="dot" />SLM online
          </span>
        </div>
      </header>

      <nav className="tabs">
        <button className={`tab ${tab === "l1" ? "active" : ""}`} onClick={() => setTab("l1")}>
          <span className="tn">01</span> L1 · Intake &amp; Triage
        </button>
        <button className={`tab ${tab === "resolve" ? "active" : ""}`} onClick={() => setTab("resolve")}>
          <span className="tn">02</span> L2 / L3 · Resolver
          {queued > 0 && <span className="badge-n">{queued}</span>}
        </button>
      </nav>

      {tab === "l1" ? (
        <L1View queued={queued} resolvedN={resolved.length} onSubmit={addFromSLM} goResolve={() => setTab("resolve")} />
      ) : (
        <ResolveView
          queue={queue}
          shown={shownTicket}
          reveal={shownReveal}
          activeId={activeId}
          viewId={viewId}
          onSelect={(id) => setViewId(id)}
          kpi={kpi}
        />
      )}
    </div>
  );
}

/* ======================= L1 View ======================= */
interface TriageView {
  rc: string;
  conf: number;
  sev: "2" | "3" | null;
  layer: Layer | null;
  rb: string;
  evidence: string[];
  latencyMs: number | null;
  isMock: boolean;
  abstain: boolean;
  source: "slm" | "fallback";
  name: string;
}

function fmtLatency(ms: number | null) {
  if (ms == null) return "—";
  return ms >= 1000 ? `${(ms / 1000).toFixed(1)}s` : `${ms}ms`;
}

function L1View({
  queued,
  resolvedN,
  onSubmit,
  goResolve,
}: {
  queued: number;
  resolvedN: number;
  onSubmit: (c: Customer, complaint: string, d: any) => void;
  goResolve: () => void;
}) {
  const [name, setName] = useState(SAMPLES[0].customer.name);
  const [email, setEmail] = useState(SAMPLES[0].customer.email);
  const [phone, setPhone] = useState(SAMPLES[0].customer.phone);
  const [complaint, setComplaint] = useState(SAMPLES[0].complaint);
  const [busy, setBusy] = useState(false);
  const [result, setResult] = useState<TriageView | null>(null);

  function pick(i: number) {
    const s = SAMPLES[i];
    setName(s.customer.name);
    setEmail(s.customer.email);
    setPhone(s.customer.phone);
    setComplaint(s.complaint);
    setResult(null);
  }

  function mkCustomer(rc: string): Customer {
    const base = defaultCustomer(rc);
    return { ...base, name: name || base.name, email: email || base.email, phone: phone || base.phone };
  }

  async function send() {
    if (!complaint.trim() || busy) return;
    setBusy(true);
    setResult(null);
    try {
      const d = await slmTriage(complaint); // ← gọi SLM thật
      const rc: string = d.rootCause || ABSTAIN_RC;
      const conf: number = typeof d.confidence === "number" ? d.confidence : 0;
      const abstain = d.status === "abstain" || rc === ABSTAIN_RC || !d.rootCause;
      const customer = mkCustomer(rc);
      if (!abstain) onSubmit(customer, complaint, d);
      setResult({
        rc,
        conf,
        sev: abstain ? null : sevForRc(rc),
        layer: abstain ? null : rcToLayer(rc),
        rb: d.runbook || "—",
        evidence: Array.isArray(d.evidence) ? d.evidence.slice(0, 4) : [],
        latencyMs: typeof d.latencyMs === "number" ? d.latencyMs : null,
        isMock: !!d.isMock,
        abstain,
        source: "slm",
        name: customer.name,
      });
    } catch {
      // Fallback keyword nếu backend không sẵn sàng — demo không gãy.
      const tr = triage(complaint);
      const rc = tr.rc;
      const customer = mkCustomer(rc);
      onSubmit(customer, complaint, {
        rootCause: rc,
        confidence: tr.conf,
        runbook: tr.rb,
        evidence: tr.cues,
        summary: "",
        nextActions: [],
      });
      setResult({
        rc,
        conf: tr.conf,
        sev: tr.sev,
        layer: tr.layer,
        rb: tr.rb,
        evidence: tr.cues,
        latencyMs: null,
        isMock: true,
        abstain: false,
        source: "fallback",
        name: customer.name,
      });
    } finally {
      setBusy(false);
    }
  }

  const lb = result && result.layer ? layerBadge(result.layer) : null;

  return (
    <div className="l1-grid">
      <section className="card">
        <div className="hd">Customer complaint (L1)</div>
        <div className="l1-form">
          <div className="samples">
            {SAMPLES.map((s, i) => (
              <button key={i} className="samp" onClick={() => pick(i)} disabled={busy}>
                {s.label}
              </button>
            ))}
          </div>
          <div className="frow">
            <label>
              <span>Customer</span>
              <input value={name} onChange={(e) => setName(e.target.value)} placeholder="Customer name" />
            </label>
            <label>
              <span>Email</span>
              <input value={email} onChange={(e) => setEmail(e.target.value)} placeholder="name@gmail.com" />
            </label>
            <label>
              <span>Phone</span>
              <input value={phone} onChange={(e) => setPhone(e.target.value)} placeholder="+1 …" />
            </label>
          </div>
          <label className="ta">
            <span>What the customer reported</span>
            <textarea rows={4} value={complaint} onChange={(e) => setComplaint(e.target.value)} />
          </label>
          <button className="send" onClick={send} disabled={busy || !complaint.trim()}>
            {busy ? "SLM analyzing…" : "Run SLM triage"}
          </button>
          {busy && <div className="hint">SLM runs closed-book on GPU (~20–60s).</div>}
        </div>
      </section>

      <section className="card">
        <div className="hd">
          SLM triage result
          {result && (
            <span className="tag">
              {result.source === "fallback"
                ? "fallback keyword (backend offline)"
                : result.isMock
                  ? "ground-truth (model not loaded)"
                  : `SLM · ${fmtLatency(result.latencyMs)}`}
            </span>
          )}
        </div>
        <div className="triage-body">
          {!result && !busy && (
            <div className="tri-empty">Fill in and submit — the SLM reads cues in the complaint to infer the root cause.</div>
          )}
          {busy && <div className="tri-empty run">SLM reasoning over the cues…</div>}
          {result && (
            <>
              <div className="tri-verdict">
                <span className={`rc ${result.abstain ? "abstain" : ""}`}>{result.rc}</span>
                {result.sev && <span className={`sev s${result.sev}`}>SEV{result.sev}</span>}
                <span className="conf">conf {result.conf.toFixed(2)}</span>
              </div>

              {!result.abstain && lb && (
                <div className="tri-row">
                  <span className="lbl">Route</span>
                  <span className={`layer ${lb.cls}`}>{lb.label}</span>
                  <span className="rb">{result.rb}</span>
                </div>
              )}

              {result.evidence.length > 0 && (
                <div className="tri-row cues">
                  <span className="lbl">{result.source === "slm" ? "Evidence (SLM)" : "Matched cues"}</span>
                  <div className="cue-chips">
                    {result.evidence.map((c, i) => (
                      <span className="cue" key={i}>
                        {c}
                      </span>
                    ))}
                  </div>
                </div>
              )}

              {result.abstain ? (
                <div className="tri-abstain">
                  ⓘ SLM <b>abstains</b> — not enough evidence. Kept at L1 triage, NOT routed to L2/L3 (honesty rule).
                </div>
              ) : (
                <>
                  <div className="tri-ok">
                    ✓ Triaged for <b>{result.name}</b> — sent to the L2/L3 resolver
                  </div>
                  <button className="goto" onClick={goResolve}>
                    Open Resolver ({queued} waiting) →
                  </button>
                </>
              )}
            </>
          )}
        </div>
        <div className="q-mini">
          <span>
            L2/L3 queue: <b>{queued}</b> waiting
          </span>
          <span>
            · resolved: <b>{resolvedN}</b>
          </span>
        </div>
      </section>
    </div>
  );
}

/* ======================= Resolve View ======================= */
function ResolveView({
  queue,
  shown,
  reveal,
  activeId,
  viewId,
  onSelect,
  kpi,
}: {
  queue: Ticket[];
  shown: Ticket | null;
  reveal: Reveal;
  activeId: string | null;
  viewId: string | null;
  onSelect: (id: string) => void;
  kpi: { n: number; total: number; avg: string; auto: string };
}) {
  return (
    <>
      {kpi.total > 0 && (
        <div className="kpi-strip">
          <span className="ks">
            Resolved <b>{kpi.n}/{kpi.total}</b>
          </span>
          <span className="ks">
            Auto <b>{kpi.auto}</b>
          </span>
          <span className="ks">
            MTTR (SLM) <b>{kpi.avg}</b>
          </span>
          <span className="ks">
            Root-cause <b>{kpi.n}/{kpi.total}</b>
          </span>
        </div>
      )}

      <div className="res-grid">
        <section className="card">
          <div className="hd">
            Ticket queue <span className="tag">click to review</span>
          </div>
          <div className="q-list">
            {queue.length === 0 && <div className="q-none">No tickets yet — submit from the L1 tab.</div>}
            {queue.map((t) => {
              const sel = viewId ? viewId === t.id : t.id === activeId;
              return (
                <div
                  key={t.id}
                  className={`tk ${t.status} ${sel ? "viewing" : ""}`}
                  onClick={() => onSelect(t.id)}
                  role="button"
                  tabIndex={0}
                  onKeyDown={(e) => (e.key === "Enter" || e.key === " ") && onSelect(t.id)}
                >
                  <div className="row1">
                    <span className="id">{t.id}</span>
                    <span className={`sev s${t.sev}`}>SEV{t.sev}</span>
                    <span className="st">
                      {t.status === "queued" ? "waiting" : t.status === "processing" ? "…" : "✓"}
                    </span>
                  </div>
                  <div className="ti">{t.title}</div>
                  <div className="who">{t.customer.name}</div>
                </div>
              );
            })}
          </div>
        </section>

        <section className="card center">
          {!shown ? (
            <div className="rv-empty">No tickets yet — submit from the L1 tab.</div>
          ) : (
            <ResolverBody t={shown} r={reveal} />
          )}
        </section>
      </div>
    </>
  );
}

/* ---------- Resolver body (timeline + panel wow) ---------- */
function ResolverBody({ t, r }: { t: Ticket; r: Reveal }) {
  const cu = t.customer;
  const lb = layerBadge(t.layer);
  const icons = ["◦", "◎", "◉", "✎"];
  return (
    <>
      <div className="rv-head">
        <div className="r1">
          <span className="id">{t.id}</span>
          <span className={`sev s${t.sev}`}>SEV{t.sev}</span>
          <span className="sys">{t.sys}</span>
          <span className={`layer ${lb.cls} r-layer`}>{lb.label}</span>
        </div>
        <div className="title">{t.title}</div>
        <div className="cust">
          <div className="ava">{cu.ini}</div>
          <div className="ci">
            <div className="cn">
              {cu.name}
              <span className="cloc">{cu.loc}</span>
            </div>
            <div className="cc">
              <span>✉ {cu.email}</span>
              <span>☎ {cu.phone}</span>
            </div>
          </div>
          <div className="cv">
            <div className="veh" {...H(cu.veh)} />
            <div className="vin" {...H(cu.vin)} />
          </div>
        </div>
        <div className="quote">“{t.complaint}”</div>
      </div>

      <div className="rv-body">
        {t.status === "queued" && r.shown === 0 && (
          <div className="wait-note">⏳ Waiting in the queue — the resolver will process it when its turn comes.</div>
        )}
        <div className="tl">
          {t.steps.map((s, i) => {
            if (i >= r.shown) return null;
            const cls = r.running === i ? "run" : "done";
            return (
              <div className={`step show ${cls}`} key={i}>
                <div className="node">{icons[i] || "◦"}</div>
                <div className="k">
                  {s.k}
                  <span className="kk">{s.kk}</span>
                  {s.real && <span className="slm-badge">SLM</span>}
                </div>
                <div className="v">
                  <span {...H(s.v)} />
                  {i === 1 && <span className="conf"> · confidence {t.conf.toFixed(2)}</span>}
                </div>
                {s.ev && (
                  <div className="ev show">
                    {s.ev.map((l, j) => (
                      <span className="ln" key={j} {...H(l)} />
                    ))}
                  </div>
                )}
              </div>
            );
          })}
        </div>

        {r.shown >= t.steps.length && t.slm && (t.slm.summary || t.slm.nextActions.length > 0) && (
          <div className="rca show">
            <div className="rca-h">
              RCA — generated by SLM <span className="slm-badge">SLM</span>
              <span className="rca-conf">conf {t.conf.toFixed(2)} · {t.rb}</span>
            </div>
            {t.slm.summary && <div className="rca-why">{t.slm.summary}</div>}
            {t.slm.nextActions.length > 0 && (
              <ol className="rca-steps">
                {t.slm.nextActions.slice(0, 6).map((a, i) => (
                  <li key={i}>{a}</li>
                ))}
              </ol>
            )}
            <div className="rca-meta">
              {t.slm.affectedSystem && (
                <span className="m">
                  system <b>{t.slm.affectedSystem}</b>
                </span>
              )}
              {t.slm.owner && (
                <span className="m">
                  owner <b>{t.slm.owner}</b>
                </span>
              )}
              {t.slm.escalation && (
                <span className="m">
                  escalation <b>{t.slm.escalation}</b>
                </span>
              )}
              {t.slm.similarIncident && (
                <span className="m">
                  similar <b>{t.slm.similarIncident}</b>
                </span>
              )}
            </div>
          </div>
        )}

        {r.panel && t.layer === "l3" && t.diff && (
          <div className="panel show">
            <div className="ph">
              git diff <span className="path">· src/entitlement/eligibility_matrix.py</span>
              <span className="mock-tag">mock</span>
              <span className="pill l3">L3 · code fix</span>
            </div>
            <div className="code">
              {t.diff.slice(0, r.diff).map((l, i) => (
                <span className={`ln ${l.t}`} key={i}>
                  {l.x}
                </span>
              ))}
            </div>
            {r.test && (
              <div className="testbar">
                <span>{t.testName}</span>
                <span className="arrow">·</span>
                <span className={`badge ${r.test}`}>{r.test === "fail" ? "FAIL" : "PASS"}</span>
                {r.test === "pass" && <span className="passmsg">→ 1 passed · reproduces {cu.name.split(" ")[0]}'s case</span>}
                {r.reeval && (
                  <span className="reeval">
                    <br />
                    {t.reEval}
                  </span>
                )}
              </div>
            )}
          </div>
        )}

        {r.panel && t.layer === "l2" && t.exec && (
          <div className="panel show">
            <div className="ph">
              execution trace <span className="path">· gtc-cli ({t.rb})</span>
              <span className="mock-tag">mock</span>
              <span className="pill l2">L2 · auto-exec</span>
            </div>
            <div className="code exec">
              {t.exec.slice(0, r.exec).map((l, i) => (
                <span className={`ln ${l.t}`} key={i}>
                  {l.x}
                </span>
              ))}
            </div>
          </div>
        )}

        {r.panel && t.layer === "route" && t.route && (
          <div className="panel show">
            <div className="ph">
              routing decision <span className="mock-tag">mock</span>
              <span className="pill route">route · no-code</span>
            </div>
            <div className="route-box">
              <div className="rc">root cause: {t.route.rc}</div>
              <div className="no">✕ No code change, no fabricated telemetry</div>
              <div className="to" {...H("→ " + t.route.to)} />
              <div className="chips">
                {t.route.chips.map((c, i) => (
                  <span className={`chip ${c.ok ? "ok" : ""}`} key={i}>
                    {c.ok ? "✓ " : ""}
                    {c.t}
                  </span>
                ))}
              </div>
            </div>
          </div>
        )}

        {r.artifacts && (
          <div className="artifacts show">
            {["RCA doc", "work order", "change request", "customer email"].map((a) => (
              <span className="art" key={a}>
                <span className="c">✓</span>
                {a}
              </span>
            ))}
          </div>
        )}

        {r.resolved && (
          <div className="resolved-bar show">
            <span className="st">✓ {t.layer === "route" ? "Routed" : "Resolved"}</span>
            <span className="gate">Lv4 auto-approve ✓ · policy</span>
            <span className="mttr">
              MTTR <s>~4h 30m</s> → <b>{t.mttr}</b>
            </span>
          </div>
        )}
      </div>
    </>
  );
}
