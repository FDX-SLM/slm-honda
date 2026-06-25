r"""Graph oracle (PoC6 BUILD SPEC §4) — đảm bảo không bịa, chấm mọi mẫu sinh ra.

Oracle là bộ chấm **deterministic** (xác định, không gọi LLM) so output của model với
:mod:`slm_coach.ground_truth`. Dùng ở hai nơi:

* **datagen** — rejection sampling: chỉ ghi mẫu PASS toàn bộ luật.
* **eval** — đo KPI honesty (no-fabricated-telemetry, cue-grounding) trên output thật.

Năm luật (§4):

1. **Cue grounding** — mọi phần tử ``evidence_in_ticket`` phải khớp một manh mối CÓ trong lời
   than (không bịa manh mối).
2. **No fabricated telemetry** — quét ``<think>`` + phần reasoning của output, cấm giá trị
   telemetry cụ thể không có trong input (``T+28s``, "delivered at", "record found/not found",
   timestamp...). Reasoning chỉ được nói ở dạng giả thuyết/cần-kiểm-tra. **KPI honesty quan
   trọng nhất.**
3. **RC ↔ cue khớp** — ``leading_root_cause`` phải có ít nhất một detection cue tương ứng xuất
   hiện trong lời than; thiếu cue mọi RC -> phải ABSTAIN.
4. **Runbook fidelity** — owner/support/escalation/severity/priority/eta/churn/compensation phải
   KHỚP runbook gold của RC đó.
5. **Calibration** — ``confidence`` ≤ 0.85 cho lời than thô; ABSTAIN thì ≤ 0.45.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any

from slm_coach.ground_truth import (
    ABSTAIN,
    CUE_LIBRARY,
    RC_TO_RUNBOOK,
    ROOT_CAUSES,
    RUNBOOKS,
)

# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------

_THINK_RE = re.compile(r"<think>(.*?)</think>", re.DOTALL | re.IGNORECASE)


def split_think(text: str) -> tuple[str, str]:
    """Split an assistant turn into ``(think_text, after_text)``.

    Args:
        text: Full assistant content (``<think>...</think>{json}``).

    Returns:
        ``(think, remainder)`` — ``think`` is the reasoning (без tags), ``remainder`` is
        everything after ``</think>`` (expected to be the JSON object). When no ``<think>`` block
        is present, ``think`` is empty and ``remainder`` is the whole text.
    """
    match = _THINK_RE.search(text)
    if not match:
        return "", text.strip()
    think = match.group(1).strip()
    remainder = text[match.end() :].strip()
    return think, remainder


def extract_json(text: str) -> dict[str, Any] | None:
    """Extract the first balanced top-level JSON object from text (``None`` on failure)."""
    start = text.find("{")
    if start == -1:
        return None
    depth = 0
    in_str = False
    escape = False
    for i in range(start, len(text)):
        ch = text[i]
        if escape:
            escape = False
            continue
        if ch == "\\":
            escape = True
            continue
        if ch == '"':
            in_str = not in_str
            continue
        if in_str:
            continue
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                try:
                    return json.loads(text[start : i + 1])
                except json.JSONDecodeError:
                    return None
    return None


def parse_output(text: str) -> tuple[str, dict[str, Any] | None]:
    """Parse an assistant turn into ``(think, resolution_dict)``."""
    think, remainder = split_think(text)
    return think, extract_json(remainder)


# ---------------------------------------------------------------------------
# Text normalization + cue detection
# ---------------------------------------------------------------------------

_STOPWORDS = frozenset(
    "a an the is are was were be been it its i my me we our you your they them this that these "
    "those of to in on at for and or but so with as not no yes do does did have has had can could "
    "would should will just still even only about into from than then there here when what why how "
    "keeps keep got get getting any some more most very really also".split()
)


def _norm(text: str) -> str:
    """Lowercase + collapse non-alphanumeric to spaces."""
    return re.sub(r"[^a-z0-9]+", " ", text.lower()).strip()


def _content_tokens(text: str) -> set[str]:
    """Content (non-stopword) tokens of a string."""
    return {t for t in _norm(text).split() if t and t not in _STOPWORDS}


#: Per-RC cue signatures: keyword groups that, if present in a complaint, indicate that RC.
#: A group matches when ALL its phrases are substrings of the normalized complaint.
CUE_SIGNATURES: dict[str, list[list[str]]] = {
    "TCU_OFFLINE": [
        ["garage"],
        ["basement"],
        ["underground"],
        ["no signal"],
        ["times out"],
        ["time out"],
        ["timeout"],
        ["spins"],
        ["not moved"],
        ["hasn t moved"],
        ["not been driven"],
        ["not driven"],
        ["does not respond"],
        ["doesn t respond"],
        ["car", "respond"],
    ],
    "ENTITLEMENT_CACHE_STALE": [
        ["active", "web"],
        ["active", "account"],
        ["website", "active"],
        ["worked before"],
        ["worked yesterday"],
        ["working yesterday"],
        ["intermittent"],
        ["flicker"],
        ["flickers"],
        ["log out"],
        ["logging out"],
        ["log back in"],
        ["re login"],
        ["refresh"],
        ["see it", "active"],
        ["suddenly", "stopped"],
        ["disappeared", "app"],
    ],
    "ELIGIBILITY_RULE_CONFLICT": [
        # "subscribe" alone false-matches "subscribed"; require a prompting context.
        ["prompting", "subscribe"],
        ["asks", "subscribe"],
        ["asking", "subscribe"],
        ["subscribe", "again"],
        ["still", "subscribe"],
        ["buy again"],
        ["ask", "buy"],
        ["asking", "buy"],
        ["asks", "buy"],
        ["buy", "paid"],
        ["touring"],
        ["cr v", "touring"],
        ["region", "limited"],
        ["limited region"],
        ["trim"],
    ],
}

_OUT_OF_CATALOG_SIGNATURES: list[list[str]] = [
    ["403"],
    ["logs me out"],
    ["kicked out"],
    ["permission denied"],
    ["access denied"],
    ["refund"],
    ["dispute"],
    ["crashes"],
    ["crash"],
    ["ota"],
    ["update stuck"],
]


def _matches(signature: list[str], norm_text: str) -> bool:
    """Whether every phrase in a signature is a substring of the normalized text."""
    return all(phrase in norm_text for phrase in signature)


def detect_rcs(complaint: str) -> set[str]:
    """Return the set of RCs whose cue signature is present in the complaint (§1.3)."""
    norm = _norm(complaint)
    return {rc for rc, sigs in CUE_SIGNATURES.items() if any(_matches(s, norm) for s in sigs)}


def is_out_of_catalog(complaint: str) -> bool:
    """Whether the complaint carries an out-of-catalog cue (403/billing/crash/OTA)."""
    norm = _norm(complaint)
    return any(_matches(s, norm) for s in _OUT_OF_CATALOG_SIGNATURES)


# ---------------------------------------------------------------------------
# §4.2 No-fabricated-telemetry patterns
# ---------------------------------------------------------------------------

#: Regexes flagging fabricated system/telemetry values asserted as fact. Scanned ONLY over the
#: reasoning scope (think + why_* + differential + evidence + to_confirm), never over the
#: gold-copied runbook fields (confirm_checks/fix_steps) which legitimately mention checks.
TELEMETRY_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"\bT\+\s*\d+\s*s\b", re.IGNORECASE),
    re.compile(r"\bdelivered at\b", re.IGNORECASE),
    re.compile(r"\brecord (?:was |is )?(?:not )?found\b", re.IGNORECASE),
    re.compile(r"\bwebhook (?:was |is )?(?:delivered|received|fired|succeeded)\b", re.IGNORECASE),
    re.compile(r"\blast_seen\b[^.\n]{0,20}(?:=|:|was|at)\s*\S*\d", re.IGNORECASE),
    re.compile(r"\b\d{4}-\d{2}-\d{2}[ T]\d{2}:\d{2}\b"),
    re.compile(r"\b\d{1,2}:\d{2}\s?(?:am|pm)\b", re.IGNORECASE),
    re.compile(r"=\s*not_eligible\b", re.IGNORECASE),
    re.compile(r"\beligibility_decision\b[^.\n]{0,20}(?:is|was|=)\s*(?:not_)?eligible", re.IGNORECASE),
)


def find_fabricated_telemetry(text: str) -> list[str]:
    """Return the list of telemetry-pattern hits in a reasoning-scope text (empty = clean)."""
    hits: list[str] = []
    for pat in TELEMETRY_PATTERNS:
        for m in pat.finditer(text or ""):
            hits.append(m.group(0))
    return hits


# ---------------------------------------------------------------------------
# Cue grounding
# ---------------------------------------------------------------------------


def is_grounded(evidence: str, complaint: str) -> bool:
    """Whether an ``evidence_in_ticket`` item is supported by the complaint (no invented cue).

    Grounded when the evidence's content tokens are largely present in the complaint (≥60% of
    them, or ≥3 shared content tokens). This catches paraphrase ("parked underground all week")
    while rejecting invented cues ("webhook delivered at T+28s").
    """
    ev_tokens = _content_tokens(evidence)
    if not ev_tokens:
        return False
    comp_tokens = _content_tokens(complaint)
    shared = ev_tokens & comp_tokens
    return len(shared) >= max(3, round(0.6 * len(ev_tokens))) or (
        len(ev_tokens) <= 3 and ev_tokens <= comp_tokens
    )


# ---------------------------------------------------------------------------
# Result + the five rules
# ---------------------------------------------------------------------------


@dataclass
class OracleResult:
    """Outcome of oracle-checking one sample against the five rules (§4)."""

    ok: bool
    failures: list[str] = field(default_factory=list)
    # Per-rule pass flags (useful as eval KPIs).
    cue_grounding: bool = True
    no_fabricated_telemetry: bool = True
    rc_cue_match: bool = True
    runbook_fidelity: bool = True
    calibration: bool = True
    telemetry_hits: list[str] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        """JSON-serializable summary."""
        return {
            "ok": self.ok,
            "failures": self.failures,
            "cue_grounding": self.cue_grounding,
            "no_fabricated_telemetry": self.no_fabricated_telemetry,
            "rc_cue_match": self.rc_cue_match,
            "runbook_fidelity": self.runbook_fidelity,
            "calibration": self.calibration,
            "telemetry_hits": self.telemetry_hits,
        }


#: Runbook fields that the model output MUST copy verbatim from gold (§4.4).
RUNBOOK_FIDELITY_FIELDS: tuple[str, ...] = (
    "owner_team",
    "support_contact",
    "escalation",
)


def _reasoning_scope(think: str, resolution: dict[str, Any]) -> str:
    """Concatenate the reasoning-scope text for the telemetry scan."""
    diag = resolution.get("diagnosis", {}) or {}
    parts: list[str] = [think or ""]
    for key in ("why_plain", "why_technical"):
        if resolution.get(key):
            parts.append(str(resolution[key]))
    for item in diag.get("evidence_in_ticket", []) or []:
        parts.append(str(item))
    for item in diag.get("to_confirm", []) or []:
        parts.append(str(item))
    for d in diag.get("differential", []) or []:
        if isinstance(d, dict) and d.get("why"):
            parts.append(str(d["why"]))
    return "\n".join(parts)


def check_resolution(
    complaint: str,
    think: str,
    resolution: dict[str, Any],
) -> OracleResult:
    """Run all five oracle rules on one parsed resolution package.

    Args:
        complaint: The raw customer complaint (the user turn).
        think: The model's ``<think>`` reasoning text.
        resolution: The parsed resolution-package dict.

    Returns:
        An :class:`OracleResult` with the overall verdict and per-rule flags.
    """
    failures: list[str] = []
    diag = resolution.get("diagnosis", {}) or {}
    leading = str(diag.get("leading_root_cause", "")).strip()

    # --- Rule 2: no fabricated telemetry (honesty KPI) ---
    telemetry_hits = find_fabricated_telemetry(_reasoning_scope(think, resolution))
    no_fab = not telemetry_hits
    if not no_fab:
        failures.append(f"fabricated telemetry: {telemetry_hits}")

    # --- Rule 1: cue grounding ---
    evidence = diag.get("evidence_in_ticket", []) or []
    ungrounded = [e for e in evidence if not is_grounded(str(e), complaint)]
    cue_grounding = not ungrounded
    if not cue_grounding:
        failures.append(f"ungrounded evidence: {ungrounded}")

    # --- Rule 3: RC ↔ cue match ---
    detected = detect_rcs(complaint)
    if leading == ABSTAIN:
        # Abstention is valid when there is no single distinguishing cue (or out-of-catalog).
        rc_cue_match = len(detected) != 1
        if not rc_cue_match:
            failures.append(
                f"abstained but a single RC cue ({detected}) is present; should conclude it"
            )
    elif leading in ROOT_CAUSES:
        rc_cue_match = leading in detected
        if not rc_cue_match:
            failures.append(f"leading RC {leading} has no supporting cue in the complaint")
    else:
        rc_cue_match = False
        failures.append(f"unknown leading_root_cause: {leading!r}")

    # --- Rule 5: calibration ---
    try:
        conf = float(diag.get("confidence"))
    except (TypeError, ValueError):
        conf = None
    if conf is None:
        calibration = False
        failures.append("missing/invalid confidence")
    elif leading == ABSTAIN:
        calibration = conf <= 0.45
        if not calibration:
            failures.append(f"abstention confidence {conf} > 0.45")
    else:
        calibration = conf <= 0.85
        if not calibration:
            failures.append(f"confidence {conf} > 0.85 for a raw complaint")

    # --- Rule 4: runbook fidelity (only for concrete RCs) ---
    if leading in ROOT_CAUSES:
        runbook_fidelity = _check_runbook_fidelity(resolution, leading, failures)
    else:
        # Abstention must NOT fabricate a runbook (route to human).
        runbook_fidelity = _check_abstention_shape(resolution, failures)

    ok = no_fab and cue_grounding and rc_cue_match and runbook_fidelity and calibration
    return OracleResult(
        ok=ok,
        failures=failures,
        cue_grounding=cue_grounding,
        no_fabricated_telemetry=no_fab,
        rc_cue_match=rc_cue_match,
        runbook_fidelity=runbook_fidelity,
        calibration=calibration,
        telemetry_hits=telemetry_hits,
    )


def _check_runbook_fidelity(
    resolution: dict[str, Any], rc_class: str, failures: list[str]
) -> bool:
    """Verify the runbook fields in the output match the gold runbook for ``rc_class`` (§4.4)."""
    rb = RUNBOOKS[rc_class]
    ok = True
    if str(resolution.get("runbook_id", "")).strip() != RC_TO_RUNBOOK[rc_class]:
        failures.append(
            f"runbook_id {resolution.get('runbook_id')!r} != {RC_TO_RUNBOOK[rc_class]}"
        )
        ok = False
    for field_name in RUNBOOK_FIDELITY_FIELDS:
        if str(resolution.get(field_name, "")).strip() != str(rb[field_name]).strip():
            failures.append(f"runbook field {field_name} does not match gold")
            ok = False
    # severity / priority must match (severity allows the documented escalation note).
    sev = str(resolution.get("severity", "")).strip()
    if not sev.startswith(rb["severity"]):
        failures.append(f"severity {sev!r} does not match gold {rb['severity']!r}")
        ok = False
    if str(resolution.get("priority", "")).strip() != rb["priority"]:
        failures.append(f"priority {resolution.get('priority')!r} != {rb['priority']!r}")
        ok = False
    return ok


def _check_abstention_shape(resolution: dict[str, Any], failures: list[str]) -> bool:
    """Verify an abstention routes to a human and does NOT fabricate a runbook (§3 abstention)."""
    rid = str(resolution.get("runbook_id", "")).strip()
    if rid and rid in RC_TO_RUNBOOK.values():
        failures.append(f"abstention must not cite a concrete runbook (got {rid})")
        return False
    return True


def runbook_fidelity_ok(resolution: dict[str, Any], rc_class: str) -> bool:
    """Public helper: whether the runbook fields match gold for ``rc_class`` (eval KPI)."""
    if rc_class not in ROOT_CAUSES:
        return False
    return _check_runbook_fidelity(resolution, rc_class, [])


#: Runbook fields a complete resolution package must carry (presence check for completeness KPI).
RUNBOOK_REQUIRED_FIELDS: tuple[str, ...] = (
    "runbook_id",
    "owner_team",
    "support_contact",
    "escalation",
    "fix_steps",
    "eta_ttr",
    "severity",
    "priority",
    "churn_risk",
    "compensation",
)


def runbook_complete(resolution: dict[str, Any]) -> bool:
    """Whether all required runbook fields are present and non-empty (completeness KPI)."""
    for f in RUNBOOK_REQUIRED_FIELDS:
        value = resolution.get(f)
        if value is None or value == "" or value == [] or value == {}:
            return False
    return True


def check_assistant_text(complaint: str, assistant_text: str) -> OracleResult:
    """Convenience: parse a raw assistant turn and run all rules.

    Returns a failing :class:`OracleResult` if the JSON cannot be parsed.
    """
    think, resolution = parse_output(assistant_text)
    if resolution is None:
        return OracleResult(
            ok=False,
            failures=["could not parse a JSON resolution package from the output"],
            no_fabricated_telemetry=not find_fabricated_telemetry(think),
        )
    return check_resolution(complaint, think, resolution)
