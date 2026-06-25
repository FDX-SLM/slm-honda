r"""RAG baseline (spec §8 split-screen) — the closed-book SLM's foil.

Một RAG ngây thơ: index incident cũ, retrieve ticket "nghe giống" theo **surface text** rồi chép
RC/runbook của nó. Vì nó match theo bề mặt chứ không đọc **cue phân biệt**, nó sai RC trên các ca
cue-flip (vd lời than TCU có "subscription active" lại trúng ticket cache) — đúng money-shot của
demo: SLM đọc cue → đúng RC; RAG chép ticket giống → sai RC.

Retriever thuần Python (TF-IDF cosine, không thêm dependency) nên chạy offline + trong test.
"""

from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass
from typing import Any

from slm_coach.ground_truth import INCIDENTS, RC_TO_RUNBOOK, runbook_for

_TOKEN_RE = re.compile(r"[a-z0-9]+")


def _tokens(text: str) -> list[str]:
    return _TOKEN_RE.findall(text.lower())


@dataclass
class Retrieved:
    """A retrieval hit: the matched incident plus its similarity score."""

    incident: dict[str, Any]
    score: float


class RagBaseline:
    """Naive TF-IDF retriever over the incident history (surface match, cue-blind)."""

    def __init__(self, corpus: list[dict[str, Any]] | None = None) -> None:
        """Build the TF-IDF index from incidents (``customer_complaint`` as the document text)."""
        self.docs = corpus if corpus is not None else INCIDENTS
        self._doc_tokens = [_tokens(d["customer_complaint"]) for d in self.docs]
        df: Counter[str] = Counter()
        for toks in self._doc_tokens:
            for term in set(toks):
                df[term] += 1
        n = max(1, len(self.docs))
        self._idf = {t: math.log((1 + n) / (1 + c)) + 1.0 for t, c in df.items()}
        self._doc_vecs = [self._vec(toks) for toks in self._doc_tokens]

    def _vec(self, toks: list[str]) -> dict[str, float]:
        tf = Counter(toks)
        return {t: tf[t] * self._idf.get(t, 0.0) for t in tf}

    @staticmethod
    def _cosine(a: dict[str, float], b: dict[str, float]) -> float:
        if not a or not b:
            return 0.0
        common = set(a) & set(b)
        dot = sum(a[t] * b[t] for t in common)
        na = math.sqrt(sum(v * v for v in a.values()))
        nb = math.sqrt(sum(v * v for v in b.values()))
        return dot / (na * nb) if na and nb else 0.0

    def retrieve(self, complaint: str) -> Retrieved:
        """Return the single most surface-similar past incident."""
        q = self._vec(_tokens(complaint))
        best_i, best_s = 0, -1.0
        for i, dv in enumerate(self._doc_vecs):
            s = self._cosine(q, dv)
            if s > best_s:
                best_i, best_s = i, s
        return Retrieved(incident=self.docs[best_i], score=best_s)

    def predict(self, complaint: str) -> dict[str, Any]:
        """Predict an RC by copying the nearest incident's resolution (cue-blind).

        Returns a small resolution-like dict (leading_root_cause + runbook fields + the cited
        incident) — what a copy-the-nearest-ticket RAG system would output.
        """
        hit = self.retrieve(complaint)
        rc = hit.incident["root_cause_class"]
        rb = runbook_for(rc)
        return {
            "leading_root_cause": rc,
            "confidence": round(min(0.95, max(0.5, hit.score)), 2),
            "runbook_id": RC_TO_RUNBOOK[rc],
            "owner_team": rb["owner_team"],
            "fix_steps": rb["fix_steps"],
            "retrieved_incident": hit.incident["id"],
            "retrieved_complaint": hit.incident["customer_complaint"],
            "similarity": round(hit.score, 3),
            "method": "rag-copy-nearest-ticket",
        }
