"""Tests for config models and the base.yaml + override loader."""

from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from slm_coach.config import (
    EvalFileConfig,
    load_align_config,
    load_eval_config,
    load_sft_config,
)

CONFIGS = Path(__file__).resolve().parents[1] / "configs"


def test_sft_config_merges_base():
    cfg = load_sft_config(CONFIGS / "sft.yaml")
    assert cfg.model_name == "Qwen/Qwen3.5-9B"  # default base (override with --base)
    assert cfg.seed == 42
    assert cfg.data.dir == "data"  # from base.yaml
    assert cfg.data.lang == "en"  # Honda data is English
    assert cfg.tracking.langfuse is True
    assert cfg.run_name == "sft"
    assert cfg.sft.epochs == 2  # 2 epochs; early-stopping + load-best cut overfit short
    assert cfg.lora.r > 0
    assert cfg.quant.load_in_4bit is False  # full bf16 LoRA (fits on 96GB; no quant quality loss)
    assert cfg.data.holdout_dir == "data/holdout"  # frozen stratified split shared across 4 bases
    assert cfg.data.val_min_total == 60


def test_dpo_config_parses():
    dpo = load_align_config(CONFIGS / "dpo.yaml")
    assert dpo.align.method == "dpo"
    assert dpo.sft_checkpoint == "checkpoints/sft/best"
    assert dpo.align.lr == 5.0e-6


def test_smoke_config_parses():
    cfg = load_sft_config(CONFIGS / "sft_lora_smoke.yaml")
    assert cfg.sft.max_steps == 60  # smoke cap
    assert cfg.quant.load_in_4bit is False


def test_align_coach_dpo_config():
    dpo = load_align_config(CONFIGS / "align_coach_dpo.yaml")
    assert dpo.align.method == "dpo"
    assert dpo.sft_checkpoint == "checkpoints/sft_coach_9b/best"  # DPO needs an SFT start
    assert dpo.align.loss_type == "sigmoid"  # pref_loss
    assert dpo.align.rpo_alpha is None  # pref_ftx: 0 -> off
    assert dpo.align.lr == 5.0e-6  # low LR for DPO
    assert dpo.train.optim == "adamw_torch"  # optimizer


def test_eval_config_values():
    cfg = load_eval_config(CONFIGS / "eval.yaml")
    assert cfg.gold == "data/gold/gold_test.jsonl"
    assert cfg.seed == 999  # held-out eval seed
    assert cfg.generation.max_new_tokens == 1700  # full think+JSON+ladder package needs the room
    assert cfg.system_prompt is None  # falls back to ground_truth.SYSTEM_PROMPT
    assert "claude" not in cfg.judges and "deepseek" not in cfg.judges


def test_eval_config_rejects_teacher_judges():
    with pytest.raises(ValidationError):
        EvalFileConfig(model_name="x", judges=["gpt", "claude"])
    with pytest.raises(ValidationError):
        EvalFileConfig(model_name="x", judges=["gpt", "deepseek"])


def test_env_expansion_unset_var_becomes_none(monkeypatch):
    from slm_coach.config import _expand_env

    monkeypatch.delenv("SLM_TEST_VAR", raising=False)
    assert _expand_env({"k": "${SLM_TEST_VAR}"}) == {"k": None}


def test_env_expansion_resolves_set_var(monkeypatch):
    from slm_coach.config import _expand_env

    monkeypatch.setenv("SLM_TEST_VAR", "file:./somewhere")
    assert _expand_env({"k": "${SLM_TEST_VAR}"}) == {"k": "file:./somewhere"}
