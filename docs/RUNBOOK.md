# Runbook — chạy pipeline Honda Entitlement Resolver từ đầu tới cuối

Pipeline: **generate → train → evaluate → export** rồi **dừng ở việc tạo ra model** (closed-book,
không serve). Mọi lệnh chạy qua `uv`. Spec gốc: `PoC6_SLM_BUILD_SPEC.pdf`.

## 0. Môi trường
```bash
uv sync                                   # CPU core: đủ để gen data / validate / dry-run / pytest
# Trên máy GPU (Linux+CUDA):
uv sync --extra train --extra gpu --extra export --extra viz --extra tracking
cp .env.example .env                      # (tuỳ chọn) Langfuse. Eval KHÔNG cần judge API key.
uv run pytest                             # xanh, không cần GPU/key
```

## 1. Sinh data (từ ground truth, qua oracle)
```bash
uv run python scripts/gen_sft.py  --seed 42  --out data/sft/train_sft.jsonl
uv run python scripts/gen_dpo.py  --seed 42  --out data/preference/dpo_pairs.jsonl
uv run python scripts/gen_eval.py --seed 999 --out data/gold/gold_test.jsonl
uv run python scripts/validate_data.py --data-dir data/
```
- `gen_sft` → 5 nhóm (~2.3k): complaint→resolution (3 RC) · knowledge augmentation · differential ·
  distractors · abstention. In phân bổ per-slice.
- `gen_dpo` → 6 loại cặp (~600): cue_dropped · fabricated_telemetry · overconfident · missing_fields ·
  forced_guess · overpromise. `chosen` qua oracle; `rejected` cố tình sai.
- `gen_eval` → `gold_test.jsonl` (180, seed 999) + `gold/eval_hard.jsonl` (20 viết tay).
- Thêm `--limit 30` để smoke nhanh. **Mọi mẫu có resolution đều qua oracle** trước khi ghi.

## 2. (tuỳ chọn) Holdout phân tầng
```bash
uv run python scripts/split_holdout.py --config configs/sft.yaml   # → data/holdout/{train,val}.jsonl
```
Nếu không chạy, SFT dùng `sft.val_split` in-memory.

## 3. Train SFT (model-agnostic — đổi base bằng --base)
```bash
uv run python scripts/train_sft.py --config configs/sft.yaml --base qwen     # → checkpoints/sft_qwen/best
# --base gemma | phi | granite
```
Mỗi run ghi báo cáo vào `checkpoints/sft_<base>/metrics/`:
- `run_facts.csv/.md` — base, method (QLoRA/LoRA), precision, **gradient_checkpointing**, effective
  batch, masking (assistant-only), LR/scheduler, n_train/n_val…
- `training_log.csv`, `training_summary.md`
- `loss_curve.png` · `eval_metric.png` · `lr_schedule.png` · `grad_norm.png` (cần extra `viz`).
- Checkpoint: `best/` + `last/` + `meta.json` (config, git commit, seed, metrics). `--resume <path>` để tiếp.

## 4. Eval SFT (oracle KPIs)
```bash
uv run python scripts/evaluate.py --config configs/eval.yaml --model checkpoints/sft_qwen/best \
    --base qwen --run-name eval_sft_qwen
uv run python scripts/evaluate.py --config configs/eval.yaml --model checkpoints/sft_qwen/best \
    --base qwen --hard --run-name eval_hard_qwen
```
`outputs/eval/<run>/report.{md,json}` + `per_sample.csv`: RC accuracy, confusion (3 RC + ABSTAIN),
cue-faithfulness, no-fabrication, runbook completeness/fidelity, ECE, abstention hallucination,
artifact valid@1, latency p50/p95. Bảng KPI có cột ✅/❌ so target.

Offline không GPU: thêm `--mock` (replay gold reference) để xem report pipeline.

## 5. DPO (khởi từ checkpoint SFT)
```bash
uv run python scripts/train_align.py --config configs/dpo.yaml --base qwen \
    --sft-checkpoint checkpoints/sft_qwen/best                 # → checkpoints/dpo_qwen/best
uv run python scripts/evaluate.py --config configs/eval.yaml --model checkpoints/dpo_qwen/best \
    --base qwen --run-name eval_dpo_qwen
```

## 6. Export (deliverable)
```bash
uv run python scripts/export_model.py --checkpoint checkpoints/dpo_qwen/best --formats gguf,awq
```
GGUF cần llama.cpp (`$LLAMA_CPP_DIR` trỏ tới checkout có `convert_hf_to_gguf.py` + `llama-quantize`).
AWQ cần extra `export` + GPU.

## 7. So 4 base → chọn con tốt nhất
```bash
uv run python scripts/compare_models.py --eval-root outputs/eval --out outputs/eval/leaderboard.md
```

## 8. Demo + RAG baseline
```bash
uv run python scripts/rag_baseline.py --gold data/gold/gold_test.jsonl        # foil: thấp hơn, không abstain
HONDA_ADAPTER=checkpoints/dpo_qwen/best uv run streamlit run app.py            # split-screen SLM vs RAG
```
Thiếu adapter/GPU → app tự chạy DEMO mode từ ground truth (vẫn trung thực) để chụp hình offline.

## Train cả 4 base một vòng
```bash
for M in qwen gemma phi granite; do
  uv run python scripts/train_sft.py   --config configs/sft.yaml --base $M
  uv run python scripts/train_align.py --config configs/dpo.yaml --base $M --sft-checkpoint checkpoints/sft_$M/best
  uv run python scripts/evaluate.py    --config configs/eval.yaml --model checkpoints/dpo_$M/best --base $M --run-name eval_dpo_$M
done
uv run python scripts/compare_models.py --eval-root outputs/eval
```
