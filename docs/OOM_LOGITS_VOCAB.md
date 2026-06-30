# Vì sao train Qwen bị CUDA OOM (logits ∝ vocab) — lý thuyết & cách khắc phục

> Ghi chú kỹ thuật cho PoC6. Bối cảnh: train SFT chung 1 config (`configs/sft.yaml`) cho 4 base.
> **granite** và **phi** train trơn tru, nhưng **qwen** OOM ở ~step 31/110. File này giải thích
> hiện tượng, lý thuyết đằng sau, và vì sao qwen đã chiếm ~80GB trước khi tràn.

---

## 1. Hiện tượng

```
OutOfMemoryError: CUDA out of memory. Tried to allocate 15.15 GiB.
GPU 0 total 94.97 GiB, of which 15.12 GiB is free.
Process: 79.84 GiB in use (67.76 GiB allocated by PyTorch, 11.40 GiB reserved-but-unallocated).
   ... shift_logits = outputs.logits[..., :-1, :].contiguous()   # ← dòng gây tràn
```

Lỗi rơi đúng vào bước **tính loss ở LM-head** (`outputs.logits`), không phải attention hay weight.
Cùng `batch=8, seq=2048, bf16 LoRA`, chỉ khác base → khác **vocab size**.

| base | vocab | hidden | kiến trúc | params |
|---|---|---|---|---|
| granite-4.1 | 100,352 | 4096 | **hybrid Mamba/SSM + attention** | 8B |
| phi-4 | 100,352 | 5120 | dense transformer | **14B** |
| **qwen3.5** | **248,077** | ~4096 | dense transformer | 9B |

Điểm mấu chốt: **vocab qwen lớn ~2.5×** granite/phi.

---

## 2. Lý thuyết: vocab & logits là gì, làm gì trong base model

### 2.1 Vocab = bộ "mảnh chữ" (sub-word token), cố định theo model
Vocab **được chốt lúc pretrain** (tokenizer + kích thước ma trận embedding/LM-head bị đóng băng);
muốn đổi phải train lại. Token **không phải từ nguyên** mà là mảnh BPE. Ví dụ thật (qwen tokenize):

```
"Remote start times out in the underground garage. 403 permission denied. 地下车库没信号"
→ 21 token:
['Remote',' start',' times',' out',' in',' the',' underground',' garage','.',' ',
 '4','0','3',' permission',' denied','.',' ','地下','车库','没','信号']
```
Gồm: từ/cụm-có-dấu-cách thông dụng, chữ số tách lẻ, dấu câu, mảnh đa ngôn ngữ (tiếng Trung),
và **token đặc biệt/điều khiển** (`<|im_end|>`, `<|endoftext|>`, `<|audio_start|>`, `<|vision_start|>`…).

**Vì sao qwen nhiều (248K):** đa ngôn ngữ (Trung + nhiều thứ tiếng), code/emoji/byte-fallback,
token placeholder đa phương thức (audio/image/video). Đánh đổi: vocab to → ít token hơn cho cùng
đoạn văn (nén tốt, context dài) nhưng ma trận embedding/LM-head và **logits** to hơn.

### 2.2 Vai trò trong base model — vocab là **cả đầu vào lẫn đầu ra**
```
text → [tokenizer] → ids → [Embedding: vocab×hidden] → vectors → …Transformer…
     → hidden cuối → [LM-head: hidden×vocab] → logits → softmax → token kế tiếp
```
- **Đầu vào (Embedding table `vocab × hidden`):** mỗi token id tra 1 hàng → vector. Cách chữ thành số.
- **Đầu ra (LM-head `hidden × vocab`):** nhiệm vụ DUY NHẤT của base LM = *đoán token kế tiếp* —
  một bài **phân loại trên toàn vocab** tại mỗi vị trí. **logits** = điểm thô cho từng token vocab
  (trước softmax). Muốn tính cross-entropy phải có điểm của **mọi từ vocab tại mọi vị trí**.

Cả hai ma trận **tỉ lệ thẳng với vocab**:

| base | embedding table `vocab×hidden×2B` |
|---|---|
| granite (100K×4096) | ~0.82 GB |
| phi (100K×5120) | ~1.03 GB |
| qwen (248K×4096) | **~2 GB** |

---

## 3. Vì sao logits gây phình bộ nhớ (∝ vocab)

Tensor logits lúc tính loss:
```
logits.shape = [batch, seq_len, vocab]
bộ nhớ 1 bản = batch × seq_len × vocab × bytes
```
`batch`, `seq` cố định → **bộ nhớ ∝ vocab**. Cross-entropy của HF **upcast logits lên fp32** (4 byte)
cho ổn định số học, nên dùng 4 byte/phần tử kể cả khi model chạy bf16.

**Cụ thể batch 8 × seq 2048 = 16,384 token**, mỗi token gánh 1 hàng dài `vocab`:

| base | bytes/token (vocab×4) | × 16,384 token = **1 bản logits** |
|---|---|---|
| granite/phi (100,352) | ~0.40 MB | **~6.6 GB** |
| **qwen (248,077)** | ~0.99 MB | **~16.3 GB** |

→ "*Tried to allocate 15.15 GiB*" chính là **một bản logits của qwen**.

Mỗi sample trong batch cộng nguyên khối `seq×vocab` ≈ **2 GB/sample** (qwen) — đó là lý do **giảm
micro-batch cũng cứu được** (batch 8→4 ⇒ logits 16→8 GB).

---

## 4. Vì sao qwen đã chiếm ~80GB *trước khi* OOM

Với LoRA, **gradient và optimizer states rất nhỏ** (chỉ cho adapter ~vài trăm MB) — nên ~80GB
**không phải** do optimizer. Phân rã hợp lý (qwen 9B, bf16, batch 8, seq 2048):

| thành phần | ước lượng | ghi chú |
|---|---|---|
| Weights (gồm embedding 248K×4096 ~1B params) | **~18 GB** | bf16; embedding/LM-head chiếm phần lớn |
| **logits forward** `[8,2048,248077]` bf16 | **~16 GB** | model đã sinh ra tensor này ở forward |
| **upcast logits → fp32** cho cross-entropy | **~32 GB** | bản copy fp32 (2× của bf16) |
| activations còn giữ cho backward (dù có gradient_checkpointing) | phần còn lại | dense → nhiều hơn hybrid |
| reserved-but-unallocated (phân mảnh) | ~11 GB | PyTorch giữ sẵn |

→ Chỉ riêng **logits (bf16 16GB + fp32 32GB ≈ 48GB) + weights 18GB ≈ 66GB** đã khớp với "67.76 GiB
allocated". Sau đó `shift_logits = outputs.logits[..., :-1, :].contiguous()` **xin thêm 1 bản logits
nữa (~15GB)** trong khi chỉ còn ~15GB trống → **tràn**. Tức phần lớn 80GB là **các tensor cỡ-logits**,
đúng thủ phạm.

---

## 5. Vì sao granite & phi KHÔNG OOM mà qwen có

1. **Vocab quyết định, không phải số params.** logits ∝ vocab. qwen 248K → logits ~2.5× granite/phi.
   - phi-4 **to hơn (14B)** nhưng **vocab chỉ 100K** → cú phình logits nhỏ → vừa chỗ.
   - qwen **nhẹ weight hơn (9B)** nhưng **vocab 248K** → logits phình → tràn.
2. **granite-4 là hybrid Mamba/SSM:** activation theo seq nhỏ hơn nhiều so với full self-attention →
   thêm headroom. qwen/phi là dense.
3. Tổng hợp: qwen = **dense + vocab khổng lồ** → vừa activation cao vừa logits cao.

---

## 6. Cách khắc phục (theo thứ tự ưu tiên)

### ✅ 6.1 Liger fused linear-cross-entropy (chính — đã áp dụng)
`configs/sft.yaml → sft.use_liger_kernel: true`. Liger tính loss **không bao giờ vật chất hóa nguyên
tensor `[batch,seq,vocab]`**: nó nhân hidden với LM-head weight **theo từng chunk** và cộng dồn loss
tại chỗ → triệt tiêu cú phình 16–48GB. **Giữ nguyên batch 8 / seq 2048**, không đổi chất lượng. qwen
(vocab lớn) là ca hưởng lợi nhiều nhất. (Yêu cầu Liger hỗ trợ arch; đã cài sẵn trong env.)

### 6.2 Giảm phân mảnh
```bash
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True uv run python scripts/train_sft.py ... --base qwen
```

### 6.3 Giảm micro-batch (giữ effective batch = 16)
```yaml
sft:
  batch_size: 4     # hoặc 2
  grad_accum: 4     # hoặc 8
```
logits ∝ batch nên 8→4 giảm một nửa. Kết quả tương đương vì effective batch không đổi (chỉ chậm hơn).

### 6.4 QLoRA 4-bit (phương án cuối)
`quant.load_in_4bit: true` → weight 9B 18GB→~5GB, dư RAM cho logits — nhưng đổi sang regime 4-bit.

### 6.5 Giảm `max_seq_len`
logits ∝ seq. Hạ 2048→1536 giảm 25% — nhưng dữ liệu của ta cần ~2048 (p95≈1752) nên dễ truncate đuôi
JSON/artifact → **không khuyến nghị** trừ khi bí.

---

## 7. Bài học

- **Cảnh giác model đa ngôn ngữ vocab lớn** (Qwen, Gemma, …): chi phí train/infer ∝ vocab ở LM-head,
  không chỉ ∝ số params. Một model "nhỏ params" vẫn có thể OOM nếu vocab khổng lồ.
- **Liger / fused CE là tiêu chuẩn** khi vocab lớn — bật mặc định cho mọi base, vô hại với base vocab nhỏ
  (chỉ nhanh hơn).
- Khi đọc OOM, nhìn **dòng traceback**: nếu ở `outputs.logits …contiguous()` hoặc cross-entropy →
  gần như chắc chắn là vấn đề **vocab × batch × seq** ở LM-head, không phải attention/weight.
