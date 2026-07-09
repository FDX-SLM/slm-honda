# Honda PoC⑥ — Kịch bản demo & việc cần sửa trong PDF lời dẫn

> Mục đích file này: (1) ghi **cụ thể L2/L3 “sửa” gì trong từng root cause**, (2) mô tả **kịch bản
> webdemo hiện tại** (thứ sẽ chạy trên màn hình khi demo), (3) liệt kê **chính xác cần sửa gì trong
> PDF lời dẫn v4** để khớp. Dùng phần 3 để nhờ Claude web sửa file kịch bản.

---

## 0. Vì sao PDF v4 phải sửa

PDF lời dẫn v4 hiện tại có **2 giả định không còn khớp**:

1. **Chỉ dừng ở L1** — cả 4 ticket đều là *triage / route / abstain* (phân loại + đưa ticket đã điều
   tra sẵn cho người). Chủ tịch đã bác: *“demo ở L1 là cái dễ, không thể hiện độ sâu domain. Phải cho
   SLM xử lý tiếp ở L2 (tự thực thi) hoặc L3 (sửa code)”* — và *“giữ triage + BỔ SUNG business case
   xử lý ở hướng 2”*.
2. **So sánh SLM vs LLM song song (trái/phải)** — webdemo hiện tại **đã bỏ cột Gemini**, đổi thành
   **2 tab: `L1 · Tiếp nhận` → `L2/L3 · Resolver`**. Nên mọi câu “bên trái/bên phải, chạy song song”
   trong PDF không còn cảnh tương ứng trên màn hình.

→ Việc cần làm: **thêm Màn 2 (L2/L3 resolution)** vào kịch bản, và **quyết định về phần so sánh LLM**
(giữ thì phải thêm lại cột Gemini vào webdemo — xem §3, mục “Quyết định Gemini”).

---

## 1. Bản đồ layer — CỤ THỂ L2/L3 “sửa” gì (5 root cause + abstain)

Nguyên tắc phân biệt:
- **L2 = code ĐÚNG, chỉ DỮ LIỆU/STATE lệch** → chạy lệnh vận hành nắn lại state (không sửa code).
- **L3 = LOGIC/CONFIG code SAI** → vá source code + unit test.
- **ROUTE / ABSTAIN = không thuộc L2/L3** → không tự xử lý (điều hướng, hoặc im lặng chờ dữ kiện).

| # | Root cause | Cue phân biệt | Tầng | SLM làm gì **cụ thể** | Đụng code? | Runbook |
|---|---|---|---|---|---|---|
| 1 | `TCU_OFFLINE` | “đỗ trong hầm”, timeout, no-signal, sub vẫn active | **ROUTE** | Nhận ra lỗi môi trường (mất cellular). **Không tạo ticket, không sửa gì** → trả lời khách tự xử (lái ra chỗ thoáng) / route Carrier. | ❌ | RB-TCU-04 |
| 2 | `ENTITLEMENT_CACHE_STALE` | web OK / app off, **chập chờn**, logout-vào-lại thì đỡ | **L2 · auto-exec** | `invalidate` cache stale (vd node cache-b2) → `resync` từ RTS → verify app==web → rollback snapshot. **Nắn lại state cache cho khớp nguồn.** | ❌ | RB-CACHE-02 |
| 3 | `PAYMENT_WEBHOOK_LOST` | “vừa trả sáng nay chưa bật”, charge captured nhưng chưa active | **L2 · auto-exec** | `event status` (thấy charge CAPTURED, entitlement MISSING, event DROPPED) → `activation replay` (idempotent) → tạo entitlement, propagate GTC → verify. **Phát lại sự kiện bị mất.** | ❌ | RB-PAY-01 |
| 4 | `ELIGIBILITY_RULE_CONFLICT` | đã trả tiền, app đòi Subscribe, **dai dẳng** (không chập chờn), combo region/trim | **L3 · code fix** | Sinh **patch** sửa dòng rule trong `eligibility_matrix` (region-trim bị loại nhầm khỏi gói) + **unit test** reproduce → re-eval các VIN. **Sửa LOGIC code.** | ✅ | RB-ELIG-05 |
| 5 | `TOKEN_SCOPE` | báo **403 / permission denied**, out-in vẫn chặn, plan active | **L3 · code fix** | Sinh **patch** bổ sung scope thiếu trong `scope_mapping` (entitlement→scope map sai) + reissue token + **unit test**. **Sửa CONFIG/mapping code.** | ✅ | RB-IAM-03 |
| 6 | `INSUFFICIENT_EVIDENCE` | quá ít dữ kiện, nhiều RC cùng khớp | **ABSTAIN** | Không đoán. Nêu 2 điều cần xác nhận (24h app có liên lạc xe? mua bao giờ?). **Chưa đủ thì chưa tạo ticket, chưa gán ai.** | ❌ | — |

**Cặp dễ nhầm (điểm chốt của demo):** #2 (cache) và #4 (eligibility) đều là *“đã trả tiền, app không
thấy quyền, xe online”* — khác **đúng một chữ**: **chập chờn → L2 cache** vs **dai dẳng → L3
eligibility**. Hai đội khác nhau, hai runbook khác nhau, một cái sửa state một cái sửa code.

---

## 2. Kịch bản webdemo hiện tại (thứ chạy trên màn hình)

**Tab 01 · L1 · Tiếp nhận & Triage** (SLM THẬT):
- Form có **tên khách / gmail / sđt** + nội dung phản ánh (5 mẫu tình huống, nhãn không lộ root cause).
- Bấm **“Gửi ticket → SLM triage”** → gọi model thật `/api/diagnose/slm` (~25s trên GPU) → hiện
  **root cause + confidence + evidence (trích lời than) + tuyến L2/L3/route**. Nếu SLM `abstain` →
  “giữ ở L1, không route”.
- Ticket được đẩy vào hàng đợi (chen lên đầu để xử lý trước).

**Tab 02 · L2/L3 · Resolver** (RCA thật + renovation mock):
- Ticket tự chạy: Ingest → **Investigate (evidence THẬT do SLM sinh, badge `SLM`)** → Root cause →
  **card “RCA — do SLM sinh”** (why + fix steps + owner/escalation/similar).
- Panel renovation theo tầng (hiện gắn nhãn `mô phỏng`):
  - **L3** → git diff + unit test đỏ→xanh.
  - **L2** → execution trace (`gtc-cli` / `pay-cli`) + verify + rollback.
  - **ROUTE** → quyết định điều hướng, không sửa code.
- Kết: auto-approve Lv4 → `resolved` → KPI (MTTR người ~4h30m → SLM phút). Bấm lại ticket để xem lại.

**Ranh giới thật/mock (đang hiện ngay trên UI):**
- **THẬT (badge `SLM`)**: root cause, confidence, evidence, RCA (why + fix steps + owner…).
- **MOCK (nhãn `mô phỏng`)**: git diff, exec trace, test pass, các con số (1,412 VIN…), MTTR/KPI.

---

## 3. CẦN SỬA GÌ TRONG PDF LỜI DẪN v4 → v5

> Ký hiệu: **GIỮ** = giữ nguyên · **THAY** = viết lại · **THÊM** = viết mới · **BỎ** = xoá.

### 3.1. Quyết định Gemini (làm TRƯỚC, vì nó chi phối cả kịch bản)
PDF v4 dựng WOW3 (đỉnh) trên cảnh **SLM vs LLM song song** (“LLM trả lời y hệt cho cặp song sinh”).
Webdemo hiện **không có cột Gemini**. Chọn 1:
- **(A) Giữ so sánh LLM** — mạnh nhất cho WOW3. ⇒ cần **thêm lại cột Gemini vào webdemo** (báo tôi làm).
- **(B) Bỏ so sánh LLM** — WOW3 đổi thành “**SLM phân biệt được cặp song sinh nhờ đọc lịch sử
  incident của Honda**” (vẫn mạnh, mất vế “LLM to mấy cũng thua”). ⇒ chỉ sửa lời, không đụng webdemo.

Khuyến nghị nếu gấp cho mai: **(B)** để khớp webdemo đang có; nếu còn thời gian để tôi thêm Gemini thì **(A)**.

### 3.2. MỞ
- **THAY** đoạn “Bên trái SLM… Bên phải LLM… chạy song song… các anh chấm.”
  → giới thiệu **2 màn**: *“Màn 1: model đọc lời than, phân loại đúng đội (L1). Màn 2 — chỗ khác biệt
  thật — nó **tự xử lý**: cái sửa được bằng vận hành thì nó tự chạy (L2), cái phải sửa code thì nó ra
  luôn bản vá (L3).”*
- (Nếu chọn (B)) **BỎ** mọi câu “bên phải / mô hình lớn / chạy song song”. (Nếu (A) thì GIỮ.)
- **GIỮ** toàn bộ phần mở về “6h chiều thứ Sáu / 1.636 inquiry / tri thức nằm trong đầu vài kỹ sư” — rất hay.

### 3.3. TICKET 1 — TCU (ROUTE)
- **GIỮ** gần như nguyên (đây đã là WOW1 “biết khi nào KHÔNG tạo ticket” — đúng tinh thần).
- **THÊM** 1 câu chốt tầng: *“Đây là quyết định điều hướng — không phải L2, không phải L3. Model biết
  ca này **không có gì để sửa**.”* (để phân biệt rõ với 2 ticket sau nó CÓ xử lý).

### 3.4. TICKET 2 — CACHE STALE (nâng từ L1 → **L2 tự thực thi**)  ⟵ SỬA QUAN TRỌNG
- **GIỮ** phần chẩn đoán + WOW2 “ticket đến đã được điều tra sẵn”.
- **THÊM Màn 2 cho ticket này** (đây là cái chủ tịch muốn): *“Nhưng nó không dừng ở việc đưa ticket
  cho người. Cache sai thì không cần kỹ sư viết code — chỉ cần nắn lại state. Model **tự chạy runbook
  RB-CACHE-02**: invalidate các entry stale ở cache, resync từ RTS — nguồn sự thật — rồi verify app đã
  khớp web. Có sẵn rollback. Ticket đóng, không ai viết một dòng code.”*
- **THÊM** nhãn mới: **WOW 2b — L2: model tự chữa, không chỉ chẩn đoán.**

### 3.5. TICKET 3 — ELIGIBILITY (nâng từ “route đúng đội” → **L3 sửa code**)  ⟵ ĐỈNH MỚI
- **GIỮ** toàn bộ đoạn “cặp song sinh / khác nhau đúng một chữ / chập chờn vs dai dẳng” — đây là vàng.
- **THAY** đoạn kết hiện tại (chỉ *“gửi ticket cho đội khác kèm runbook khác”*) → **đi tiếp vào L3**:
  *“Và vì đây là **rule sai trong code** — không phải state lệch như cache — model không dừng ở việc
  báo đội. Nó mở đúng file rule, thấy trim CR-V vùng US-West bị loại nhầm khỏi Touring, **sinh bản vá**
  đúng dòng đó, kèm **một unit test tái hiện đúng ca của khách** — đỏ trước khi vá, xanh sau khi vá.
  Con người chỉ bấm duyệt.”*
- **THÊM** nhãn: **WOW 3b — L3: từ chẩn đoán tới bản vá code + test, chờ người duyệt (Lv4).**
- **GIỮ** caveat “dữ liệu tự sinh, chưa chạm data Honda thật; PM2.0 trỏ vào incident thật” — đặt ngay đây.

### 3.6. (TÙY CHỌN) THÊM TICKET — PAYMENT_WEBHOOK_LOST / TOKEN_SCOPE
- Nếu muốn khoe **đủ 2 ví dụ mỗi tầng**: chèn 1 ticket **PAYMENT_WEBHOOK_LOST** (L2 · replay activation)
  hoặc **TOKEN_SCOPE** (L3 · vá scope-mapping). Webdemo đã có sẵn.
- Nếu hết giờ thì bỏ — 4 ticket gốc là đủ.

### 3.7. TICKET 4 — ABSTAIN
- **GIỮ** nguyên (WOW4 “biết im lặng khi thiếu dữ kiện”). Không sửa.

### 3.8. ĐÓNG (cập nhật phần đếm 4 ngón cho khớp L2/L3)
- **THAY** phần liệt kê để phản ánh **có xử lý**, không chỉ phân loại:
  - Ticket 1 (TCU): *xong — không ai động tay, không có gì để sửa.*
  - Ticket 2 (cache): *model **tự chữa** ở L2 — resync, verify, đóng ticket.*
  - Ticket 3 (eligibility): *model **ra bản vá code + test** ở L3, chờ duyệt.*
  - Ticket 4 (abstain): *giữ lại đúng cách, chưa đủ dữ kiện thì chưa giao.*
- **GIỮ** WOW5 (lock-in “tri thức không nghỉ việc / nằm trong model”) — kết thương mại rất mạnh.
- **GIỮ** dòng độ chính xác eligibility 62→74% (minh hoạ compound learning).

### 3.9. Bản đồ wow (trang 7) — cập nhật
- **THAY** dòng 2 (Ticket 2): Wow = **“L2 — model tự chữa (resync), không chỉ chẩn đoán.”**
- **THAY** dòng 3 (Ticket 3): Wow = **“L3 — ra bản vá code + test; và phân biệt cặp song sinh.”**
- **GIỮ** dòng 1, 4, 5.
- **THÊM** một dòng kỷ luật: *“Ticket 2 và 3 — sau khi chẩn đoán, BẤM sang tab Resolver cho họ xem nó
  tự xử lý. Đó là phần chủ tịch muốn thấy.”*

---

## 4. Câu một dòng để chốt với chủ tịch
> *“Màn 1 là triage — đưa đúng ticket cho đúng đội (giữ nguyên cái cũ). Màn 2 là thứ mới: cache thì
> model **tự resync** (L2), eligibility/token thì model **ra bản vá code + test** (L3), TCU thì **không
> đụng gì** (route), thiếu dữ kiện thì **im lặng** (abstain). Nó không chỉ phân loại — nó xử lý.”*
