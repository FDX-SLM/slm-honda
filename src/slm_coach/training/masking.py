"""Bật train-on-responses-only bằng cách chèn ``{% generation %}`` vào chat template.

TRL ``assistant_only_loss`` và ``tokenizer.apply_chat_template(..., return_assistant_tokens_mask=True)``
chỉ tính loss trên token assistant khi template bọc *nội dung* mỗi lượt assistant (gồm cả khối
``<think>`` literal) trong block Jinja ``{% generation %}…{% endgeneration %}``. Template gốc của các
base không có. Module này giữ các *patch theo base đã kiểm chứng* để thêm marker mà KHÔNG đổi cách
render gì khác, cộng một bước **verify**: chỉ bật masking khi vùng token unmask thực sự khớp đúng nội
dung assistant (chứng minh bằng ``return_assistant_tokens_mask``). Base không có patch an toàn — hoặc
template nuốt ``<think>`` (Gemma ``strip_thinking`` / Qwen ``reasoning_content``) — sẽ tự degrade về
full-sequence loss thay vì crash.

Cơ chế chọn patch là *model-agnostic*: không cần biết alias base — chỉ patch nào có đủ anchor (mọi
chuỗi ``old`` đều xuất hiện) trong template mới được áp, sau đó verify mới chấp nhận.
"""

from __future__ import annotations

from typing import Any

from slm_coach.utils.logging import get_logger

logger = get_logger(__name__)

#: Chuỗi đánh dấu vùng loss. TRL/transformers nhận diện block Jinja này.
_GEN_OPEN = "{% generation %}"
_GEN_CLOSE = "{% endgeneration %}"

#: Patch theo base: danh sách (old, new) thay-thế-chuỗi, bọc *nội dung* lượt assistant trong
#: ``{% generation %}…{% endgeneration %}``. Mỗi ``old`` phải là substring DUY NHẤT & chính xác trong
#: chat template gốc của base đó. Một bộ patch chỉ được áp khi TẤT CẢ ``old`` của nó có trong template.
GENERATION_PATCHES: dict[str, list[tuple[str, str]]] = {
    # Granite 4.x: header và content tách rời; nhánh assistant đóng bằng end-of-text đứng riêng
    # (anchor kèm ``{%- endif %}`` phía trước + ``{%- elif ... 'tool' %}`` phía sau cho duy nhất,
    # vì ``{{- '<|end_of_text|>\\n' }}`` còn xuất hiện ở nhánh tool).
    "granite": [
        (
            "{{- '<|start_of_role|>' + message.role + '<|end_of_role|>' + content.val }}",
            "{{- '<|start_of_role|>' + message.role + '<|end_of_role|>' }}"
            + _GEN_OPEN
            + "{{- content.val }}",
        ),
        (
            "        {%- endif %}\n        {{- '<|end_of_text|>\\n' }}\n    "
            "{%- elif message.role == 'tool' %}",
            "        {%- endif %}\n        {{- '<|end_of_text|>' }}"
            + _GEN_CLOSE
            + "{{- '\\n' }}\n    {%- elif message.role == 'tool' %}",
        ),
    ],
    # Phi-4: ChatML một dòng, nhánh assistant gọn — bọc content + <|im_end|>.
    "phi": [
        (
            "{% elif (message['role'] == 'assistant') %}"
            "{{'<|im_start|>assistant<|im_sep|>' + message['content'] + '<|im_end|>'}}",
            "{% elif (message['role'] == 'assistant') %}{{'<|im_start|>assistant<|im_sep|>'}}"
            + _GEN_OPEN
            + "{{message['content'] + '<|im_end|>'}}"
            + _GEN_CLOSE,
        ),
    ],
}


#: Some stock templates can't be safely patched by substring (capture-then-emit, reasoning
#: reconstruction, multi-pass). For those, install a minimal *training* template that renders the
#: SAME tokens for our single-turn data, keeps ``<think>`` literal, and wraps the assistant content
#: in ``{% generation %}``. Keyed by a unique substring that identifies the stock template; gated by
#: ``_verify_mask`` like every other path. Qwen is plain ChatML, so a minimal ChatML template is
#: token-identical to the stock render for a normal assistant turn (minus the reasoning_content
#: reconstruction, which only re-derives the ``<think>`` our content already carries literally).
_QWEN_CHATML = (
    "{%- for message in messages %}\n"
    "{%- if message['role'] == 'system' %}\n"
    "{{- '<|im_start|>system\\n' + message['content'] + '<|im_end|>\\n' }}\n"
    "{%- elif message['role'] == 'user' %}\n"
    "{{- '<|im_start|>user\\n' + message['content'] + '<|im_end|>\\n' }}\n"
    "{%- elif message['role'] == 'assistant' %}\n"
    "{{- '<|im_start|>assistant\\n' }}"
    "{% generation %}{{- message['content'] + '<|im_end|>' }}{% endgeneration %}"
    "{{- '\\n' }}\n"
    "{%- endif %}\n"
    "{%- endfor %}\n"
    "{%- if add_generation_prompt %}{{- '<|im_start|>assistant\\n' }}{% endif %}"
)

#: Gemma-4 stock template uses a capture-then-emit pattern + a native "thought channel" generation
#: prompt + ``strip_thinking`` — none of which can be substring-patched safely. This minimal template
#: reproduces gemma's exact turn format (``<bos><|turn>{role}\n{content}<turn|>\n``, assistant role
#: = ``model``), keeps ``<think>`` literal, and uses a plain ``<|turn>model\n`` generation prompt so
#: train/inference both use the literal ``<think>`` we trained on (not the native thought channel).
_GEMMA_TURN = (
    "{{- '<bos>' }}\n"
    "{%- for message in messages %}\n"
    "{%- if message['role'] == 'assistant' %}\n"
    "{{- '<|turn>model\\n' }}"
    "{% generation %}{{- message['content'] + '<turn|>' }}{% endgeneration %}{{- '\\n' }}\n"
    "{%- else %}\n"
    "{{- '<|turn>' + message['role'] + '\\n' + message['content'] + '<turn|>\\n' }}\n"
    "{%- endif %}\n"
    "{%- endfor %}\n"
    "{%- if add_generation_prompt %}{{- '<|turn>model\\n' }}{% endif %}"
)

FULL_TEMPLATE_OVERRIDES: list[tuple[str, str]] = [
    ("render_content", _QWEN_CHATML),  # qwen stock template uses a render_content() macro
    ("strip_thinking", _GEMMA_TURN),  # gemma-4 stock template defines a strip_thinking() macro
]


def has_generation_markers(template: str | None) -> bool:
    """Template đã có block ``{% generation %}`` thật chưa (không phải chữ ``add_generation_prompt``)."""
    import re

    return bool(template and re.search(r"\{%-?\s*generation\s*-?%\}", template))


def _apply_patch(template: str) -> str | None:
    """Trả về template đã chèn marker nếu có đúng một bộ patch khớp, ngược lại ``None``."""
    for base, patches in GENERATION_PATCHES.items():
        if all(old in template for old, _ in patches):
            patched = template
            for old, new in patches:
                if patched.count(old) != 1:  # anchor phải duy nhất, nếu không là không an toàn
                    logger.warning(
                        "Masking patch anchor not unique; skipping",
                        extra={"base": base, "count": patched.count(old)},
                    )
                    return None
                patched = patched.replace(old, new)
            logger.info("Applied generation-marker patch", extra={"base": base})
            return patched
    return None


def _verify_mask(tokenizer: Any) -> bool:
    """Render một mẫu 1-lượt và xác nhận vùng unmask = đúng nội dung assistant (gồm ``<think>``).

    An toàn-là-trên-hết: bất kỳ lỗi/khớp-sai nào cũng trả ``False`` để caller degrade về full-seq loss.
    """
    sample = [
        {"role": "system", "content": "SYSTEM_SENTINEL_PROMPT do not learn this."},
        {"role": "user", "content": "USER_SENTINEL complaint text here."},
        {
            "role": "assistant",
            "content": '<think>\nASSISTANT_REASONING_SENTINEL\n</think>\n{"k": 1}',
        },
    ]
    try:
        enc = tokenizer.apply_chat_template(
            sample,
            tokenize=True,
            return_dict=True,
            return_assistant_tokens_mask=True,
        )
    except Exception as exc:  # noqa: BLE001 - degrade on any template/tokenizer error
        logger.warning("Mask verification render failed", extra={"error": str(exc)})
        return False

    masks = enc.get("assistant_masks")
    ids = enc.get("input_ids")
    if not masks or not ids or sum(masks) == 0:
        return False

    unmasked = tokenizer.decode([t for t, m in zip(ids, masks, strict=False) if m])
    masked = tokenizer.decode([t for t, m in zip(ids, masks, strict=False) if not m])
    # Vùng tính loss PHẢI chứa reasoning + JSON assistant, gồm cả <think> literal...
    ok_unmasked = (
        "ASSISTANT_REASONING_SENTINEL" in unmasked
        and "<think>" in unmasked
        and '"k": 1' in unmasked
    )
    # ...và TUYỆT ĐỐI không được rò system prompt / user complaint vào vùng loss.
    no_leak = "SYSTEM_SENTINEL_PROMPT" not in unmasked and "USER_SENTINEL" not in unmasked
    # ...và prompt phải nằm ở vùng bị mask.
    prompt_masked = "SYSTEM_SENTINEL_PROMPT" in masked and "USER_SENTINEL" in masked
    return ok_unmasked and no_leak and prompt_masked


def enable_assistant_masking(tokenizer: Any) -> bool:
    """Bật train-on-responses-only cho tokenizer này nếu chứng minh được là an toàn.

    Thứ tự: (1) đã có marker → verify luôn; (2) chưa có → thử patch theo base, áp rồi verify; chỉ khi
    verify đạt mới gán lại ``tokenizer.chat_template``. Trả ``True`` nếu masking đã bật & đã verify.

    Args:
        tokenizer: Tokenizer của base (``chat_template`` sẽ bị mutate tại chỗ khi patch thành công).

    Returns:
        ``True`` nếu ``assistant_only_loss`` dùng được; ``False`` để caller degrade về full-seq loss.
    """
    template = getattr(tokenizer, "chat_template", None) or ""
    if has_generation_markers(template):
        return _verify_mask(tokenizer)

    original = tokenizer.chat_template

    # (a) try a surgical substring patch on the stock template (granite/phi)
    patched = _apply_patch(template)
    if patched is not None:
        tokenizer.chat_template = patched
        if _verify_mask(tokenizer):
            logger.info("assistant_only_loss enabled via verified generation-marker patch")
            return True
        tokenizer.chat_template = original

    # (b) else try a minimal verified training template for templates we can't patch (qwen)
    for detector, tmpl in FULL_TEMPLATE_OVERRIDES:
        if detector in template:
            tokenizer.chat_template = tmpl
            if _verify_mask(tokenizer):
                logger.info(
                    "assistant_only_loss enabled via verified training-template override",
                    extra={"detector": detector},
                )
                return True
            tokenizer.chat_template = original
            break

    logger.warning("Could not enable verified masking; reverting to stock template (full-seq loss)")
    return False
