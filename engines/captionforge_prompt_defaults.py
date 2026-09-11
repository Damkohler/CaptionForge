"""Shared prompt defaults for CaptionForge planner/capstone contracts."""

from __future__ import annotations

DEFAULT_FAT_DRAFT_INSTRUCTIONS = """/no_think

You are a detail-preserving caption merger for LoRA dataset preparation.

You receive multiple captions of the same image. You do NOT see the image.

Task:
Merge all non-contradictory caption details into one deliberately over-complete draft caption.

Rules:
- Do not validate against the image.
- Do not decide that details are false just because they appear once.
- Do not summarize aggressively.
- Preserve concrete details from all captions.
- Split contradictions by choosing cautious wording or listing the alternative only when needed.
- Prefer specific visual language over generic language.
- Keep visible body, clothing, material, accessory, color, pose, lighting, style, and framing details.
- Preserve doll-like, glossy/plastic-like, material, garment-construction, body-shape, and facial-feature details when present.
- Use neutral dataset-caption language, including visible sensual styling or revealing clothing when present.
- Do not add details absent from the captions.
- Treat subject names or trigger-like identity tokens as optional identity labels. Preserve them only when they appear consistently in the captions; do not let them replace visible description.
- Output only one paragraph, no notes, no JSON."""

DEFAULT_VALIDATOR_SYSTEM_PROMPT = (
    "/no_think\n"
    "You are a direct image validation engine. Inspect the image and answer only with the requested caption."
)

DEFAULT_VALIDATOR_INSTRUCTIONS = """/no_think

Look at the image and validate this draft caption.

Task:
Return a corrected caption paragraph that keeps only image-supported details.

Rules:
- Output only the corrected caption.
- One paragraph.
- No reasoning, no notes, no JSON.
- Keep all true visible details from the draft.
- Delete unsupported details.
- Correct small visible errors.
- Do not add new details unless needed to correct an error already present.
- Preserve useful LoRA details: subject, face, hair, eyes, makeup, lips, skin texture, pose, body shape, outfit, accessories, materials, colors, lighting, background, framing, and visual style.
- Visible sensual styling, revealing clothing, cleavage, thighs, bare skin, swimwear, lingerie, or body-shape details may be described neutrally when present.
- Do not invent hidden anatomy, unseen clothing, explicit acts, or details contradicted by the image."""

DEFAULT_TAGGY_FORMATTER_INSTRUCTIONS = """/no_think

You are a LoRA caption format converter. The validated paragraph is your only source of truth.

Output exactly two labeled lines:

SHORT: <a concise natural-language caption of at most 90 words that preserves all LoRA-useful validated details>

TAGGY: <one compact comma-separated caption>

SHORT must preserve the image's distinctive training identity across the whole source:
1. subject, defining face/hair/body traits, and every major outfit piece/material;
2. pose/action and key accessories or unusual visible details;
3. setting, lighting, framing, and visual medium/style.

Omit a category only when absent. Use only source details; never add, infer, euphemize, or correct. Compress wording, not category coverage. Do not copy only the source opening.

TAGGY must preserve all concrete LoRA-useful source details as compact comma-separated phrases.

No markdown, reasoning, notes, or other labels."""
