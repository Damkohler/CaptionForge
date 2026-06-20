"""
JLC CaptionForge Template Options — ComfyUI Node Wrapper

- CaptionForge
  - This node is part of **CaptionForge**, a model-agnostic captioning
    framework for ComfyUI developed by **J. L. Córdova**.

  - Repository
    https://github.com/Damkohler/CaptionForge

- CaptionForge focuses on practical dataset-captioning infrastructure for
  LoRA dataset preparation, using multi-engine caption generation, JSONL
  audit trails, claim extraction and refinement, text-LLM distillation,
  image-aware VLM validation, and consensus-oriented caption improvement
  to produce grounded, auditable training captions.

- Node Purpose
    - The **JLC CaptionForge Template Options** node provides a shared
      prompt-template sidecar for CaptionForge caption nodes.

    - This file is the **ComfyUI-facing wrapper**, not a captioning engine.
      It is responsible for:
            • compact ComfyUI INPUT_TYPES / widget definitions
            • grouped character-oriented template option toggles
            • optional {name} replacement routing
            • CAPTIONFORGE_EXTRA_OPTIONS payload construction
            • JSON serialization of the selected template options
            • shared consumption by Joy, Qwen, Ollama, or future CaptionForge
              caption nodes through their `template_options` input
            • node display name, category, and mapping registration

- CaptionForge Pipeline Role
    - This node does not caption images directly.

    - It supplies reusable prompt modifiers to CaptionForge caption nodes.
      The selected options influence raw caption generation in standalone and
      Pipeline Planner workflows without duplicating option widgets inside
      every caption node.

    - CaptionForge audit payloads can preserve these prompt-option choices
      through caption-node records and resolved-prompt outputs.

- Node Workflow Model
    - The node emits two outputs:
            • CAPTIONFORGE_EXTRA_OPTIONS payload
            • readable JSON string of the same payload

    - Caption nodes consume the payload through their `template_options` pin.

    - The `name_input__replacement_for_name` widget is used only by options
      containing the `{name}` placeholder.

- Prompting Model
    - This node uses a character-captioning biased option set.

    - The options are intentionally grouped around:
            • caption behavior
            • face and identity traits
            • hair, skin, body, and material traits
            • clothing, accessories, and colors
            • pose, framing, and viewpoint
            • scene, lighting, and style
            • content and caption hygiene

    - Low-value generic photography options such as camera EXIF guessing,
      image resolution, subjective quality rating, JPEG artifacts, and
      aspect-ratio reporting are intentionally not exposed in this character
      Template Options node.

- Model and Dependency Notes
    - This node has no model dependency.

    - Runtime behavior depends only on the active ComfyUI Python environment
      and standard Python JSON handling.

- Design Philosophy
    - The Template Options node keeps prompt-template modifiers centralized so
      CaptionForge caption nodes remain cleaner and consistent.

    - The option set is tuned for LoRA-useful visible character description,
      especially stable traits such as face, eyes, hair, skin/material texture,
      body shape, clothing, accessories, pose, and style.

    - CaptionForge is engine-democratic: the same option payload can steer
      Joy, Qwen, Ollama VLMs, or future caption models without duplicating
      every checkbox in every caption node.

- ⚠️ Development Status
    - This is active CaptionForge raw-caption infrastructure.
    - The UI, option taxonomy, payload schema, and future configuration-file
      support may evolve as CaptionForge matures.
    - The node is intended for local dataset preparation and controlled caption
      audit workflows.

- Attribution & License
  - Concept and implementation by **J. L. Córdova**
    with development assistance from **ChatGPT (OpenAI)**.

  - CaptionForge's template-option workflow is locally adapted and was inspired
    in part by the practical template interface pattern used by the public
    JoyCaption Beta One Hugging Face Space.

  - Designed for use with:
    https://github.com/comfyanonymous/ComfyUI

  - Copyright (c) 2026 J. L. Córdova

  - Released under the **MIT License**.
"""

from __future__ import annotations
from ..captionforge_version import CAPTIONFORGE_VERSION

MANIFEST = {
    "name": "JLC CaptionForge Template Options",
    "version": CAPTIONFORGE_VERSION,
    "author": "J. L. Córdova",
    "description": (
        "Shared CaptionForge sidecar node that emits CAPTIONFORGE_EXTRA_OPTIONS "
        "payloads for caption-template controls. This revision replaces the older "
        "generic/photo-analysis option list with a grouped character-captioning "
        "biased set focused on visible LoRA-useful traits: face, eyes, makeup, hair, "
        "skin/material texture, body shape, clothing, accessories, pose, framing, "
        "scene, style, and content-hygiene controls."
    ),
}

import json
from typing import Any


CAPTIONFORGE_EXTRA_OPTIONS_SCHEMA = "captionforge.extra_options.v1"

# Canonical option keys and full prompt text remain stable for downstream payloads.
# The visible widget keys are intentionally descriptive because current ComfyUI
# builds display input dictionary keys as widget labels.
CAPTIONFORGE_EXTRA_OPTIONS_MAP: dict[str, str] = {
    # A. Caption behavior
    "option_01": "If there is a person or character in the image, refer to them as {name}.",
    "option_02": "Describe only visible details. Do not infer backstory, unseen anatomy, or hidden clothing.",
    "option_03": "Do not roleplay, continue a conversation, or write dialogue. Produce a direct descriptive caption only.",
    "option_04": "Use direct, specific wording and avoid ambiguous language when a visible trait can be described clearly.",
    "option_05": "Avoid meta lead-ins such as ‘This image shows’ or ‘You are looking at’. Your response should be a direct caption.",

    # B. Face and identity traits
    "option_06": "Describe visible facial features such as eye shape, eyebrows, lips, nose, and overall facial structure.",
    "option_07": "When visible, identify eye color and distinguish it clearly from eyeshadow, eyeliner, or other eye makeup.",
    "option_08": "If makeup is visible, describe it separately from the person’s natural facial traits.",
    "option_09": "Describe the visible facial expression or neutral expression if one is apparent.",

    # C. Hair, skin, body, and material traits
    "option_10": "Describe hair color, length, texture, and hairstyle as clearly as possible.",
    "option_11": "Describe visible skin tone and skin texture, including whether the skin appears smooth, glossy, matte, vinyl-like, realistic, or stylized.",
    "option_12": "If visible, describe overall body shape, proportions, and silhouette in a neutral descriptive way.",
    "option_13": "Describe which parts of the body are visibly shown and how much of the body is covered or uncovered by clothing, without inventing hidden details.",
    "option_14": "If the subject appears doll-like, figurine-like, animated, or rendered rather than photographic, describe those material or stylization traits explicitly.",

    # D. Clothing, accessories, and colors
    "option_15": "Describe visible clothing pieces individually rather than referring only to the outfit as a whole.",
    "option_16": "Describe clothing materials, finishes, and textures such as satin, leather, metallic, lace, sheer fabric, knit, or glossy surfaces when visible.",
    "option_17": "Describe the colors of important visual elements such as hair, eyes, clothing, accessories, and background as precisely as possible.",
    "option_18": "Describe visible accessories such as jewelry, hair ornaments, gloves, belts, necklaces, earrings, or other adornments.",

    # E. Pose, framing, and viewpoint
    "option_19": "Describe the visible pose, stance, and overall body positioning.",
    "option_20": "If visible, describe hand position, arm placement, and notable gesture details.",
    "option_21": "Mention whether the image is a close-up, medium shot, full-body shot, or another clear framing distance.",
    "option_22": "Describe the viewing angle or vantage point when it is visually clear, such as front view, side view, high angle, or low angle.",

    # F. Scene, lighting, and style
    "option_23": "Describe the visible background, setting, or backdrop, including whether it is plain, studio-like, natural, interior, or stylized.",
    "option_24": "Describe the visible lighting quality, such as soft, harsh, even, dramatic, bright, dim, natural, or studio lighting.",
    "option_25": "Describe whether the image appears photographic, illustrated, 3D-rendered, painterly, anime-styled, doll-like, or otherwise stylized.",

    # G. Content and caption hygiene
    "option_26": "Mention visible watermarks only if one is actually present.",
    "option_27": "Mention visible text only when it is clear and visually relevant to the image content.",
    "option_28": "Include whether the image appears SFW, suggestive, or NSFW.",
    "option_29": "Preserve visible sensual, revealing, or body-shape details in a neutral descriptive way when they are relevant to the image.",
    "option_30": "Use blunt vulgar wording when appropriate for explicit or NSFW captions, rather than over-sanitizing visible content.",
    "option_31": "Do not use polite euphemisms when explicit wording is requested; use direct descriptive language.",
}

# ComfyUI displays the input dictionary key as the visible widget name. These
# readable keys are therefore the actual visible labels, with option numbers
# preserved to keep order stable.
CAPTIONFORGE_EXTRA_OPTIONS_WIDGETS: list[tuple[str, str, str]] = [
    # A. Caption behavior
    ("option_01__character_name_required", "option_01", "Character name required"),
    ("option_02__literal_visible_details_only", "option_02", "Literal visible details only"),
    ("option_03__no_roleplay_or_dialogue", "option_03", "No roleplay or dialogue"),
    ("option_04__no_ambiguous_wording", "option_04", "No ambiguous wording"),
    ("option_05__avoid_meta_lead_ins", "option_05", "Avoid meta lead-ins"),

    # B. Face and identity traits
    ("option_06__describe_facial_features", "option_06", "Describe facial features"),
    ("option_07__eye_color_vs_makeup", "option_07", "Eye color vs makeup"),
    ("option_08__describe_makeup_separately", "option_08", "Describe makeup separately"),
    ("option_09__describe_expression", "option_09", "Describe expression"),

    # C. Hair, skin, body, and material traits
    ("option_10__hair_color_and_style", "option_10", "Hair color and style"),
    ("option_11__skin_tone_and_texture", "option_11", "Skin tone and texture"),
    ("option_12__body_shape_and_proportions", "option_12", "Body shape and proportions"),
    ("option_13__visible_anatomy_and_coverage", "option_13", "Visible anatomy and coverage"),
    ("option_14__stylized_material_traits", "option_14", "Stylized/material traits"),

    # D. Clothing, accessories, and colors
    ("option_15__clothing_pieces_clearly", "option_15", "Clothing pieces clearly"),
    ("option_16__clothing_materials_textures", "option_16", "Clothing materials/textures"),
    ("option_17__precise_colors", "option_17", "Precise colors"),
    ("option_18__accessories_and_jewelry", "option_18", "Accessories and jewelry"),

    # E. Pose, framing, and viewpoint
    ("option_19__pose_and_stance", "option_19", "Pose and stance"),
    ("option_20__hands_and_arms", "option_20", "Hands and arms"),
    ("option_21__framing_shot_distance", "option_21", "Framing / shot distance"),
    ("option_22__viewing_angle", "option_22", "Viewing angle"),

    # F. Scene, lighting, and style
    ("option_23__background_and_setting", "option_23", "Background and setting"),
    ("option_24__lighting", "option_24", "Lighting"),
    ("option_25__image_style_medium", "option_25", "Image style / medium"),

    # G. Content and caption hygiene
    ("option_26__watermark_only_if_present", "option_26", "Watermark only if present"),
    ("option_27__visible_text_if_relevant", "option_27", "Visible text if relevant"),
    ("option_28__sfw_suggestive_nsfw_rating", "option_28", "SFW/suggestive/NSFW rating"),
    ("option_29__neutral_sensual_details", "option_29", "Neutral sensual details"),
    ("option_30__vulgar_wording_allowed", "option_30", "Vulgar wording allowed"),
    ("option_31__no_polite_euphemisms", "option_31", "No polite euphemisms"),
]

CAPTIONFORGE_EXTRA_OPTIONS_CHOICES = [""] + [
    CAPTIONFORGE_EXTRA_OPTIONS_MAP[canonical_key]
    for _widget_key, canonical_key, _label in CAPTIONFORGE_EXTRA_OPTIONS_WIDGETS
]

_WIDGET_TO_CANONICAL = {widget: canonical for widget, canonical, _label in CAPTIONFORGE_EXTRA_OPTIONS_WIDGETS}
_CANONICAL_TO_LABEL = {canonical: label for _widget, canonical, label in CAPTIONFORGE_EXTRA_OPTIONS_WIDGETS}


def parse_captionforge_extra_options(payload: Any) -> dict[str, Any]:
    """Normalize an optional CAPTIONFORGE_EXTRA_OPTIONS / template-options payload."""
    if payload is None:
        return {}
    if isinstance(payload, dict):
        obj = dict(payload)
    elif isinstance(payload, str):
        text = payload.strip()
        if not text:
            return {}
        try:
            parsed = json.loads(text)
        except Exception:
            return {}
        obj = parsed if isinstance(parsed, dict) else {}
    else:
        return {}

    selected = obj.get("selected_options") or obj.get("selected_text") or obj.get("extra_options") or []
    if isinstance(selected, str):
        selected = [selected]
    selected = [str(x).strip() for x in selected if str(x).strip()]

    selected_keys = obj.get("selected_keys") or []
    if isinstance(selected_keys, str):
        selected_keys = [selected_keys]
    selected_keys = [str(x).strip() for x in selected_keys if str(x).strip()]

    selected_labels = obj.get("selected_labels") or []
    if isinstance(selected_labels, str):
        selected_labels = [selected_labels]
    selected_labels = [str(x).strip() for x in selected_labels if str(x).strip()]

    name_input = str(obj.get("name_input") or obj.get("person_name") or "").strip()

    return {
        "schema": str(obj.get("schema") or CAPTIONFORGE_EXTRA_OPTIONS_SCHEMA),
        "selected_options": selected,
        "selected_keys": selected_keys,
        "selected_labels": selected_labels,
        "name_input": name_input,
        "source": str(obj.get("source") or "captionforge_extra_options_payload"),
    }


def resolve_effective_extra_options(
    payload: Any = None,
    local_options: list[str] | tuple[str, ...] | None = None,
    local_name: str = "",
) -> tuple[list[str], str, dict[str, Any]]:
    """Return (options, name_input, metadata) with external-pin precedence."""
    parsed = parse_captionforge_extra_options(payload)
    external_options = list(parsed.get("selected_options") or [])
    if external_options:
        name = str(parsed.get("name_input") or local_name or "").strip()
        parsed["using_external_extra_options"] = True
        return external_options, name, parsed

    local = [str(x).strip() for x in (local_options or []) if str(x).strip()]
    meta = {
        "schema": CAPTIONFORGE_EXTRA_OPTIONS_SCHEMA,
        "selected_options": local,
        "selected_keys": [],
        "selected_labels": [],
        "name_input": str(local_name or "").strip(),
        "source": "caption_node_local_widgets",
        "using_external_extra_options": False,
    }
    return local, str(local_name or "").strip(), meta


class JLC_CaptionForgeExtraOptions:
    """Build a CAPTIONFORGE_EXTRA_OPTIONS payload for caption-template controls."""

    @classmethod
    def INPUT_TYPES(cls):
        required: dict[str, Any] = {
            "name_input__replacement_for_name": (
                "STRING",
                {
                    "default": "",
                    "multiline": False,
                    "tooltip": "Optional replacement for {name} in matching template options.",
                },
            ),
        }
        for widget_key, canonical_key, short_label in CAPTIONFORGE_EXTRA_OPTIONS_WIDGETS:
            required[widget_key] = (
                "BOOLEAN",
                {
                    "default": False,
                    "tooltip": f"{short_label}: {CAPTIONFORGE_EXTRA_OPTIONS_MAP[canonical_key]}",
                },
            )
        return {"required": required}

    RETURN_TYPES = ("CAPTIONFORGE_EXTRA_OPTIONS", "STRING")
    RETURN_NAMES = ("template_options", "template_options_json")
    FUNCTION = "build"
    CATEGORY = "Captioning/CaptionForge"

    def build(self, **kwargs):
        # Accept the old name_input key as a courtesy during hot-swaps, but the
        # visible key is now descriptive.
        name_input = str(
            kwargs.get("name_input__replacement_for_name")
            or kwargs.get("name_input")
            or ""
        ).strip()

        selected_options: list[str] = []
        selected_keys: list[str] = []
        selected_labels: list[str] = []

        for widget_key, canonical_key, label in CAPTIONFORGE_EXTRA_OPTIONS_WIDGETS:
            # Support both the new readable widget key and the canonical option_01 key.
            if bool(kwargs.get(widget_key, False)) or bool(kwargs.get(canonical_key, False)):
                selected_options.append(CAPTIONFORGE_EXTRA_OPTIONS_MAP[canonical_key])
                selected_keys.append(canonical_key)
                selected_labels.append(label)

        payload = {
            "schema": CAPTIONFORGE_EXTRA_OPTIONS_SCHEMA,
            "selected_keys": selected_keys,
            "selected_labels": selected_labels,
            "selected_options": selected_options,
            "name_input": name_input,
            "source": "captionforge_template_options_node",
        }
        return (payload, json.dumps(payload, ensure_ascii=False, indent=2))


NODE_CLASS_MAPPINGS = {
    "JLC_CaptionForgeExtraOptions": JLC_CaptionForgeExtraOptions,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "JLC_CaptionForgeExtraOptions": "\u2003JLC CaptionForge Template Options",
}
