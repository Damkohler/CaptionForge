"""
CaptionForge Template Options — ComfyUI Node Wrapper

A small prompt-template sidecar node for CaptionForge caption witnesses.
It exposes template option toggles once and passes the selected instructions
to Joy, Qwen, SmolVLM, or future caption nodes.
"""
from __future__ import annotations

MANIFEST = {
    "name": "JLC CaptionForge Template Options",
    "version": (0, 1, 3),
    "author": "J. L. Córdova",
    "description": "Shared CaptionForge sidecar node that emits CAPTIONFORGE_EXTRA_OPTIONS payloads for caption-template controls.",
}

import json
from typing import Any

try:
    from ..engines.captionforge_joy_space_prompt_kit import EXTRA_OPTIONS as _JOY_SPACE_EXTRA_OPTIONS
except Exception:
    _JOY_SPACE_EXTRA_OPTIONS = [
        'If there is a person/character in the image you must refer to them as {name}.',
        'Do NOT include information about people/characters that cannot be changed (like ethnicity, gender, etc), but do still include changeable attributes (like hair style).',
        'Include information about lighting.',
        'Include information about camera angle.',
        'Include information about whether there is a watermark or not.',
        'Include information about whether there are JPEG artifacts or not.',
        'If it is a photo you MUST include information about what camera was likely used and details such as aperture, shutter speed, ISO, etc.',
        'Do NOT include anything sexual; keep it PG.',
        "Do NOT mention the image's resolution.",
        'You MUST include information about the subjective aesthetic quality of the image from low to very high.',
        "Include information on the image's composition style, such as leading lines, rule of thirds, or symmetry.",
        'Do NOT mention any text that is in the image.',
        'Specify the depth of field and whether the background is in focus or blurred.',
        'If applicable, mention the likely use of artificial or natural lighting sources.',
        'Do NOT use any ambiguous language.',
        'Include whether the image is sfw, suggestive, or nsfw.',
        'ONLY describe the most important elements of the image.',
        "If it is a work of art, do not include the artist's name or the title of the work.",
        'Identify the image orientation (portrait, landscape, or square) and aspect ratio if obvious.',
        'Use vulgar slang and profanity, such as (but not limited to) "fucking," "slut," "cock," etc.',
        'Do NOT use polite euphemisms—lean into blunt, casual phrasing.',
        'Include information about the ages of any people/characters when applicable.',
        'Mention whether the image depicts an extreme close-up, close-up, medium close-up, medium shot, cowboy shot, medium wide shot, wide shot, or extreme wide shot.',
        'Do not mention the mood/feeling/etc of the image.',
        'Explicitly specify the vantage height (eye-level, low-angle worm’s-eye, bird’s-eye, drone, rooftop, etc.).',
        'If there is a watermark, you must mention it.',
        'Your response will be used by a text-to-image model, so avoid useless meta phrases like “This image shows…”, "You are looking at...", etc.',
    ]

CAPTIONFORGE_EXTRA_OPTIONS_SCHEMA = "captionforge.extra_options.v1"
CAPTIONFORGE_EXTRA_OPTIONS_CHOICES = [""] + [str(x) for x in _JOY_SPACE_EXTRA_OPTIONS if str(x).strip()]

# Canonical option keys and full prompt text remain stable for downstream payloads.
CAPTIONFORGE_EXTRA_OPTIONS_MAP: dict[str, str] = {
    "option_01": 'If there is a person/character in the image you must refer to them as {name}.',
    "option_02": 'Do NOT include information about people/characters that cannot be changed (like ethnicity, gender, etc), but do still include changeable attributes (like hair style).',
    "option_03": 'Include information about lighting.',
    "option_04": 'Include information about camera angle.',
    "option_05": 'Include information about whether there is a watermark or not.',
    "option_06": 'Include information about whether there are JPEG artifacts or not.',
    "option_07": 'If it is a photo you MUST include information about what camera was likely used and details such as aperture, shutter speed, ISO, etc.',
    "option_08": 'Do NOT include anything sexual; keep it PG.',
    "option_09": "Do NOT mention the image's resolution.",
    "option_10": 'You MUST include information about the subjective aesthetic quality of the image from low to very high.',
    "option_11": "Include information on the image's composition style, such as leading lines, rule of thirds, or symmetry.",
    "option_12": 'Do NOT mention any text that is in the image.',
    "option_13": 'Specify the depth of field and whether the background is in focus or blurred.',
    "option_14": 'If applicable, mention the likely use of artificial or natural lighting sources.',
    "option_15": 'Do NOT use any ambiguous language.',
    "option_16": 'Include whether the image is sfw, suggestive, or nsfw.',
    "option_17": 'ONLY describe the most important elements of the image.',
    "option_18": "If it is a work of art, do not include the artist's name or the title of the work.",
    "option_19": 'Identify the image orientation (portrait, landscape, or square) and aspect ratio if obvious.',
    "option_20": 'Use vulgar slang and profanity, such as (but not limited to) "fucking," "slut," "cock," etc.',
    "option_21": 'Do NOT use polite euphemisms—lean into blunt, casual phrasing.',
    "option_22": 'Include information about the ages of any people/characters when applicable.',
    "option_23": 'Mention whether the image depicts an extreme close-up, close-up, medium close-up, medium shot, cowboy shot, medium wide shot, wide shot, or extreme wide shot.',
    "option_24": 'Do not mention the mood/feeling/etc of the image.',
    "option_25": 'Explicitly specify the vantage height (eye-level, low-angle worm’s-eye, bird’s-eye, drone, rooftop, etc.).',
    "option_26": 'If there is a watermark, you must mention it.',
    "option_27": 'Your response will be used by a text-to-image model, so avoid useless meta phrases like “This image shows…”, "You are looking at...", etc.',
}

# ComfyUI displays the input dictionary key as the visible widget name. The
# previous implementation used option_01 + {"label": "..."}, but current
# ComfyUI builds do not render that label in the widget row. These readable
# keys are therefore the actual visible labels.
CAPTIONFORGE_EXTRA_OPTIONS_WIDGETS: list[tuple[str, str, str]] = [
    ("option_01__character_name_required", "option_01", "Character name required"),
    ("option_02__skip_immutable_traits", "option_02", "Skip immutable traits"),
    ("option_03__lighting_details", "option_03", "Lighting details"),
    ("option_04__camera_angle", "option_04", "Camera angle"),
    ("option_05__watermark_status", "option_05", "Watermark status"),
    ("option_06__jpeg_artifact_status", "option_06", "JPEG artifact status"),
    ("option_07__camera_exif_guess", "option_07", "Camera EXIF guess"),
    ("option_08__pg_non_sexual", "option_08", "PG / non-sexual"),
    ("option_09__no_resolution_mention", "option_09", "No resolution mention"),
    ("option_10__aesthetic_quality_rating", "option_10", "Aesthetic quality rating"),
    ("option_11__composition_style", "option_11", "Composition style"),
    ("option_12__ignore_image_text", "option_12", "Ignore image text"),
    ("option_13__depth_of_field", "option_13", "Depth of field"),
    ("option_14__light_source_type", "option_14", "Light source type"),
    ("option_15__no_ambiguous_wording", "option_15", "No ambiguous wording"),
    ("option_16__sfw_rating", "option_16", "SFW rating"),
    ("option_17__key_elements_only", "option_17", "Key elements only"),
    ("option_18__no_artist_title", "option_18", "No artist/title"),
    ("option_19__orientation_aspect_ratio", "option_19", "Orientation/aspect ratio"),
    ("option_20__vulgar_language_allowed", "option_20", "Vulgar language allowed"),
    ("option_21__blunt_phrasing", "option_21", "Blunt phrasing"),
    ("option_22__apparent_age", "option_22", "Apparent age"),
    ("option_23__shot_distance", "option_23", "Shot distance"),
    ("option_24__no_mood_language", "option_24", "No mood language"),
    ("option_25__vantage_height", "option_25", "Vantage height"),
    ("option_26__watermark_if_present", "option_26", "Watermark if present"),
    ("option_27__no_meta_phrases", "option_27", "No meta phrases"),
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
    @classmethod
    def INPUT_TYPES(cls):
        required: dict[str, Any] = {
            "name_input__replacement_for_name": (
                "STRING",
                {
                    "default": "",
                    "multiline": False,
                    "tooltip": "Optional replacement for {name} in matching extra options.",
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
    CATEGORY = "JLC/Captioning"

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
            # Support both the new readable widget key and the old option_01 key.
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
    "JLC_CaptionForgeExtraOptions": " JLC CaptionForge Template Options",
}
