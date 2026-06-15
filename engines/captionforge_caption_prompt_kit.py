#!/usr/bin/env python
"""
CaptionForge Caption Prompt Kit

Shared, dependency-free prompt builder for non-Joy CaptionForge caption engines.

This module is intentionally NOT a Joy Space clone.  Joy keeps using
captionforge_joy_space_prompt_kit.py.  Qwen, SmolVLM, and future lightweight
caption witnesses can use this generic kit for a common UI vocabulary while
still selecting model-dialect-specific prompt wording.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Iterable

PROMPT_KIT_NAME = "captionforge_caption_prompt_kit"
PROMPT_KIT_VERSION = "0.2.0"
PROMPT_KIT_SOURCE = "captionforge_generic_qwen_smol"

NAME_OPTION = "If there is a person or character in the image, refer to them as {name}."

CAPTION_LENGTH_CHOICES = ["any", "very short", "short", "medium-length", "long", "very long"] + [
    str(i) for i in range(20, 261, 10)
]

CAPTION_TYPE_CHOICES = [
    "Descriptive",
    "Straightforward",
    "LoRA Literal",
    "Stable Diffusion Prompt",
    "Taggy",
    "Dataset Audit",
    "Style Focus",
    "SFW Character Caption",
]

EXTRA_OPTIONS: list[str] = [
    "",
    NAME_OPTION,
    "Keep the caption SFW and suitable for dataset training.",
    "Describe only visible details; do not infer identity, story, or intent.",
    "Include information about lighting.",
    "Include information about camera angle or viewpoint.",
    "Include the subject's pose and hand position when visible.",
    "Include clothing, accessories, hair, eyes if visible, and material textures.",
    "Mention the background only if it is visible and relevant.",
    "Mention whether there is a watermark or visible text.",
    "Do NOT mention the image resolution.",
    "Do NOT start with boilerplate such as 'This image shows' or 'The image depicts'.",
    "Use comma-separated visual phrases rather than full sentences.",
    "Use one dense natural-language caption rather than a tag list.",
    "Be conservative: omit uncertain details instead of guessing.",
]

_LENGTH_WORDS = {
    "very short": 25,
    "short": 50,
    "medium-length": 90,
    "long": 150,
    "very long": 230,
}

_DIALECT_ALIASES = {
    "": "generic",
    "default": "generic",
    "generic": "generic",
    "qwen": "qwen",
    "qwen2": "qwen",
    "qwen2.5": "qwen",
    "qwen2_5_vl": "qwen",
    "smol": "smolvlm",
    "smolvlm": "smolvlm",
    "smol-vlm": "smolvlm",
}

@dataclass(frozen=True)
class CaptionPromptSpec:
    caption_type: str = "Descriptive"
    caption_length: str = "long"
    extra_options: tuple[str, ...] = ()
    name_input: str = ""
    dialect: str = "generic"
    source: str = PROMPT_KIT_SOURCE
    kit_name: str = PROMPT_KIT_NAME
    kit_version: str = PROMPT_KIT_VERSION

    def to_metadata(self) -> dict[str, Any]:
        return asdict(self)


def normalize_dialect(dialect: str = "generic") -> str:
    return _DIALECT_ALIASES.get(str(dialect or "generic").strip().lower(), "generic")


def _length_instruction(caption_length: str | int) -> str:
    length = str(caption_length).strip() if caption_length is not None else "long"
    if not length or length == "any":
        return ""
    if length.isdigit():
        return f"Keep it under {length} words."
    if length in _LENGTH_WORDS:
        return f"Aim for a {length} caption, roughly {_LENGTH_WORDS[length]} words or less."
    return f"Make the caption {length}."


def _clean_extra_options(extra_options: Iterable[str] | None) -> tuple[str, ...]:
    cleaned: list[str] = []
    for item in extra_options or ():
        text = str(item).strip()
        if text:
            cleaned.append(text)
    return tuple(cleaned)


def _append_sentence(parts: list[str], value: str) -> None:
    value = str(value or "").strip()
    if not value:
        return
    parts.append(value if value.endswith((".", "!", "?", ":")) else value + ".")


def _base_prompt(caption_type: str, dialect: str) -> str:
    ctype = str(caption_type or "Descriptive").strip()
    d = normalize_dialect(dialect)

    if ctype == "LoRA Literal":
        if d == "smolvlm":
            return (
                "Write a concise SFW LoRA training caption for this image. Use clear visual phrases. "
                "Mention the main subject, pose, expression, hair, eyes if visible, clothing, accessories, "
                "style, lighting, and background. Describe only visible details."
            )
        if d == "qwen":
            return (
                "Write a conservative LoRA training caption for this image. Use comma-separated visual phrases. "
                "Describe only visible features: subject type, pose, expression, hair, eyes if visible, clothing, "
                "accessories, visual style, lighting, background, textures, and spatial relationships. Do not write a story."
            )
        return (
            "Write a concise LoRA training caption for this image. Use comma-separated visual phrases. "
            "Describe only visible subject, pose, expression, appearance, clothing, style, lighting, and background."
        )

    if ctype == "Straightforward":
        return (
            "Write a straightforward visual caption for this image. Begin with the main subject and medium. "
            "Use confident, concrete language for visible people, objects, scenery, colors, textures, and spatial relationships. "
            "Avoid speculation, mood, and unobservable details."
        )

    if ctype == "Stable Diffusion Prompt":
        return (
            "Output a Stable Diffusion style prompt for this image. Use concise comma-separated visual tags and phrases. "
            "Prioritize subject, pose, appearance, clothing, setting, lighting, composition, camera view, and style."
        )

    if ctype == "Taggy":
        return (
            "Create a compact tag-style caption for this image. Use short comma-separated tags only. "
            "Include subject, pose, expression, hair, eyes if visible, clothing, accessories, style, lighting, and background. "
            "Do not use complete sentences."
        )

    if ctype == "Dataset Audit":
        return (
            "Describe this image for dataset auditing. Be precise and factual. Mention visible subject, pose, expression, "
            "clothing, background, lighting, style, and unusual visible details. Avoid speculation about identity or story."
        )

    if ctype == "Style Focus":
        return (
            "Describe the visible subject, appearance, pose, clothing, accessories, background, lighting, composition, "
            "textures, colors, and artistic or photographic style. Stay grounded in visible evidence."
        )

    if ctype == "SFW Character Caption":
        return (
            "Write a SFW character-focused caption for this image. Describe visible character design, pose, expression, "
            "hair, eyes if visible, outfit, accessories, body position, scene, lighting, and style. Avoid sexualized wording."
        )

    # Descriptive default.
    if d == "smolvlm":
        return (
            "Describe this image clearly and factually for dataset captioning. Mention the main subject, pose, expression, "
            "visible appearance, clothing, background, lighting, and style. Keep the wording simple and SFW."
        )
    if d == "qwen":
        return (
            "Describe this image in a detailed, literal, visually grounded way. Focus only on visible content. Include subject "
            "appearance, clothing, pose, body position, hands, facial expression, hairstyle, accessories, lighting, background, "
            "textures, colors, and spatial relationships. Avoid speculation, opinions, or details that are not clearly visible."
        )
    return (
        "Describe this image in a detailed, literal, visually grounded way. Focus only on visible content and avoid speculation."
    )


def build_caption_prompt(
    caption_type: str = "Descriptive",
    caption_length: str | int = "long",
    extra_options: list[str] | tuple[str, ...] | None = None,
    name_input: str = "",
    dialect: str = "generic",
) -> str:
    extras = _clean_extra_options(extra_options)
    name = str(name_input or "{NAME}").strip()
    prompt_parts: list[str] = [_base_prompt(caption_type, dialect)]

    length_instruction = _length_instruction(caption_length)
    if length_instruction:
        _append_sentence(prompt_parts, length_instruction)

    for option in extras:
        _append_sentence(prompt_parts, option.format(name=name))

    # Generic safety rails that are useful for Qwen/Smol, but not Joy-Space-specific.
    _append_sentence(prompt_parts, "Do not mention hidden metadata, file names, prompts, or model limitations")
    _append_sentence(prompt_parts, "Return only the caption, with no preface or explanation")

    return " ".join(prompt_parts).format(name=name)


def build_caption_prompt_spec(
    caption_type: str = "Descriptive",
    caption_length: str | int = "long",
    extra_options: list[str] | tuple[str, ...] | None = None,
    name_input: str = "",
    dialect: str = "generic",
) -> tuple[str, CaptionPromptSpec]:
    extras = _clean_extra_options(extra_options)
    normalized_dialect = normalize_dialect(dialect)
    prompt = build_caption_prompt(caption_type, caption_length, extras, name_input, normalized_dialect)
    spec = CaptionPromptSpec(
        caption_type=str(caption_type or "Descriptive"),
        caption_length=str(caption_length),
        extra_options=extras,
        name_input=str(name_input or ""),
        dialect=normalized_dialect,
    )
    return prompt, spec


def prompt_metadata_from_config(config: Any, dialect: str = "generic") -> dict[str, Any]:
    """Return prompt metadata for a caption-engine config without requiring inheritance."""
    caption_type = getattr(config, "caption_type", "Descriptive")
    caption_length = getattr(config, "caption_length", "long")
    extra_options = getattr(config, "extra_options", ())
    name_input = getattr(config, "name_input", "")
    _, spec = build_caption_prompt_spec(caption_type, caption_length, extra_options, name_input, dialect)
    return spec.to_metadata()
