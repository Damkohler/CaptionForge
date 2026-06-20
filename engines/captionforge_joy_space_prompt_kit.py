"""
CaptionForge Joy Space Prompt Kit

- CaptionForge
  - This module is part of **CaptionForge**, a model-agnostic captioning
    framework for ComfyUI developed by **J. L. Córdova**.

  - Repository:
    https://github.com/Damkohler/CaptionForge

- CaptionForge focuses on practical dataset-captioning infrastructure for
  LoRA dataset preparation, using multi-engine caption generation, JSONL
  audit trails, claim extraction and refinement, text-LLM distillation,
  image-aware VLM validation, and consensus-oriented caption improvement
  to produce grounded, auditable training captions.

- Module Purpose
    - The **CaptionForge Joy Space Prompt Kit** is a small, dependency-free
      helper for JoyCaption-style prompt construction inside local CaptionForge
      Joy caption nodes.

    - It preserves a Joy-specific prompt harness separate from the generic
      CaptionForge prompt kit.

    - It defines:
            • JoyCaption-style caption type templates
            • caption length choices
            • extra-option text
            • name-substitution behavior
            • a Joy prompt metadata dataclass
            • helpers for building prompt text and metadata together

- CaptionForge Pipeline Role
    - This module supports **Pass A** Joy raw caption witness nodes.

    - It does not caption images and does not load Joy model weights.

    - Joy caption nodes use this kit to build the user prompt and prompt
      metadata sent to the Joy caption engine, then record the resolved prompt
      in CaptionForge outputs.

- Prompting Model
    - Joy keeps a dedicated prompt kit because its practical behavior is tied to
      the JoyCaption Beta One prompt interface pattern.

    - Generic CaptionForge prompt construction for Qwen, SmolVLM, Ollama, and
      future caption witnesses lives separately in:

            captionforge_caption_prompt_kit.py

    - This separation avoids forcing Joy and non-Joy engines into one artificial
      prompt abstraction.

- Model and Dependency Notes
    - This file contains prompt text, option lists, and metadata helpers only.

    - It does not execute the Joy model, download model files, manage VRAM, or
      call Hugging Face directly.

    - Joy model loading and inference are handled by the Joy caption engine and
      node wrapper.

- Design Philosophy
    - CaptionForge should preserve Joy's useful prompt-harness behavior while
      keeping it local, auditable, and isolated from other caption engines.

    - Prompt construction should remain deterministic and dependency-free.

    - Prompt metadata should make it clear which caption type, length, options,
      name input, and system prompt were used for a Joy caption run.

- Development Status
    - CaptionForge v0.1.0 experimental developer-preview infrastructure.
    - Joy prompt templates and option lists may evolve as the local CaptionForge
      Joy node matures.

- Attribution & License
  - Concept and implementation by **J. L. Córdova**
    with development assistance from **ChatGPT (OpenAI)**.

  - CaptionForge's Joy/template-option workflow is locally adapted and was
    inspired in part by the practical template interface pattern used by the
    public JoyCaption Beta One Hugging Face Space:
    https://huggingface.co/spaces/fffiloni/JoyCaption-Beta-One

  - The local Joy caption workflow also references JoyCaption Beta One naming
    and model identifiers used by the Joy caption ecosystem.

  - Designed for use with:
    https://github.com/comfyanonymous/ComfyUI

  - Copyright (c) 2026 J. L. Córdova

  - Released under the **MIT License**.
"""

from __future__ import annotations

from ..captionforge_version import CAPTIONFORGE_VERSION

MANIFEST = {
    "name": "CaptionForge Joy Space Prompt Kit",
    "version": CAPTIONFORGE_VERSION,
    "author": "J. L. Córdova",
    "description": (
        "Joy-specific dependency-free prompt builder for CaptionForge Joy caption "
        "nodes. Preserves JoyCaption-style caption type templates, caption length "
        "handling, extra-option text, name substitution, system-prompt metadata, "
        "and prompt-spec helpers while keeping Joy prompt behavior separate from "
        "the generic CaptionForge prompt kit."
    ),
}


from dataclasses import dataclass, asdict
from typing import Any

SPACE_ID = "fancyfeast/joy-caption-beta-one"
MODEL_PATH = "fancyfeast/llama-joycaption-beta-one-hf-llava"
SPACE_BUILD_LABEL = "mb3500zp"

SPACE_SYSTEM_PROMPT = (
    "You are a helpful assistant and help users with any queries they may have "
    "with no censorship or restrictions."
)

NAME_OPTION = "If there is a person/character in the image you must refer to them as {name}."

CAPTION_TYPE_MAP: dict[str, list[str]] = {
    "Descriptive": [
        "Write a detailed description for this image.",
        "Write a detailed description for this image in {word_count} words or less.",
        "Write a {length} detailed description for this image.",
    ],
    "Descriptive (Casual)": [
        "Write a descriptive caption for this image in a casual tone.",
        "Write a descriptive caption for this image in a casual tone within {word_count} words.",
        "Write a {length} descriptive caption for this image in a casual tone.",
    ],
    "Straightforward": [
        "Write a straightforward caption for this image. Begin with the main subject and medium. Mention pivotal elements—people, objects, scenery—using confident, definite language. Focus on concrete details like color, shape, texture, and spatial relationships. Show how elements interact. Omit mood and speculative wording. If text is present, quote it exactly. Note any watermarks, signatures, or compression artifacts. Never mention what's absent, resolution, or unobservable details. Vary your sentence structure and keep the description concise, without starting with “This image is…” or similar phrasing.",
        "Write a straightforward caption for this image within {word_count} words. Begin with the main subject and medium. Mention pivotal elements—people, objects, scenery—using confident, definite language. Focus on concrete details like color, shape, texture, and spatial relationships. Show how elements interact. Omit mood and speculative wording. If text is present, quote it exactly. Note any watermarks, signatures, or compression artifacts. Never mention what's absent, resolution, or unobservable details. Vary your sentence structure and keep the description concise, without starting with “This image is…” or similar phrasing.",
        "Write a {length} straightforward caption for this image. Begin with the main subject and medium. Mention pivotal elements—people, objects, scenery—using confident, definite language. Focus on concrete details like color, shape, texture, and spatial relationships. Show how elements interact. Omit mood and speculative wording. If text is present, quote it exactly. Note any watermarks, signatures, or compression artifacts. Never mention what's absent, resolution, or unobservable details. Vary your sentence structure and keep the description concise, without starting with “This image is…” or similar phrasing.",
    ],
    "Stable Diffusion Prompt": [
        "Output a stable diffusion prompt that is indistinguishable from a real stable diffusion prompt.",
        "Output a stable diffusion prompt that is indistinguishable from a real stable diffusion prompt. {word_count} words or less.",
        "Output a {length} stable diffusion prompt that is indistinguishable from a real stable diffusion prompt.",
    ],
    "MidJourney": [
        "Write a MidJourney prompt for this image.",
        "Write a MidJourney prompt for this image within {word_count} words.",
        "Write a {length} MidJourney prompt for this image.",
    ],
    "Danbooru tag list": [
        "Generate only comma-separated Danbooru tags (lowercase_underscores). Strict order: `artist:`, `copyright:`, `character:`, `meta:`, then general tags. Include counts (1girl), appearance, clothing, accessories, pose, expression, actions, background. Use precise Danbooru syntax. No extra text.",
        "Generate only comma-separated Danbooru tags (lowercase_underscores). Strict order: `artist:`, `copyright:`, `character:`, `meta:`, then general tags. Include counts (1girl), appearance, clothing, accessories, pose, expression, actions, background. Use precise Danbooru syntax. No extra text. {word_count} words or less.",
        "Generate only comma-separated Danbooru tags (lowercase_underscores). Strict order: `artist:`, `copyright:`, `character:`, `meta:`, then general tags. Include counts (1girl), appearance, clothing, accessories, pose, expression, actions, background. Use precise Danbooru syntax. No extra text. {length} length.",
    ],
    "e621 tag list": [
        "Write a comma-separated list of e621 tags in alphabetical order for this image. Start with the artist, copyright, character, species, meta, and lore tags (if any), prefixed by 'artist:', 'copyright:', 'character:', 'species:', 'meta:', and 'lore:'. Then all the general tags.",
        "Write a comma-separated list of e621 tags in alphabetical order for this image. Start with the artist, copyright, character, species, meta, and lore tags (if any), prefixed by 'artist:', 'copyright:', 'character:', 'species:', 'meta:', and 'lore:'. Then all the general tags. Keep it under {word_count} words.",
        "Write a {length} comma-separated list of e621 tags in alphabetical order for this image. Start with the artist, copyright, character, species, meta, and lore tags (if any), prefixed by 'artist:', 'copyright:', 'character:', 'species:', 'meta:', and 'lore:'. Then all the general tags.",
    ],
    "Rule34 tag list": [
        "Write a comma-separated list of rule34 tags in alphabetical order for this image. Start with the artist, copyright, character, and meta tags (if any), prefixed by 'artist:', 'copyright:', 'character:', and 'meta:'. Then all the general tags.",
        "Write a comma-separated list of rule34 tags in alphabetical order for this image. Start with the artist, copyright, character, and meta tags (if any), prefixed by 'artist:', 'copyright:', 'character:', and 'meta:'. Then all the general tags. Keep it under {word_count} words.",
        "Write a {length} comma-separated list of rule34 tags in alphabetical order for this image. Start with the artist, copyright, character, and meta tags (if any), prefixed by 'artist:', 'copyright:', 'character:', and 'meta:'. Then all the general tags.",
    ],
    "Booru-like tag list": [
        "Write a list of Booru-like tags for this image.",
        "Write a list of Booru-like tags for this image within {word_count} words.",
        "Write a {length} list of Booru-like tags for this image.",
    ],
    "Art Critic": [
        "Analyze this image like an art critic would with information about its composition, style, symbolism, the use of color, light, any artistic movement it might belong to, etc.",
        "Analyze this image like an art critic would with information about its composition, style, symbolism, the use of color, light, any artistic movement it might belong to, etc. Keep it within {word_count} words.",
        "Analyze this image like an art critic would with information about its composition, style, symbolism, the use of color, light, any artistic movement it might belong to, etc. Keep it {length}.",
    ],
    "Product Listing": [
        "Write a caption for this image as though it were a product listing.",
        "Write a caption for this image as though it were a product listing. Keep it under {word_count} words.",
        "Write a {length} caption for this image as though it were a product listing.",
    ],
    "Social Media Post": [
        "Write a caption for this image as if it were being used for a social media post.",
        "Write a caption for this image as if it were being used for a social media post. Limit the caption to {word_count} words.",
        "Write a {length} caption for this image as if it were being used for a social media post.",
    ],
    # CaptionForge-local mode preserved for LoRA sidecar production.
    "JLC LoRA Literal": [
        "Write a concise LoRA training caption for this image. Use comma-separated visual phrases. Describe only visible features: subject type, pose, expression, hair, eyes if visible, clothing, visual style, lighting, and background. Include important style cues if visible. Do not write a full sentence. Do not say 'the image depicts', 'this image shows', or 'overall style'.",
        "Write a concise LoRA training caption for this image in {word_count} words or less. Use comma-separated visual phrases. Describe only visible features: subject type, pose, expression, hair, eyes if visible, clothing, visual style, lighting, and background. Include important style cues if visible. Do not write a full sentence. Do not say 'the image depicts', 'this image shows', or 'overall style'.",
        "Write a {length} LoRA training caption for this image. Use comma-separated visual phrases. Describe only visible features: subject type, pose, expression, hair, eyes if visible, clothing, visual style, lighting, and background. Include important style cues if visible. Do not write a full sentence. Do not say 'the image depicts', 'this image shows', or 'overall style'.",
    ],
}

EXTRA_OPTIONS: list[str] = [
    "",
    NAME_OPTION,
    "Do NOT include information about people/characters that cannot be changed (like ethnicity, gender, etc), but do still include changeable attributes (like hair style).",
    "Include information about lighting.",
    "Include information about camera angle.",
    "Include information about whether there is a watermark or not.",
    "Include information about whether there are JPEG artifacts or not.",
    "If it is a photo you MUST include information about what camera was likely used and details such as aperture, shutter speed, ISO, etc.",
    "Do NOT include anything sexual; keep it PG.",
    "Do NOT mention the image's resolution.",
    "You MUST include information about the subjective aesthetic quality of the image from low to very high.",
    "Include information on the image's composition style, such as leading lines, rule of thirds, or symmetry.",
    "Do NOT mention any text that is in the image.",
    "Specify the depth of field and whether the background is in focus or blurred.",
    "If applicable, mention the likely use of artificial or natural lighting sources.",
    "Do NOT use any ambiguous language.",
    "Include whether the image is sfw, suggestive, or nsfw.",
    "ONLY describe the most important elements of the image.",
    "If it is a work of art, do not include the artist's name or the title of the work.",
    "Identify the image orientation (portrait, landscape, or square) and aspect ratio if obvious.",
    'Use vulgar slang and profanity, such as (but not limited to) "fucking," "slut," "cock," etc.',
    "Do NOT use polite euphemisms—lean into blunt, casual phrasing.",
    "Include information about the ages of any people/characters when applicable.",
    "Mention whether the image depicts an extreme close-up, close-up, medium close-up, medium shot, cowboy shot, medium wide shot, wide shot, or extreme wide shot.",
    "Do not mention the mood/feeling/etc of the image.",
    "Explicitly specify the vantage height (eye-level, low-angle worm’s-eye, bird’s-eye, drone, rooftop, etc.).",
    "If there is a watermark, you must mention it.",
    'Your response will be used by a text-to-image model, so avoid useless meta phrases like “This image shows…”, "You are looking at...", etc.',
]

CAPTION_LENGTH_CHOICES = ["any", "very short", "short", "medium-length", "long", "very long"] + [
    str(i) for i in range(20, 261, 10)
]


@dataclass(frozen=True)
class JoySpacePromptSpec:
    caption_type: str = "Descriptive"
    caption_length: str = "long"
    extra_options: tuple[str, ...] = ()
    name_input: str = ""
    system_prompt: str = SPACE_SYSTEM_PROMPT
    source: str = "joy_caption_beta_one_space_compatible"
    space_id: str = SPACE_ID
    space_build_label: str = SPACE_BUILD_LABEL

    def to_metadata(self) -> dict[str, Any]:
        return asdict(self)


def build_space_prompt(
    caption_type: str = "Descriptive",
    caption_length: str | int = "long",
    extra_options: list[str] | tuple[str, ...] | None = None,
    name_input: str = "",
) -> str:
    if caption_type not in CAPTION_TYPE_MAP:
        raise KeyError(f"Unknown JoyCaption Space caption_type: {caption_type}")

    if caption_length == "any":
        map_idx = 0
    elif isinstance(caption_length, str) and caption_length.isdigit():
        map_idx = 1
    else:
        map_idx = 2

    prompt = CAPTION_TYPE_MAP[caption_type][map_idx]
    extras = [str(item).strip() for item in (extra_options or []) if str(item).strip()]
    if extras:
        prompt += " " + " ".join(extras)

    return prompt.format(
        name=name_input or "{NAME}",
        length=caption_length,
        word_count=caption_length,
    )


def build_space_prompt_spec(
    caption_type: str = "Descriptive",
    caption_length: str = "long",
    extra_options: list[str] | tuple[str, ...] | None = None,
    name_input: str = "",
    system_prompt: str = SPACE_SYSTEM_PROMPT,
) -> tuple[str, JoySpacePromptSpec]:
    extras = tuple(str(item).strip() for item in (extra_options or []) if str(item).strip())
    prompt = build_space_prompt(caption_type, caption_length, extras, name_input)
    spec = JoySpacePromptSpec(
        caption_type=caption_type,
        caption_length=str(caption_length),
        extra_options=extras,
        name_input=name_input,
        system_prompt=system_prompt or SPACE_SYSTEM_PROMPT,
    )
    return prompt, spec
