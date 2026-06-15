#!/usr/bin/env python
"""
JLC Joy Caption Engine

- CaptionForge
  - This module is part of **CaptionForge**, a model-agnostic captioning
    framework for ComfyUI developed by **J. L. Córdova**.

  - Repository
    https://github.com/Damkohler/CaptionForge

  - CaptionForge focuses on practical dataset-captioning infrastructure for:
        • LoRA dataset preparation
        • multi-engine caption generation
        • JSONL audit trails
        • claim extraction and refinement
        • consensus-oriented caption improvement

- Module Purpose
    - The **JLC Joy Caption Engine** provides the shared importable backend for
      JoyCaption-family image captioning inside CaptionForge.

    - It centralizes Joy/LLaVA-specific captioning behavior so ComfyUI node
      wrappers can remain thin and focused on user interface concerns.

    - The engine handles:
            • JoyCaption model registry lookup
            • Hugging Face download/probe behavior
            • local model-path resolution
            • processor and model loading
            • CaptionForge shared model-cache integration
            • 8-bit memory-efficient loading
            • prompt preset and Joy-native prompt-template resolution
            • image traversal and filtering
            • image resizing before inference
            • single-image and folder captioning
            • caption cleanup and replacement rules
            • TXT sidecar writing
            • JSONL audit writing
            • run-configuration JSON writing
            • Pass A CaptionForge audit record creation
            • error record streaming during batch runs

- CaptionForge Pass Role
    - This engine participates in **Pass A** of the CaptionForge pipeline.

    - Pass A generates caption evidence records from one or more captioning
      engines. These records can later be consumed by Pass B claim extraction
      and future consensus/refinement passes.

    - Joy output records include CaptionForge audit fields such as:
            • captionforge_pass
            • model_family
            • ensemble_run_index
            • image_key
            • raw_caption
            • final cleaned caption
            • generation parameters
            • model and prompt metadata

- Model and Memory Behavior
    - JoyCaption is handled separately from Qwen and other engines because it
      uses Joy/LLaVA-specific loading, processor behavior, prompt structure, and
      memory-management assumptions.

    - The engine integrates with `captionforge_model_cache.py` so that
      heavyweight captioning models can be reused or evicted under a shared
      CaptionForge residency policy.

    - Release behavior requires CUDA by default. Silent CPU fallback is disabled
      to avoid accidental unusable performance or hidden execution changes.
      CPU/debug experiments require deliberate source edits.

    - Supported memory modes currently include:
            • Default
            • Balanced (8-bit)

- Prompt and Cleanup Strategy
    - The engine supports both CaptionForge-oriented literal caption prompts and
      Joy-native prompt templates.

    - Caption cleanup utilities support:
            • boilerplate-prefix stripping
            • trailing-period stripping
            • forbidden phrase removal
            • replacement rules
            • trigger insertion
            • prefix/suffix insertion

    - This keeps raw model output available while also producing practical
      LoRA-ready sidecar captions.

- Design Philosophy
    - This engine preserves JoyCaption as an independent captioning voice inside
      CaptionForge rather than treating it as secondary to any other model.

    - CaptionForge is engine-democratic: Joy captions contribute evidence to the
      broader audit and consensus pipeline alongside Qwen, future local VLMs,
      cleanup LLMs, validators, or other robustness engines.

    - The engine therefore prioritizes reproducibility, auditability, and clean
      separation between model-specific inference code and ComfyUI node wrappers.

- ⚠️ Development Status
    - This is early CaptionForge Pass A engine infrastructure.
    - Registry entries are intentionally conservative and should be expanded only
      after processor/model compatibility is tested.
    - Memory behavior, prompt presets, and audit fields may evolve as the
      multi-pass CaptionForge pipeline matures.

- Attribution & License
  - Concept and implementation by **J. L. Córdova**
    with development assistance from **ChatGPT (OpenAI)**.

  - JoyCaption/LLaVA model loading is designed around compatible Hugging Face
    Transformers interfaces and publicly available JoyCaption-family checkpoints.

  - Copyright (c) 2026 J. L. Córdova

  - Released under the **MIT License**.
"""

from __future__ import annotations

MANIFEST = {
    "name": "JLC Joy Caption Engine",
    "version": (1, 1, 0),
    "author": "J. L. Córdova",
    "description": (
        "Shared JoyCaption-family CaptionForge Pass A engine for local image captioning, "
        "folder traversal, prompt resolution, caption cleanup, TXT sidecar writing, JSONL "
        "audit streaming, run-config export, model registry lookup, Hugging Face download "
        "probe support, CUDA-only release inference, 8-bit memory-efficient loading, and "
        "integration with the CaptionForge global model cache. Designed to keep Joy/LLaVA "
        "captioning logic separate from ComfyUI node wrappers while contributing auditable "
        "caption evidence to model-agnostic downstream claim extraction and consensus "
        "refinement passes."
    ),
}


import fnmatch
import json
import logging
import random
import re
import shutil
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable, Optional
import warnings

from PIL import Image
import torch

from .captionforge_model_cache import (
    make_cache_key,
    get_cached_model,
    register_model,
    prepare_for_model_load,
    unload_after_run,
)


# -------------------------------------------------------------------------
# Eliminate noise from transformers: UserWarning:
# MatMul8bitLt: inputs will be cast from torch.float32 to float16 during quantization
# -------------------------------------------------------------------------

warnings.filterwarnings(
    "ignore",
    message=r".*MatMul8bitLt: inputs will be cast.*",
    category=UserWarning,
)

warnings.filterwarnings(
    "ignore",
    message=r".*torchvision backend image processor with LANCZOS resample.*",
)


# -------------------------------------------------------------------------
# Model registry
# -------------------------------------------------------------------------
# Registry entries are intentionally conservative. Add variants only after
# testing their processor/model class compatibility.
# -------------------------------------------------------------------------

@dataclass(frozen=True)
class JoyModelInfo:
    repo_id: str
    local_folder: str
    notes: str = ""


MODEL_REGISTRY: dict[str, JoyModelInfo] = {
    "llama-joycaption-beta-one-hf-llava": JoyModelInfo(
        repo_id="fancyfeast/llama-joycaption-beta-one-hf-llava",
        local_folder="llama-joycaption-beta-one-hf-llava",
        notes="Primary JoyCaption beta-one HF/LLaVA checkpoint.",
    ),
}


SUPPORTED_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".webp",
    ".bmp",
    ".tif",
    ".tiff",
}


MEMORY_EFFICIENT_CONFIGS: dict[str, dict[str, Any]] = {
    "Balanced (8-bit)": {
        "load_in_8bit": True,
    },
    "Default": {},
}


DEFAULT_SYSTEM_PROMPT = (
    "You are a helpful image-captioning assistant. Describe only what is visible in the image."
)

DEFAULT_PROMPT = (
    "Describe this image in a highly detailed, literal, visually grounded way. "
    "Focus only on visible content. Include subject appearance, clothing, pose, "
    "body position, hands, facial expression, hairstyle, accessories, lighting, "
    "background, textures, colors, and spatial relationships. Avoid speculation, "
    "opinions, or anything not clearly visible. Write one dense descriptive caption "
    "suitable for image dataset captioning."
)


# Joy-native prompt templates adapted from common JoyCaption node behavior.
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
        "Write a straightforward caption for this image. Begin with the main subject and medium. Mention pivotal elements—people, objects, scenery—using confident, definite language. Focus on concrete details like color, shape, texture, and spatial relationships. Show how elements interact. Omit mood and speculative wording. If text is present, quote it exactly. Note any watermarks, signatures, or compression artifacts. Never mention what's absent, resolution, or unobservable details. Vary your sentence structure and keep the description concise, without starting with ‘This image is…’ or similar phrasing.",
        "Write a straightforward caption for this image within {word_count} words. Begin with the main subject and medium. Mention pivotal elements—people, objects, scenery—using confident, definite language. Focus on concrete details like color, shape, texture, and spatial relationships. Show how elements interact. Omit mood and speculative wording. If text is present, quote it exactly. Note any watermarks, signatures, or compression artifacts. Never mention what's absent, resolution, or unobservable details. Vary your sentence structure and keep the description concise, without starting with ‘This image is…’ or similar phrasing.",
        "Write a {length} straightforward caption for this image. Begin with the main subject and medium. Mention pivotal elements—people, objects, scenery—using confident, definite language. Focus on concrete details like color, shape, texture, and spatial relationships. Show how elements interact. Omit mood and speculative wording. If text is present, quote it exactly. Note any watermarks, signatures, or compression artifacts. Never mention what's absent, resolution, or unobservable details. Vary your sentence structure and keep the description concise, without starting with ‘This image is…’ or similar phrasing.",
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
    "Booru-like tag list": [
        "Write a list of Booru-like tags for this image.",
        "Write a list of Booru-like tags for this image within {word_count} words.",
        "Write a {length} list of Booru-like tags for this image.",
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
    "JLC LoRA Literal": [
        "Write a concise LoRA training caption for this image. Use comma-separated visual phrases. Describe only visible features: subject type, pose, expression, hair, eyes if visible, clothing, visual style, lighting, and background. Include important style cues if visible. Do not write a full sentence. Do not say 'the image depicts', 'this image shows', or 'overall style'.",
        "Write a concise LoRA training caption for this image in {word_count} words or less. Use comma-separated visual phrases. Describe only visible features: subject type, pose, expression, hair, eyes if visible, clothing, visual style, lighting, and background. Include important style cues if visible. Do not write a full sentence. Do not say 'the image depicts', 'this image shows', or 'overall style'.",
        "Write a {length} LoRA training caption for this image. Use comma-separated visual phrases. Describe only visible features: subject type, pose, expression, hair, eyes if visible, clothing, visual style, lighting, and background. Include important style cues if visible. Do not write a full sentence. Do not say 'the image depicts', 'this image shows', or 'overall style'.",
    ],
}

PROMPT_PRESETS: dict[str, str] = {
    "default_literal": DEFAULT_PROMPT,
    "lora": CAPTION_TYPE_MAP["JLC LoRA Literal"][0],
    "straightforward": CAPTION_TYPE_MAP["Straightforward"][0],
    "descriptive": CAPTION_TYPE_MAP["Descriptive"][0],
    "booru_like": CAPTION_TYPE_MAP["Booru-like tag list"][0],
    "stable_diffusion_prompt": CAPTION_TYPE_MAP["Stable Diffusion Prompt"][0],
}

EXTRA_OPTIONS = [
    "",
    "If there is a person/character in the image you must refer to them as {name}.",
    "Do NOT include information about people/characters that cannot be changed, but do still include changeable attributes like hairstyle, clothing, pose, expression, and accessories.",
    "Include information about lighting.",
    "Include information about camera angle.",
    "Include information about whether there is a watermark or not.",
    "Do NOT include anything sexual; keep it PG.",
    "Do NOT mention the image's resolution.",
    "Include information on the image's composition style, such as leading lines, rule of thirds, or symmetry.",
    "Do NOT mention any text that is in the image.",
    "Do NOT use any ambiguous language.",
    "ONLY describe the most important elements of the image.",
    "Mention whether the image depicts an extreme close-up, close-up, medium close-up, medium shot, cowboy shot, medium wide shot, wide shot, or extreme wide shot.",
    "Do not mention the mood or feeling of the image.",
    "Your response will be used by a text-to-image model, so avoid useless meta phrases like 'This image shows…' or 'You are looking at...'.",
]

CAPTION_LENGTH_CHOICES = ["any", "very short", "short", "medium-length", "long", "very long"] + [
    str(i) for i in range(20, 261, 10)
]

# -------------------------------------------------------------------------
# JoyCaption Beta One Space-compatible prompt harness
# -------------------------------------------------------------------------
# Keep the older local map above as historical reference, then override the
# runtime prompt map with the public HF Space-compatible helper. This keeps
# existing imports stable for node wrappers while moving the Space prompt
# builder into a small dedicated module.
try:
    from .captionforge_joy_space_prompt_kit import (
        CAPTION_TYPE_MAP as SPACE_CAPTION_TYPE_MAP,
        CAPTION_LENGTH_CHOICES as SPACE_CAPTION_LENGTH_CHOICES,
        EXTRA_OPTIONS as SPACE_EXTRA_OPTIONS,
        NAME_OPTION,
        SPACE_ID,
        SPACE_BUILD_LABEL,
        SPACE_SYSTEM_PROMPT,
        build_space_prompt,
        build_space_prompt_spec,
    )

    CAPTION_TYPE_MAP = dict(SPACE_CAPTION_TYPE_MAP)
    CAPTION_LENGTH_CHOICES = list(SPACE_CAPTION_LENGTH_CHOICES)
    EXTRA_OPTIONS = list(SPACE_EXTRA_OPTIONS)
    DEFAULT_SPACE_SYSTEM_PROMPT = SPACE_SYSTEM_PROMPT

    PROMPT_PRESETS.update(
        {
            "joy_space_descriptive_long": build_space_prompt("Descriptive", "long", [], ""),
            "joy_space_straightforward_long": build_space_prompt("Straightforward", "long", [], ""),
        }
    )

except Exception:
    NAME_OPTION = "If there is a person/character in the image you must refer to them as {name}."
    SPACE_ID = ""
    SPACE_BUILD_LABEL = ""
    DEFAULT_SPACE_SYSTEM_PROMPT = DEFAULT_SYSTEM_PROMPT

    def build_space_prompt(caption_type="Descriptive", caption_length="long", extra_options=None, name_input=""):
        return build_joy_prompt(caption_type, caption_length, extra_options or [], name_input)

    def build_space_prompt_spec(caption_type="Descriptive", caption_length="long", extra_options=None, name_input="", system_prompt=DEFAULT_SYSTEM_PROMPT):
        return build_space_prompt(caption_type, caption_length, extra_options or [], name_input), None



# -------------------------------------------------------------------------
# Dataclasses
# -------------------------------------------------------------------------

@dataclass
class GenerationConfig:
    max_new_tokens: int = 512
    temperature: float = 0.60
    top_p: float = 0.90
    top_k: int = 0
    repetition_penalty: float = 1.0
    seed: Optional[int] = None

    @property
    def do_sample(self) -> bool:
        return self.temperature > 0


@dataclass
class CleanupConfig:
    trigger: str = ""
    prefix: str = ""
    suffix: str = ""
    forbidden_phrases: list[str] = field(default_factory=list)
    replacement_rules: list[tuple[str, str]] = field(default_factory=list)
    replace_case_insensitive: bool = True
    replace_whole_words_only: bool = False
    strip_boilerplate_prefixes: bool = True
    strip_trailing_period: bool = True


def _unload_joy_bundle(bundle: dict[str, Any]) -> None:
    model = bundle.get("model") if isinstance(bundle, dict) else None

    if model is not None:
        try:
            model.to("cpu")
        except Exception:
            pass

        try:
            del model
        except Exception:
            pass

    if isinstance(bundle, dict):
        bundle.clear()


@dataclass
class JoyCaptionConfig:
    # Either model_name or model_path may be used. If model_path is supplied, it wins.
    model_name: str = "llama-joycaption-beta-one-hf-llava"
    model_path: str = ""

    # Root used only for registry-managed model folders.
    model_root: str = "models/LLM/JLC_JoyCaption"

    # Loading behavior.
    memory_mode: str = "Balanced (8-bit)"
    dtype: str = "bf16"               # auto, bf16, bfloat16, fp16, float16, fp32, float32
    device: str = "auto"              # auto, cuda, cpu, cuda:0, etc.
    trust_remote_code: bool = True
    keep_loaded: bool = True
    cache_policy: str = ""             # "", evict_other_caption_models, keep_this_model, unload_after_run
    quiet_transformers_load: bool = True
    apply_liger_kernel: bool = False
    space_compatible_mode: bool = False

    # Image and prompt behavior.
    max_size: int = 1024
    system_prompt: str = DEFAULT_SYSTEM_PROMPT
    prompt: str = DEFAULT_PROMPT

    # Download/probe behavior for registry models.
    allow_download: bool = True

    # If True and running inside ComfyUI, use comfy.model_management for device/offload.
    use_comfy_model_management: bool = True


@dataclass
class BatchCaptionConfig:
    input_path: str = ""
    recursive: bool = True
    filename_glob: str = "*"
    extensions: set[str] = field(default_factory=lambda: set(SUPPORTED_EXTENSIONS))
    output_dir: str = ""
    write_txt: bool = True
    write_jsonl: bool = False
    jsonl_filename: str = "captions.jsonl"
    also_jsonl_path: str = ""
    write_run_config: bool = True
    run_config_filename: str = ""
    overwrite: bool = False
    backup_existing: bool = True
    dry_run: bool = False
    limit: int = 0
    skip_existing_txt: bool = True
    skip_existing_jsonl_images: bool = False


@dataclass
class CaptionRecord:
    image: str
    caption: str
    raw_caption: str
    model_name: str
    model_path: str
    prompt: str
    system_prompt: str
    seed: Optional[int]
    temperature: float
    top_p: float
    top_k: int
    max_new_tokens: int
    max_size: int
    timestamp: str
    status: str = "ok"
    error: str = ""
    # CaptionForge audit fields.
    captionforge_pass: str = "A"
    model_family: str = "joy"
    ensemble_run_index: int = 0
    image_key: str = ""


@dataclass
class BatchCaptionResult:
    records: list[CaptionRecord] = field(default_factory=list)
    skipped: int = 0
    failed: int = 0

    @property
    def processed(self) -> int:
        return len([r for r in self.records if r.status == "ok"])

    @property
    def captions_text(self) -> str:
        return "\n\n".join(r.caption for r in self.records if r.status == "ok")

    @property
    def jsonl_text(self) -> str:
        return "\n".join(json.dumps(record_to_json(r), ensure_ascii=False) for r in self.records)


# -------------------------------------------------------------------------
# Small helpers
# -------------------------------------------------------------------------

def timestamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def iso_timestamp() -> str:
    return datetime.now().isoformat(timespec="seconds")


def safe_mkdir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def parse_extensions(value: str) -> set[str]:
    if not value.strip():
        return set(SUPPORTED_EXTENSIONS)

    exts: set[str] = set()
    for item in value.split(","):
        item = item.strip().lower()
        if not item:
            continue
        if not item.startswith("."):
            item = "." + item
        exts.add(item)
    return exts


def parse_replacement_rule(value: str) -> tuple[str, str]:
    if "=>" in value:
        old, new = value.split("=>", 1)
    elif ";" in value:
        old, new = value.split(";", 1)
    else:
        raise ValueError(f"Replacement rule must use old;new or old=>new format: {value}")
    return old.strip(), new.strip()


def parse_replacement_rules_text(value: str) -> list[tuple[str, str]]:
    rules: list[tuple[str, str]] = []
    for line in value.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        rules.append(parse_replacement_rule(line))
    return rules


def load_prompt_file(path: str | Path) -> str:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Prompt file does not exist: {p}")
    return p.read_text(encoding="utf-8").strip()


def build_joy_prompt(
    caption_type: str,
    caption_length: str | int,
    extra_options: list[str],
    name_input: str = "",
) -> str:
    if caption_type not in CAPTION_TYPE_MAP:
        raise KeyError(f"Unknown caption_type: {caption_type}")

    if caption_length == "any":
        map_idx = 0
    elif isinstance(caption_length, str) and caption_length.isdigit():
        map_idx = 1
    else:
        map_idx = 2

    prompt = CAPTION_TYPE_MAP[caption_type][map_idx]

    extras = [item.strip() for item in extra_options if item and item.strip()]
    if extras:
        prompt += " " + " ".join(extras)

    return prompt.format(
        name=name_input or "{NAME}",
        length=caption_length,
        word_count=caption_length,
    )


def resolve_prompt(
    prompt: str = "",
    prompt_file: str = "",
    prompt_preset: str = "default_literal",
    caption_type: str = "",
    caption_length: str = "any",
    extra_options: Optional[list[str]] = None,
    name_input: str = "",
) -> str:
    prompt = prompt.strip() if prompt else ""
    prompt_file = prompt_file.strip() if prompt_file else ""

    if prompt:
        return prompt

    if prompt_file:
        text = load_prompt_file(prompt_file)
        if text:
            return text

    if caption_type:
        return build_joy_prompt(
            caption_type=caption_type,
            caption_length=caption_length,
            extra_options=extra_options or [],
            name_input=name_input,
        )

    return PROMPT_PRESETS.get(prompt_preset, DEFAULT_PROMPT)


# -------------------------------------------------------------------------
# Model path / download helpers
# -------------------------------------------------------------------------

def model_folder_has_weights(local_path: Path) -> bool:
    if not local_path.exists() or not local_path.is_dir():
        return False

    weight_patterns = [
        "*.safetensors",
        "*.bin",
        "*.pt",
        "*.pth",
        "*.gguf",
        "*.ckpt",
    ]

    return any(any(local_path.rglob(pattern)) for pattern in weight_patterns)


def get_registry_model_path(model_name: str, model_root: str | Path) -> Path:
    if model_name not in MODEL_REGISTRY:
        raise KeyError(
            f"Unknown Joy model_name: {model_name}. "
            f"Known models: {', '.join(MODEL_REGISTRY.keys())}"
        )
    return Path(model_root) / MODEL_REGISTRY[model_name].local_folder


def download_registry_model_if_needed(
    model_name: str,
    model_root: str | Path,
    metadata_only: bool = False,
    allow_download: bool = True,
) -> Path:
    local_path = get_registry_model_path(model_name, model_root)

    if not metadata_only and model_folder_has_weights(local_path):
        return local_path

    if not allow_download:
        if metadata_only:
            safe_mkdir(local_path)
            return local_path
        raise FileNotFoundError(
            f"Model folder does not contain weights and allow_download=False: {local_path}"
        )

    try:
        from huggingface_hub import snapshot_download
    except Exception as exc:
        raise RuntimeError(
            "Missing dependency: huggingface_hub. Install it in the active venv:\n"
            "  pip install huggingface_hub"
        ) from exc

    info = MODEL_REGISTRY[model_name]
    safe_mkdir(Path(model_root))
    safe_mkdir(local_path)

    if metadata_only:
        print(
            f"[JLC Joy Engine] Probe download for {info.repo_id} -> {local_path}\n"
            "[JLC Joy Engine] Large weight files will be skipped."
        )
        snapshot_download(
            repo_id=info.repo_id,
            local_dir=str(local_path),
            ignore_patterns=[
                "*.safetensors",
                "*.bin",
                "*.pt",
                "*.pth",
                "*.gguf",
                "*.onnx",
                "*.ckpt",
                "*.h5",
                "*.msgpack",
                "*.tflite",
            ],
        )
        return local_path

    print(f"[JLC Joy Engine] Downloading {info.repo_id} -> {local_path}")
    snapshot_download(
        repo_id=info.repo_id,
        local_dir=str(local_path),
    )
    return local_path


def probe_registry_model_download(model_name: str, model_root: str | Path) -> str:
    local_path = download_registry_model_if_needed(
        model_name=model_name,
        model_root=model_root,
        metadata_only=True,
        allow_download=True,
    )

    files: list[str] = []
    try:
        for p in sorted(local_path.rglob("*")):
            if p.is_file():
                files.append(str(p.relative_to(local_path)))
    except Exception:
        files = []

    preview = "\n".join(files[:40])
    if len(files) > 40:
        preview += f"\n... plus {len(files) - 40} more files"

    return (
        "JLC Joy Caption download probe completed.\n\n"
        f"Model: {model_name}\n"
        f"Folder: {local_path}\n\n"
        "Large model weight files were intentionally skipped.\n"
        "This folder is not a complete usable model unless full weights are later downloaded or copied in.\n\n"
        f"Files found:\n{preview if preview else '(no files listed)'}"
    )


# -------------------------------------------------------------------------
# Caption cleanup
# -------------------------------------------------------------------------

UNWANTED_PREFIXES = [
    "The image depicts ",
    "The image shows ",
    "This image depicts ",
    "This image shows ",
    "An image of ",
    "A photo of ",
    "A photograph of ",
    "The photo shows ",
    "This photo shows ",
]


def normalize_caption(
    caption: str,
    strip_boilerplate_prefixes: bool = True,
    strip_trailing_period: bool = True,
) -> str:
    text = caption.strip()

    if strip_boilerplate_prefixes:
        for prefix in UNWANTED_PREFIXES:
            if text.lower().startswith(prefix.lower()):
                text = text[len(prefix):].strip()
                break

    text = text.replace("\n", " ")
    text = re.sub(r"\s+", " ", text).strip()
    text = re.sub(r"\s+,", ",", text)
    text = re.sub(r",\s*,+", ",", text)
    text = re.sub(r"\s+\.", ".", text)

    if strip_trailing_period and text.endswith("."):
        text = text[:-1].strip()

    return text


def apply_replacements(
    caption: str,
    rules: list[tuple[str, str]],
    case_insensitive: bool = True,
    whole_words_only: bool = False,
) -> str:
    if not rules:
        return caption

    result = caption

    for old, new in rules:
        old = old.strip()
        new = new.strip()

        if not old:
            continue

        flags = re.IGNORECASE if case_insensitive else 0
        pattern = re.escape(old)

        if whole_words_only:
            pattern = r"\b" + pattern + r"\b"

        result = re.sub(pattern, new, result, flags=flags)

    return result


def remove_forbidden_phrases(caption: str, forbidden_phrases: list[str]) -> str:
    if not forbidden_phrases:
        return caption

    result = caption

    for phrase in forbidden_phrases:
        phrase = phrase.strip()
        if not phrase:
            continue
        result = re.sub(re.escape(phrase), "", result, flags=re.IGNORECASE)

    result = re.sub(r"\s+,", ",", result)
    result = re.sub(r",\s*,+", ",", result)
    result = re.sub(r"\s+", " ", result)
    return result.strip(" ,")


def add_trigger_prefix_suffix(
    caption: str,
    trigger: str = "",
    prefix: str = "",
    suffix: str = "",
) -> str:
    parts: list[str] = []

    trigger = trigger.strip()
    prefix = prefix.strip()
    suffix = suffix.strip()

    if trigger:
        parts.append(trigger.strip(" ,"))

    if prefix:
        parts.append(prefix.strip(" ,"))

    if caption:
        parts.append(caption.lstrip(" ,"))

    final = ", ".join(part for part in parts if part)
    final = re.sub(r",\s*,+", ",", final).strip(" ,")

    if suffix:
        final = final + suffix

    return final.strip()


def cleanup_caption(caption: str, config: CleanupConfig) -> str:
    text = normalize_caption(
        caption,
        strip_boilerplate_prefixes=config.strip_boilerplate_prefixes,
        strip_trailing_period=config.strip_trailing_period,
    )

    text = apply_replacements(
        caption=text,
        rules=config.replacement_rules,
        case_insensitive=config.replace_case_insensitive,
        whole_words_only=config.replace_whole_words_only,
    )

    text = remove_forbidden_phrases(text, config.forbidden_phrases)

    text = normalize_caption(
        text,
        strip_boilerplate_prefixes=config.strip_boilerplate_prefixes,
        strip_trailing_period=config.strip_trailing_period,
    )

    return add_trigger_prefix_suffix(
        caption=text,
        trigger=config.trigger,
        prefix=config.prefix,
        suffix=config.suffix,
    )


# -------------------------------------------------------------------------
# Image traversal and output helpers
# -------------------------------------------------------------------------

def iter_image_files(
    input_path: str | Path,
    recursive: bool = True,
    filename_glob: str = "*",
    extensions: Optional[set[str]] = None,
) -> Iterable[Path]:
    p = Path(input_path)
    exts = extensions or set(SUPPORTED_EXTENSIONS)

    filename_glob = (filename_glob or "*").strip() or "*"

    if p.is_file():
        if p.suffix.lower() in exts and fnmatch.fnmatch(p.name, filename_glob):
            yield p
        return

    if not p.exists():
        raise FileNotFoundError(f"input_path does not exist: {p}")

    if not p.is_dir():
        raise NotADirectoryError(f"input_path is not a file or folder: {p}")

    iterator = p.rglob("*") if recursive else p.glob("*")

    for child in sorted(iterator):
        if not child.is_file():
            continue
        if child.suffix.lower() not in exts:
            continue
        if not fnmatch.fnmatch(child.name, filename_glob):
            continue
        yield child


def load_image_file(path: str | Path) -> Image.Image:
    return Image.open(path).convert("RGB")


def resize_for_model(image: Image.Image, max_size: int) -> Image.Image:
    if max_size <= 0:
        return image

    width, height = image.size
    longest = max(width, height)

    if longest <= max_size:
        return image

    scale = max_size / float(longest)
    new_width = max(1, int(round(width * scale)))
    new_height = max(1, int(round(height * scale)))

    return image.resize((new_width, new_height), Image.Resampling.LANCZOS)


def sidecar_txt_path(image_path: Path, output_dir: str | Path | None = None) -> Path:
    if output_dir:
        out = Path(output_dir)
        safe_mkdir(out)
        return out / f"{image_path.stem}.txt"
    return image_path.with_suffix(".txt")


def backup_existing_file(path: Path, dry_run: bool = False) -> Optional[Path]:
    if not path.exists():
        return None

    backup_path = path.with_name(f"{path.name}.bak_{timestamp()}")
    if not dry_run:
        shutil.copy2(path, backup_path)
    return backup_path


def write_text_sidecar(
    path: Path,
    text: str,
    overwrite: bool = False,
    backup_existing: bool = True,
    dry_run: bool = False,
) -> bool:
    if path.exists() and not overwrite:
        return False

    if dry_run:
        return True

    safe_mkdir(path.parent)

    if path.exists() and overwrite and backup_existing:
        backup_existing_file(path, dry_run=False)

    path.write_text(text.rstrip() + "\n", encoding="utf-8")
    return True


def load_existing_jsonl_images(path: Path) -> set[str]:
    seen: set[str] = set()

    if not path.exists():
        return seen

    with path.open("r", encoding="utf-8") as f:
        for line in f:
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue

            image_value = record.get("image") or record.get("image_path") or record.get("source")
            if image_value:
                seen.add(str(image_value))
                try:
                    seen.add(str(Path(image_value)))
                except Exception:
                    pass

    return seen


def record_to_json(record: CaptionRecord) -> dict[str, Any]:
    return asdict(record)


def append_jsonl_records(path: Path, records: list[CaptionRecord], dry_run: bool = False) -> None:
    if dry_run or not records:
        return

    safe_mkdir(path.parent)

    with path.open("a", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record_to_json(record), ensure_ascii=False) + "\n")


def json_safe(value):
    if isinstance(value, set):
        return sorted(value)

    if isinstance(value, Path):
        return str(value)

    if isinstance(value, dict):
        return {k: json_safe(v) for k, v in value.items()}

    if isinstance(value, list):
        return [json_safe(v) for v in value]

    if isinstance(value, tuple):
        return [json_safe(v) for v in value]

    return value


def write_run_config_json(path: Path, config: dict[str, Any], dry_run: bool = False) -> None:
    if dry_run:
        return
    safe_mkdir(path.parent)
    path.write_text(json.dumps(json_safe(config), indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


# -------------------------------------------------------------------------
# Optional ComfyUI memory-management bridge
# -------------------------------------------------------------------------

def _try_get_comfy_model_management():
    try:
        import comfy.model_management as model_management
        return model_management
    except Exception:
        return None


def _cuda_device_map(dev: torch.device):
    if dev.type == "cuda":
        return {"": (dev.index or 0)}
    return {"": str(dev)}


# -------------------------------------------------------------------------
# Engine
# -------------------------------------------------------------------------

class JoyCaptionEngine:
    def __init__(
        self,
        config: JoyCaptionConfig,
        generation: Optional[GenerationConfig] = None,
        cleanup: Optional[CleanupConfig] = None,
    ) -> None:
        self.config = config
        self.generation = generation or GenerationConfig()
        self.cleanup = cleanup or CleanupConfig()
        self.processor = None
        self.model = None
        self.local_model_path: Optional[Path] = None
        self.model_size_bytes: Optional[int] = None
        self._comfy_mm = _try_get_comfy_model_management() if config.use_comfy_model_management else None
        self.inference_device = self._resolve_inference_device(config.device)
        self.offload_device = self._resolve_offload_device(config.device)
        self.is_kbit = self.config.memory_mode != "Default"

    def resolve_model_path(self) -> Path:
        if self.config.model_path.strip():
            return Path(self.config.model_path).expanduser()

        return download_registry_model_if_needed(
            model_name=self.config.model_name,
            model_root=self.config.model_root,
            metadata_only=False,
            allow_download=self.config.allow_download,
        )

    def _cache_policy(self) -> str:
        policy = getattr(self.config, "cache_policy", "")
        policy = policy.strip() if isinstance(policy, str) else ""

        if policy:
            return policy

        return (
            "evict_other_caption_models"
            if getattr(self.config, "keep_loaded", True)
            else "unload_after_run"
        )

    def _resolve_inference_device(self, device: str) -> torch.device:
        if self._comfy_mm is not None:
            try:
                dev = self._comfy_mm.get_torch_device()
                if dev.type != "cuda":
                    raise RuntimeError(
                        "CaptionForge Joy requires CUDA for release builds. "
                        "Silent CPU fallback is disabled."
                    )
                return dev
            except RuntimeError:
                raise
            except Exception:
                pass

        if device == "auto":
            if not torch.cuda.is_available():
                raise RuntimeError(
                    "CaptionForge Joy requires CUDA for release builds. "
                    "Silent CPU fallback is disabled. "
                    "For CPU/debug experiments, edit jlc_joy_caption_engine.py manually."
                )
            return torch.device("cuda")

        dev = torch.device(device)
        if dev.type != "cuda":
            raise RuntimeError(
                f"CaptionForge Joy release mode does not allow inference device={device!r}. "
                "Use CUDA, or edit source manually for CPU/debug experiments."
            )

        return dev

    def _resolve_offload_device(self, device: str) -> torch.device:
        if self._comfy_mm is not None:
            try:
                return self._comfy_mm.unet_offload_device()
            except Exception:
                pass
        return torch.device("cpu")

    @staticmethod
    def _resolve_dtype(dtype: str) -> torch.dtype | str:
        text = dtype.lower().strip()

        if text == "auto":
            return "auto"
        if text in {"bf16", "bfloat16"}:
            return torch.bfloat16
        if text in {"fp16", "float16"}:
            return torch.float16
        if text in {"fp32", "float32"}:
            return torch.float32

        raise ValueError(f"Unsupported dtype: {dtype}")

    def _module_size(self, module) -> int:
        if self._comfy_mm is not None:
            try:
                return int(self._comfy_mm.module_size(module))
            except Exception:
                pass

        total = 0
        try:
            for p in module.parameters():
                total += p.numel() * p.element_size()
            for b in module.buffers():
                total += b.numel() * b.element_size()
        except Exception:
            total = 0
        return total

    def _free_memory(self, bytes_needed: Optional[int], device: torch.device) -> None:
        if bytes_needed is None:
            return

        if self._comfy_mm is not None:
            try:
                self._comfy_mm.free_memory(bytes_needed, device)
                return
            except Exception:
                pass

        if device.type == "cuda" and torch.cuda.is_available():
            torch.cuda.empty_cache()

    def _soft_empty_cache(self) -> None:
        if self._comfy_mm is not None:
            try:
                self._comfy_mm.soft_empty_cache()
                return
            except Exception:
                pass

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def load(self) -> None:
        local_path = self.resolve_model_path()

        if not local_path.exists():
            raise FileNotFoundError(f"Model path does not exist: {local_path}")

        self.local_model_path = local_path
        cache_key = self._cache_key(local_path)

        cached = get_cached_model(cache_key)
        if cached is not None:
            self.processor = cached["processor"]
            self.model = cached.get("model")
            self.model_size_bytes = cached.get("model_size_bytes")
            print(f"[JLC Joy Engine] Reusing cached model bundle: {local_path}")
            return

        cache_policy = self._cache_policy()

        prepare_for_model_load(
            cache_key,
            policy=cache_policy,
            role="caption",
        )

        if self.generation.seed is not None:
            set_seed(self.generation.seed)

        try:
            from transformers import AutoProcessor
        except Exception as exc:
            raise RuntimeError(
                "Could not import AutoProcessor. Update transformers:\n"
                "  pip install -U transformers accelerate pillow huggingface_hub"
            ) from exc

        old_level = logging.getLogger("transformers").getEffectiveLevel()
        if self.config.quiet_transformers_load:
            logging.getLogger("transformers").setLevel(logging.ERROR)

        try:
            print(f"[JLC Joy Engine] Loading processor: {local_path}")
            self.processor = AutoProcessor.from_pretrained(
                str(local_path),
                trust_remote_code=self.config.trust_remote_code,
            )

            # Delay model construction until prepare_for_inference(). This matches Joy's
            # memory-management pattern and avoids loading the heavy model during probes.
            self.model = None
            self.model_size_bytes = None

            register_model(
                cache_key,
                {
                    "processor": self.processor,
                    "model": self.model,
                    "model_size_bytes": self.model_size_bytes,
                    "local_path": str(local_path),
                },
                family="joy",
                model_path=str(local_path),
                device=str(self.inference_device),
                quantization=self.config.memory_mode,
                role="caption",
                unload_fn=_unload_joy_bundle,
                keep=cache_policy == "keep_this_model",
            )

            print("[JLC Joy Engine] Processor loaded. Model will load on first inference.")

        finally:
            if self.config.quiet_transformers_load:
                logging.getLogger("transformers").setLevel(old_level)

    def _cache_key(self, local_path: Path) -> str:
        return make_cache_key(
            role="caption",
            family="joy",
            model_path=str(local_path.resolve()),
            device=str(self.inference_device),
            quantization=self.config.memory_mode,
            dtype=str(self.config.dtype),
        )

    def _load_model(self) -> None:
        if self.local_model_path is None:
            self.local_model_path = self.resolve_model_path()

        try:
            from transformers import LlavaForConditionalGeneration
        except Exception as exc:
            raise RuntimeError(
                "Could not import LlavaForConditionalGeneration. Update transformers:\n"
                "  pip install -U transformers accelerate bitsandbytes pillow huggingface_hub"
            ) from exc

        local_path = self.local_model_path
        torch_dtype = self._resolve_dtype(self.config.dtype)

        if self.config.memory_mode not in MEMORY_EFFICIENT_CONFIGS:
            raise ValueError(
                f"Unsupported memory_mode: {self.config.memory_mode}. "
                f"Known modes: {', '.join(MEMORY_EFFICIENT_CONFIGS.keys())}"
            )

        if self.config.memory_mode == "Default":
            print(f"[JLC Joy Engine] Loading model in Default mode: {local_path}")
            self.model = LlavaForConditionalGeneration.from_pretrained(
                str(local_path),
                torch_dtype=torch_dtype,
                trust_remote_code=self.config.trust_remote_code,
            )
            self.model_size_bytes = self._module_size(self.model)
            self._free_memory(self.model_size_bytes, self.offload_device)
            self.model.to(self.offload_device)
        else:
            print(f"[JLC Joy Engine] Loading model in {self.config.memory_mode}: {local_path}")
            try:
                from transformers import BitsAndBytesConfig
            except Exception as exc:
                raise RuntimeError(
                    "Quantized JoyCaption modes require bitsandbytes support in the active venv. "
                    "Install bitsandbytes or use memory_mode='Default'."
                ) from exc

            if self.model_size_bytes is not None:
                self._free_memory(self.model_size_bytes, self.inference_device)

            qnt_config = BitsAndBytesConfig(
                **MEMORY_EFFICIENT_CONFIGS[self.config.memory_mode],
                llm_int8_skip_modules=[
                    "vision_tower",
                    "multi_modal_projector",
                ],
            )

            self.model = LlavaForConditionalGeneration.from_pretrained(
                str(local_path),
                torch_dtype="auto",
                quantization_config=qnt_config,
                device_map=_cuda_device_map(self.inference_device),
                trust_remote_code=self.config.trust_remote_code,
            )
            self.model_size_bytes = self._module_size(self.model)

        self.model.eval()

        if getattr(self.config, "apply_liger_kernel", False):
            try:
                from liger_kernel.transformers import apply_liger_kernel_to_llama
                language_model = getattr(self.model, "language_model", None)
                if language_model is not None:
                    apply_liger_kernel_to_llama(model=language_model)
                    print("[JLC Joy Engine] Applied Liger kernel to Joy language_model.")
                else:
                    print("[JLC Joy Engine] Liger requested but language_model was not found; continuing.")
            except Exception as exc:
                print(f"[JLC Joy Engine] Liger requested but unavailable/failed: {exc}")

        self._print_load_diagnostics()

        if self.local_model_path is not None:
            cache_policy = self._cache_policy()
            register_model(
                self._cache_key(self.local_model_path),
                {
                    "processor": self.processor,
                    "model": self.model,
                    "model_size_bytes": self.model_size_bytes,
                    "local_path": str(self.local_model_path),
                },
                family="joy",
                model_path=str(self.local_model_path),
                device=str(self.inference_device),
                quantization=self.config.memory_mode,
                role="caption",
                unload_fn=_unload_joy_bundle,
                keep=cache_policy == "keep_this_model",
            )

        print(f"[JLC Joy Engine] Loaded model (mode={self.config.memory_mode}, kbit={self.is_kbit}).")
        device_map = getattr(self.model, "hf_device_map", None)
        if device_map:
            print(f"[JLC Joy Engine] hf_device_map: {device_map}")
        else:
            try:
                print(f"[JLC Joy Engine] first parameter device: {next(self.model.parameters()).device}")
            except Exception:
                pass        

    def _print_load_diagnostics(self) -> None:
        try:
            first_device = next(self.model.parameters()).device if self.model is not None else "none"
        except Exception:
            first_device = "unknown"

        print(
            "[JLC Joy Engine] load diagnostics: "
            f"memory_mode={self.config.memory_mode}, "
            f"dtype={self.config.dtype}, "
            f"kbit={self.is_kbit}, "
            f"first_parameter_device={first_device}, "
            f"inference_device={self.inference_device}, "
            f"offload_device={self.offload_device}, "
            f"max_size={self.config.max_size}, "
            f"space_compatible_mode={getattr(self.config, 'space_compatible_mode', False)}, "
            f"liger={getattr(self.config, 'apply_liger_kernel', False)}"
        )

        device_map = getattr(self.model, "hf_device_map", None)
        if device_map:
            print(f"[JLC Joy Engine] hf_device_map: {device_map}")

        if torch.cuda.is_available():
            try:
                dev = self.inference_device
                if dev.type == "cuda":
                    free, total = torch.cuda.mem_get_info(dev)
                    allocated = torch.cuda.memory_allocated(dev)
                    reserved = torch.cuda.memory_reserved(dev)
                    gib = 1024 ** 3
                    print(
                        "[JLC Joy Engine] cuda memory: "
                        f"free={free / gib:.2f} GiB, total={total / gib:.2f} GiB, "
                        f"allocated={allocated / gib:.2f} GiB, reserved={reserved / gib:.2f} GiB"
                    )
            except Exception:
                pass

    def prepare_for_inference(self) -> None:
        if self.processor is None:
            self.load()

        if self.model is None:
            self._load_model()

        if self.is_kbit:
            return

        self._free_memory(self.model_size_bytes, self.inference_device)
        assert self.model is not None
        self.model.to(self.inference_device)

    def cleanup_after_inference(self) -> None:
        if self._cache_policy() != "unload_after_run":
            return

        if self.model is None:
            return

        # Do not fully unload here: caption_pil() still needs the processor/tokenizer
        # to decode generated tokens after model.generate() returns. Full unload is
        # handled by caption_batch() / caption_one_image() after the run completes.
        if self.is_kbit:
            return

        self.model.to(self.offload_device)
        self._soft_empty_cache()

    def unload(self) -> None:
        if self.local_model_path is not None:
            unload_after_run(self._cache_key(self.local_model_path), enabled=True)

        self.processor = None
        self.model = None
        self.model_size_bytes = None
        self._soft_empty_cache()

    @torch.inference_mode()
    def caption_pil(self, image: Image.Image) -> tuple[str, str]:
        """
        Return (final_caption, raw_caption).
        """
        if self.generation.seed is not None:
            set_seed(self.generation.seed)

        self.prepare_for_inference()

        if self.model is None or self.processor is None:
            raise RuntimeError("Model is not loaded. Call load() first.")

        system_prompt = self.config.system_prompt.strip() or DEFAULT_SYSTEM_PROMPT
        prompt = self.config.prompt.replace("\\n", "\n").strip() or DEFAULT_PROMPT
        image = image.convert("RGB")
        if getattr(self.config, "space_compatible_mode", False) and int(self.config.max_size) <= 0:
            image_for_model = image
        else:
            image_for_model = resize_for_model(image, self.config.max_size)

        convo = [
            {
                "role": "system",
                "content": system_prompt,
            },
            {
                "role": "user",
                "content": prompt,
            },
        ]

        convo_string = self.processor.apply_chat_template(
            convo,
            tokenize=False,
            add_generation_prompt=True,
        )

        inputs = self.processor(
            text=[convo_string],
            images=[image_for_model],
            return_tensors="pt",
        ).to(self.inference_device)

        model_dtype = getattr(self.model, "dtype", None)
        if (
            "pixel_values" in inputs
            and isinstance(model_dtype, torch.dtype)
            and torch.is_floating_point(inputs["pixel_values"])
        ):
            inputs["pixel_values"] = inputs["pixel_values"].to(model_dtype)

        generation_kwargs: dict[str, Any] = {
            "max_new_tokens": int(self.generation.max_new_tokens),
            "do_sample": bool(self.generation.temperature > 0),
            "suppress_tokens": None,
            "use_cache": True,
        }

        if self.generation.temperature > 0:
            generation_kwargs["temperature"] = float(self.generation.temperature)
            generation_kwargs["top_p"] = float(self.generation.top_p)
            generation_kwargs["top_k"] = None if int(self.generation.top_k) == 0 else int(self.generation.top_k)

        if self.generation.repetition_penalty and self.generation.repetition_penalty != 1.0:
            generation_kwargs["repetition_penalty"] = float(self.generation.repetition_penalty)

        device_type = self._autocast_device_type()
        autocast_enabled = self._autocast_enabled(device_type)
        bf16_supported = (device_type != "cuda") or torch.cuda.is_bf16_supported()

        tokenizer = getattr(self.processor, "tokenizer", None)
        if tokenizer is not None:
            eos_token_id = getattr(tokenizer, "eos_token_id", None)
            pad_token_id = getattr(tokenizer, "pad_token_id", None) or eos_token_id

            if pad_token_id is not None:
                generation_kwargs["pad_token_id"] = pad_token_id
            if eos_token_id is not None:
                generation_kwargs["eos_token_id"] = eos_token_id

        try:
            with torch.autocast(
                device_type=device_type,
                dtype=torch.bfloat16,
                enabled=autocast_enabled and bf16_supported,
            ):
                generate_ids = self.model.generate(
                    **inputs,
                    **generation_kwargs,
                )[0]
        finally:
            self.cleanup_after_inference()

        generate_ids = generate_ids[inputs["input_ids"].shape[1]:]

        raw_caption = self.processor.tokenizer.decode(
            generate_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        ).strip()

        final_caption = cleanup_caption(raw_caption, self.cleanup)
        return final_caption, raw_caption

    def _autocast_device_type(self) -> str:
        if self._comfy_mm is not None:
            try:
                return self._comfy_mm.get_autocast_device(self.inference_device)
            except Exception:
                pass
        return self.inference_device.type

    @staticmethod
    def _autocast_enabled(device_type: str) -> bool:
        try:
            return torch.amp.autocast_mode.is_autocast_available(device_type)
        except Exception:
            return device_type in {"cuda", "cpu"}

    def caption_path(self, image_path: str | Path) -> CaptionRecord:
        p = Path(image_path)

        if not p.exists():
            raise FileNotFoundError(f"Image does not exist: {p}")

        image = load_image_file(p)
        final_caption, raw_caption = self.caption_pil(image)

        return CaptionRecord(
            image=str(p),
            caption=final_caption,
            raw_caption=raw_caption,
            model_name=self.config.model_name,
            model_path=str(self.local_model_path or self.config.model_path),
            prompt=self.config.prompt,
            system_prompt=self.config.system_prompt,
            seed=self.generation.seed,
            temperature=self.generation.temperature,
            top_p=self.generation.top_p,
            top_k=self.generation.top_k,
            max_new_tokens=self.generation.max_new_tokens,
            max_size=self.config.max_size,
            timestamp=iso_timestamp(),
            captionforge_pass="A",
            model_family="joy",
            ensemble_run_index=0,
            image_key=str(p.resolve()),
        )

    def caption_batch(self, batch: BatchCaptionConfig) -> BatchCaptionResult:
        if not batch.input_path.strip():
            raise ValueError("BatchCaptionConfig.input_path is required.")

        result = BatchCaptionResult()

        images = list(
            iter_image_files(
                batch.input_path,
                recursive=batch.recursive,
                filename_glob=batch.filename_glob,
                extensions=batch.extensions,
            )
        )

        if batch.limit and batch.limit > 0:
            images = images[: int(batch.limit)]

        if not images:
            print(f"[JLC Joy Engine] No images found in: {batch.input_path}")
            return result

        output_dir = Path(batch.output_dir) if batch.output_dir.strip() else None

        jsonl_path: Optional[Path] = None
        if batch.write_jsonl:
            base_dir = output_dir or images[0].parent
            jsonl_path = base_dir / (batch.jsonl_filename.strip() or "captions.jsonl")

        also_jsonl_path = Path(batch.also_jsonl_path) if batch.also_jsonl_path.strip() else None

        seen_jsonl_images: set[str] = set()
        if batch.skip_existing_jsonl_images:
            if jsonl_path is not None:
                seen_jsonl_images.update(load_existing_jsonl_images(jsonl_path))
            if also_jsonl_path is not None:
                seen_jsonl_images.update(load_existing_jsonl_images(also_jsonl_path))

        if batch.write_run_config:
            base_dir = output_dir or images[0].parent
            run_config_name = batch.run_config_filename.strip() or f"jlc_joy_caption_run_config_{timestamp()}.json"
            write_run_config_json(
                base_dir / run_config_name,
                self.build_run_config(batch),
                dry_run=batch.dry_run,
            )

        print(f"[JLC Joy Engine] Found {len(images)} image(s).")

        try:
            for index, image_path in enumerate(images, start=1):
                source_for_record = str(image_path)

                try:
                    txt_path = sidecar_txt_path(image_path, output_dir)

                    if batch.skip_existing_txt and batch.write_txt and txt_path.exists() and not batch.overwrite:
                        print(f"[{index}/{len(images)}] SKIP existing TXT: {txt_path}")
                        result.skipped += 1
                        continue

                    if batch.skip_existing_jsonl_images:
                        if source_for_record in seen_jsonl_images or image_path.name in seen_jsonl_images:
                            print(f"[{index}/{len(images)}] SKIP existing JSONL image: {source_for_record}")
                            result.skipped += 1
                            continue

                    print(f"[{index}/{len(images)}] Captioning: {image_path}")
                    record = self.caption_path(image_path)
                    result.records.append(record)

                    if batch.write_txt:
                        written = write_text_sidecar(
                            txt_path,
                            record.caption,
                            overwrite=batch.overwrite,
                            backup_existing=batch.backup_existing,
                            dry_run=batch.dry_run,
                        )
                        if not written:
                            result.skipped += 1
                            print(f"[{index}/{len(images)}] SKIP existing TXT after caption: {txt_path}")

                    if jsonl_path is not None:
                        append_jsonl_records(jsonl_path, [record], dry_run=batch.dry_run)

                    if also_jsonl_path is not None:
                        append_jsonl_records(also_jsonl_path, [record], dry_run=batch.dry_run)

                except KeyboardInterrupt:
                    raise

                except Exception as exc:
                    result.failed += 1
                    print(f"[JLC Joy Engine] ERROR on {image_path}: {exc}")

                    error_record = CaptionRecord(
                        image=str(image_path),
                        caption="",
                        raw_caption="",
                        model_name=self.config.model_name,
                        model_path=str(self.local_model_path or self.config.model_path),
                        prompt=self.config.prompt,
                        system_prompt=self.config.system_prompt,
                        seed=self.generation.seed,
                        temperature=self.generation.temperature,
                        top_p=self.generation.top_p,
                        top_k=self.generation.top_k,
                        max_new_tokens=self.generation.max_new_tokens,
                        max_size=self.config.max_size,
                        timestamp=iso_timestamp(),
                        status="error",
                        error=str(exc),
                        captionforge_pass="A",
                        model_family="joy",
                        ensemble_run_index=0,
                        image_key=str(image_path.resolve()),
                    )

                    result.records.append(error_record)

                    if jsonl_path is not None:
                        append_jsonl_records(jsonl_path, [error_record], dry_run=batch.dry_run)

                    if also_jsonl_path is not None:
                        append_jsonl_records(also_jsonl_path, [error_record], dry_run=batch.dry_run)
        finally:
            if self._cache_policy() == "unload_after_run":
                self.unload()

        return result

    def build_run_config(self, batch: Optional[BatchCaptionConfig] = None) -> dict[str, Any]:
        return {
            "timestamp": iso_timestamp(),
            "engine": "JLC Joy Caption Engine",
            "joy_config": asdict(self.config),
            "generation": asdict(self.generation),
            "cleanup": {
                **asdict(self.cleanup),
                "replacement_rules": [list(rule) for rule in self.cleanup.replacement_rules],
            },
            "batch": json_safe(asdict(batch)) if batch is not None else None,
        }


def caption_one_image(
    image_path: str | Path,
    joy_config: JoyCaptionConfig,
    generation: Optional[GenerationConfig] = None,
    cleanup: Optional[CleanupConfig] = None,
) -> CaptionRecord:
    engine = JoyCaptionEngine(joy_config, generation=generation, cleanup=cleanup)
    engine.load()
    try:
        return engine.caption_path(image_path)
    finally:
        cache_policy = getattr(joy_config, "cache_policy", "")
        cache_policy = cache_policy.strip() if isinstance(cache_policy, str) else ""

        if not cache_policy:
            cache_policy = (
                "evict_other_caption_models"
                if getattr(joy_config, "keep_loaded", True)
                else "unload_after_run"
            )

        if cache_policy == "unload_after_run":
            engine.unload()
