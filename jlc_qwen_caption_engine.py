#!/usr/bin/env python
"""
JLC Qwen Caption Engine

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
    - The **JLC Qwen Caption Engine** provides the shared importable backend for
      Qwen-family vision-language captioning inside CaptionForge.

    - It centralizes Qwen-specific captioning behavior so CLI wrappers and
      ComfyUI node wrappers can consume the same tested engine logic without
      duplicating model loading, generation, cleanup, or batch-output code.

    - The engine handles:
            • Qwen-family model registry lookup
            • Hugging Face download/probe behavior
            • local model-path resolution
            • processor and model loading
            • Qwen2-VL / Qwen2.5-VL model-class selection
            • CaptionForge shared model-cache integration
            • optional bitsandbytes 8-bit loading
            • Accelerate-dispatch-aware unload behavior
            • compatibility patching for selected Qwen variants
            • prompt preset and prompt-file resolution
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

    - Qwen output records include CaptionForge audit fields such as:
            • captionforge_pass
            • model_family
            • ensemble_run_index
            • image_key
            • raw_caption
            • final cleaned caption
            • generation parameters
            • model and prompt metadata

- Model and Compatibility Behavior
    - The engine supports registry-managed and direct-path Qwen-family models.

    - Registry entries are intentionally conservative and include tested or
      targeted Qwen2.5-VL captioning variants.

    - Model loading supports:
            • dtype selection
            • device selection
            • Accelerate `device_map`
            • no-quantization loading
            • bitsandbytes 8-bit loading
            • optional mismatched-size tolerance
            • optional `lm_head` / input-embedding weight tying patch

    - Accelerate-dispatched models are detected during unload so the engine does
      not call `.to("cpu")` on models managed by Accelerate hooks.

- Prompt and Cleanup Strategy
    - The engine supports practical CaptionForge prompt presets for:
            • literal dataset captions
            • LoRA-style comma-separated captions
            • natural captions
            • compact tag-style captions
            • dataset auditing
            • style-heavy descriptions
            • short literal descriptions

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
    - This engine preserves Qwen-family VLMs as one independent captioning voice
      inside CaptionForge rather than treating any single model as canonical.

    - CaptionForge is engine-democratic: Qwen captions contribute evidence to the
      broader audit and consensus pipeline alongside Joy, future local VLMs,
      cleanup LLMs, validators, or other robustness engines.

    - The engine therefore prioritizes reproducibility, auditability, and clean
      separation between model-specific inference code and ComfyUI node wrappers.

- ⚠️ Development Status
    - This is early CaptionForge Pass A engine infrastructure.
    - Registry entries should be expanded only after model-class, processor, and
      memory behavior are tested.
    - Quantization, prompt presets, compatibility patches, and audit fields may
      evolve as the multi-pass CaptionForge pipeline matures.

- Attribution & License
  - Concept and implementation by **J. L. Córdova**
    with development assistance from **ChatGPT (OpenAI)**.

  - Qwen-family model loading is designed around compatible Hugging Face
    Transformers interfaces and publicly available Qwen-family checkpoints.

  - Copyright (c) 2026 J. L. Córdova

  - Released under the **MIT License**.
"""

from __future__ import annotations

MANIFEST = {
    "name": "JLC Qwen Caption Engine",
    "version": (1, 0, 0),
    "author": "J. L. Córdova",
    "description": (
        "Shared Qwen-family CaptionForge Pass A engine for local image captioning, "
        "folder traversal, prompt resolution, caption cleanup, TXT sidecar writing, JSONL "
        "audit streaming, run-config export, model registry lookup, Hugging Face download "
        "probe support, Qwen2/Qwen2.5-VL model-class selection, optional bitsandbytes 8-bit "
        "loading, Accelerate-dispatch-aware unload behavior, compatibility patching for "
        "selected Qwen variants, and integration with the CaptionForge global model cache. "
        "Designed to keep Qwen-specific captioning logic separate from ComfyUI node wrappers "
        "while contributing auditable caption evidence to model-agnostic downstream claim "
        "extraction and consensus refinement passes."
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
# Model registry
# -------------------------------------------------------------------------
# Registry entries are intentionally conservative. You may add Qwen-family
# variants here after testing.
# -------------------------------------------------------------------------

@dataclass(frozen=True)
class QwenModelInfo:
    repo_id: str
    local_folder: str
    notes: str = ""


MODEL_REGISTRY: dict[str, QwenModelInfo] = {
    "Qwen2.5-VL-3B-Instruct": QwenModelInfo(
        repo_id="Qwen/Qwen2.5-VL-3B-Instruct",
        local_folder="Qwen2.5-VL-3B-Instruct",
    ),
    "Qwen2.5-VL-7B-Instruct": QwenModelInfo(
        repo_id="Qwen/Qwen2.5-VL-7B-Instruct",
        local_folder="Qwen2.5-VL-7B-Instruct",
    ),
    "Qwen2.5-VL-3B-Instruct-Unredacted-MAX": QwenModelInfo(
        repo_id="prithivMLmods/Qwen2.5-VL-3B-Instruct-Unredacted-MAX",
        local_folder="Qwen2.5-VL-3B-Instruct-Unredacted-MAX",
        notes="May need tied lm_head weight patch.",
    ),
    "Qwen2.5-VL-7B-Captioner-Relaxed": QwenModelInfo(
        repo_id="Ertugrul/Qwen2.5-VL-7B-Captioner-Relaxed",
        local_folder="Qwen2.5-VL-7B-Captioner-Relaxed",
    ),
    "Qwen2.5-VL-7B-NSFW-Caption-V3-abliterated": QwenModelInfo(
        repo_id="shutkit/Qwen2.5-VL-7B-NSFW-Caption-V3-abliterated",
        local_folder="Qwen2.5-VL-7B-NSFW-Caption-V3-abliterated",
    ),
    # Optional later test candidate. Keep disabled until confirmed.
    # "Qwen2-VL-7B-Captioner-Relaxed": QwenModelInfo(
    #     repo_id="Ertugrul/Qwen2-VL-7B-Captioner-Relaxed",
    #     local_folder="Qwen2-VL-7B-Captioner-Relaxed",
    # ),
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


DEFAULT_PROMPT = (
    "Describe this image in a highly detailed, literal, visually grounded way. "
    "Focus only on visible content. Include subject appearance, clothing, pose, "
    "body position, hands, facial expression, hairstyle, accessories, lighting, "
    "background, textures, colors, and spatial relationships. Avoid speculation, "
    "opinions, or anything not clearly visible. Write one dense descriptive caption "
    "suitable for image dataset captioning."
)


PROMPT_PRESETS: dict[str, str] = {
    "default_literal": DEFAULT_PROMPT,
    "lora": (
        "Write a concise LoRA training caption for this image. "
        "Use comma-separated visual phrases. "
        "Describe only visible features: subject type, pose, expression, hair, eyes if visible, "
        "clothing, visual style, lighting, and background. "
        "Include important style cues if visible, such as doll-like, glossy, quasi-3D, render style, "
        "vinyl-like skin, stylized proportions, studio lighting, or clean white background. "
        "Do not write a full sentence. "
        "Do not say 'the image depicts', 'this image shows', or 'overall style'."
    ),
    "natural": (
        "Describe this image clearly and naturally. "
        "Mention the subject, pose, expression, clothing, style, lighting, and background. "
        "Be factual and visual. Avoid speculation."
    ),
    "taggy": (
        "Create a compact tag-style caption for this image. "
        "Use short comma-separated tags only. "
        "Include subject, pose, expression, hair, eyes if visible, clothing, style, lighting, and background. "
        "Do not use complete sentences."
    ),
    "audit": (
        "Describe this image for dataset auditing. "
        "Be precise and factual. "
        "Mention visible subject, pose, expression, clothing, background, lighting, style, and any unusual details. "
        "Avoid speculation about identity or story."
    ),
    "style_heavy": (
        "Write a detailed image caption for dataset training. Describe the visible "
        "subject, clothing, pose, expression, hairstyle, accessories, lighting, "
        "camera angle, composition, background, textures, colors, and artistic style. "
        "Emphasize visual style, fashion details, material qualities, mood, and scene "
        "composition while staying literal and grounded in what is visible."
    ),
    "short_literal": (
        "Write a concise but specific visual caption. Describe the main subject, "
        "pose, clothing, expression, hairstyle, background, lighting, and style. "
        "Do not speculate beyond what is visible."
    ),
}


# -------------------------------------------------------------------------
# Dataclasses
# -------------------------------------------------------------------------

@dataclass
class GenerationConfig:
    max_new_tokens: int = 384
    temperature: float = 0.75
    top_p: float = 0.90
    top_k: int = 50
    repetition_penalty: float = 1.08
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


def _model_uses_accelerate_hooks(model: Any) -> bool:
    """
    Detect models loaded with Accelerate dispatch hooks.

    Accelerate-dispatched models must not be moved with model.to("cpu");
    doing so triggers:
        "You shouldn't move a model that is dispatched using accelerate hooks."

    For these models, cleanup should drop references and let the cache/GC/CUDA
    allocator reclaim memory instead of calling .to().
    """
    if model is None:
        return False

    if getattr(model, "hf_device_map", None):
        return True

    if getattr(model, "_hf_hook", None) is not None:
        return True

    try:
        for module in model.modules():
            if getattr(module, "_hf_hook", None) is not None:
                return True
    except Exception:
        pass

    return False


def _unload_qwen_bundle(bundle: dict[str, Any]) -> None:
    model = bundle.get("model") if isinstance(bundle, dict) else None

    if model is not None:
        try:
            if _model_uses_accelerate_hooks(model):
                print(
                    "[JLC Qwen Engine] Accelerate-dispatched model detected; "
                    "skipping model.to('cpu') during unload."
                )
            else:
                model.to("cpu")
        except Exception as exc:
            print(f"[JLC Qwen Engine] Non-fatal unload warning: {exc}")

    if isinstance(bundle, dict):
        bundle.clear()

    try:
        import gc
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass


@dataclass
class QwenCaptionConfig:
    # Either model_name or model_path may be used.
    # If model_path is supplied, it wins.
    model_name: str = "Qwen2.5-VL-3B-Instruct"
    model_path: str = ""

    # Root used only for registry-managed model folders.
    # CLI default can point this to ./models/LLM/JLC_QwenCaption or another local root.
    model_root: str = "models/LLM/JLC_QwenCaption"

    # Loading behavior.
    dtype: str = "auto"               # auto, bf16, bfloat16, fp16, float16, fp32, float32
    device: str = "auto"              # auto, cuda, cpu, cuda:0, etc. Used when device_map is empty.
    device_map: str = "auto"          # auto or empty string. "auto" uses accelerate placement.
    quantization: str = "none"        # none, bnb_8bit
    trust_remote_code: bool = True
    keep_loaded: bool = True
    quiet_transformers_load: bool = True
    patch_lm_head_weight: bool = True
    ignore_mismatched_sizes: bool = True

    # Image and prompt behavior.
    max_size: int = 1024
    prompt: str = DEFAULT_PROMPT

    # Download/probe behavior for registry models.
    allow_download: bool = True


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
    model_family: str = "qwen"
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


def parse_seed(value: Optional[str]) -> Optional[int]:
    if value is None:
        return None

    text = str(value).strip().lower()
    if not text:
        return None

    if text == "random":
        return random.randint(0, 2**32 - 1)

    return int(text)


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


def parse_forbidden_phrases(value: str, separator: str = ",") -> list[str]:
    if not value.strip():
        return []
    if separator == "lines":
        return [line.strip() for line in value.splitlines() if line.strip()]
    return [item.strip() for item in value.split(separator) if item.strip()]


def parse_replacement_rule(value: str) -> tuple[str, str]:
    # Accept both old;new from the CLI version and old=>new from the ComfyUI version.
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


def load_replacement_rules_file(path: Path) -> list[tuple[str, str]]:
    if not path.exists():
        raise FileNotFoundError(f"Replacement file does not exist: {path}")
    return parse_replacement_rules_text(path.read_text(encoding="utf-8"))


def load_prompt_file(path: str | Path) -> str:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Prompt file does not exist: {p}")
    return p.read_text(encoding="utf-8").strip()


def resolve_prompt(
    prompt: str = "",
    prompt_file: str = "",
    prompt_preset: str = "default_literal",
) -> str:
    prompt = prompt.strip() if prompt else ""
    prompt_file = prompt_file.strip() if prompt_file else ""

    if prompt:
        return prompt

    if prompt_file:
        text = load_prompt_file(prompt_file)
        if text:
            return text

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
            f"Unknown Qwen model_name: {model_name}. "
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
            f"[JLC Qwen Engine] Probe download for {info.repo_id} -> {local_path}\n"
            "[JLC Qwen Engine] Large weight files will be skipped."
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

    print(f"[JLC Qwen Engine] Downloading {info.repo_id} -> {local_path}")
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
        "JLC Qwen Caption download probe completed.\n\n"
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
        # Preserve the old CLI behavior: suffix is appended literally.
        # If you want a comma suffix, include it in the suffix text.
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


def write_run_config_json(path: Path, config: dict[str, Any], dry_run: bool = False) -> None:
    if dry_run:
        return
    safe_mkdir(path.parent)
    path.write_text(json.dumps(json_safe(config), indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

# -------------------------------------------------------------------------
# Engine
# -------------------------------------------------------------------------



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


class QwenCaptionEngine:
    def __init__(
        self,
        config: QwenCaptionConfig,
        generation: Optional[GenerationConfig] = None,
        cleanup: Optional[CleanupConfig] = None,
    ) -> None:
        self.config = config
        self.generation = generation or GenerationConfig()
        self.cleanup = cleanup or CleanupConfig()
        self.processor = None
        self.model = None
        self.local_model_path: Optional[Path] = None

    def resolve_model_path(self) -> Path:
        if self.config.model_path.strip():
            return Path(self.config.model_path).expanduser()

        return download_registry_model_if_needed(
            model_name=self.config.model_name,
            model_root=self.config.model_root,
            metadata_only=False,
            allow_download=self.config.allow_download,
        )

    @staticmethod
    def _resolve_device(device: str) -> torch.device:
        if device == "auto":
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return torch.device(device)

    @staticmethod
    def _resolve_dtype(dtype: str, device: torch.device) -> torch.dtype | str:
        text = dtype.lower().strip()

        if text == "auto":
            # With device_map="auto", torch_dtype="auto" usually works well.
            return "auto"

        if text in {"bf16", "bfloat16"}:
            return torch.bfloat16
        if text in {"fp16", "float16"}:
            return torch.float16
        if text in {"fp32", "float32"}:
            return torch.float32

        raise ValueError(f"Unsupported dtype: {dtype}")

    @staticmethod
    def _resolve_quantization(quantization: str) -> str:
        text = (quantization or "none").lower().strip()
        aliases = {
            "": "none",
            "none": "none",
            "off": "none",
            "false": "none",
            "bnb_8bit": "bnb_8bit",
            "8bit": "bnb_8bit",
            "8-bit": "bnb_8bit",
            "bitsandbytes_8bit": "bnb_8bit",
        }
        if text not in aliases:
            raise ValueError(
                f"Unsupported Qwen quantization: {quantization}. "
                "Supported values: none, bnb_8bit"
            )
        return aliases[text]

    @staticmethod
    def _detect_model_type(model_path: Path) -> str:
        config_path = model_path / "config.json"
        if not config_path.exists():
            return ""

        try:
            obj = json.loads(config_path.read_text(encoding="utf-8"))
            return str(obj.get("model_type", "")).strip().lower()
        except Exception:
            return ""

    @staticmethod
    def _load_model_class(model_type: str):
        try:
            if model_type == "qwen2_vl":
                from transformers import Qwen2VLForConditionalGeneration
                return Qwen2VLForConditionalGeneration

            # Default to Qwen2.5-VL because that is the primary supported family.
            from transformers import Qwen2_5_VLForConditionalGeneration
            return Qwen2_5_VLForConditionalGeneration

        except Exception as exc:
            raise RuntimeError(
                "Could not import the required Qwen VL model class. "
                "Update dependencies in the active venv:\n"
                "  pip install -U transformers accelerate pillow qwen-vl-utils huggingface_hub\n"
            ) from exc

    def load(self) -> None:
        local_path = self.resolve_model_path()

        if not local_path.exists():
            raise FileNotFoundError(f"Model path does not exist: {local_path}")

        self.local_model_path = local_path

        device = self._resolve_device(self.config.device)
        torch_dtype = self._resolve_dtype(self.config.dtype, device)
        quantization = self._resolve_quantization(self.config.quantization)

        cache_key = make_cache_key(
            role="caption",
            family="qwen",
            model_path=str(local_path.resolve()),
            device=device,
            quantization=quantization,
            dtype=str(torch_dtype),
        )

        cached = get_cached_model(cache_key)
        if cached is not None:
            self.processor = cached["processor"]
            self.model = cached["model"]
            print(f"[JLC Qwen Engine] Reusing cached model: {local_path}")
            return
        
        cache_policy = getattr(
            self.config,
            "cache_policy",
            "evict_other_caption_models" if getattr(self.config, "keep_loaded", True) else "unload_after_run",
        )

        prepare_for_model_load(
            cache_key,
            policy=cache_policy,
            role="caption",
        )

############################################
############################################

        if self.generation.seed is not None:
            set_seed(self.generation.seed)

        try:
            from transformers import AutoProcessor
        except Exception as exc:
            raise RuntimeError(
                "Could not import AutoProcessor. Update transformers:\n"
                "  pip install -U transformers accelerate pillow qwen-vl-utils"
            ) from exc

        old_level = logging.getLogger("transformers").getEffectiveLevel()
        if self.config.quiet_transformers_load:
            logging.getLogger("transformers").setLevel(logging.ERROR)

        try:

            model_type = self._detect_model_type(local_path)
            model_cls = self._load_model_class(model_type)

            max_pixels = self.config.max_size * self.config.max_size if self.config.max_size > 0 else None
            min_pixels = 256 * 256

            processor_kwargs: dict[str, Any] = {
                "trust_remote_code": self.config.trust_remote_code,
            }
            if max_pixels:
                processor_kwargs.update({"min_pixels": min_pixels, "max_pixels": max_pixels})

            print(f"[JLC Qwen Engine] Loading processor: {local_path}")
            self.processor = AutoProcessor.from_pretrained(
                str(local_path),
                **processor_kwargs,
            )

            model_kwargs: dict[str, Any] = {
                "torch_dtype": torch_dtype,
                "trust_remote_code": self.config.trust_remote_code,
            }

            if self.config.ignore_mismatched_sizes:
                model_kwargs["ignore_mismatched_sizes"] = True

            effective_device_map = self.config.device_map.strip()
            if quantization == "bnb_8bit":
                try:
                    from transformers import BitsAndBytesConfig
                except Exception as exc:
                    raise RuntimeError(
                        "Qwen 8-bit loading requires bitsandbytes support in the active ComfyUI venv. "
                        "Install/verify compatible packages before using quantization='bnb_8bit'."
                    ) from exc
                
                warnings.filterwarnings(
                    "ignore",
                    message=r"MatMul8bitLt: inputs will be cast from torch\.bfloat16 to float16 during quantization",
                    category=UserWarning,
                    module=r"bitsandbytes\.autograd\._functions",
                )

                model_kwargs["quantization_config"] = BitsAndBytesConfig(load_in_8bit=True)

                # bitsandbytes quantized models should be loaded through Accelerate dispatch.
                # Keep this explicit so users do not accidentally request a later .to(device).
                if not effective_device_map:
                    effective_device_map = "auto"

            if effective_device_map:
                model_kwargs["device_map"] = effective_device_map

            print(f"[JLC Qwen Engine] Loading model: {local_path}")
            print(
                f"[JLC Qwen Engine] dtype={torch_dtype}, device={device}, "
                f"device_map={effective_device_map!r}, quantization={quantization}"
            )

            print("[JLC Qwen Engine] bitsandbytes 8-bit enabled; "
                       "suppressing known bf16->fp16 MatMul8bitLt warning.")
            
            device_map = getattr(self.model, "hf_device_map", None)
            if device_map:
                print(f"[JLC Qwen Engine] hf_device_map: {device_map}")
            else:
                try:
                    print(f"[JLC Qwen Engine] first parameter device: {next(self.model.parameters()).device}")
                except Exception:
                    pass
            
            self.model = model_cls.from_pretrained(
                str(local_path),
                **model_kwargs,
            )

            if self.config.patch_lm_head_weight and quantization == "none":
                self._maybe_patch_lm_head_weight()
            elif self.config.patch_lm_head_weight and quantization != "none":
                print(
                    "[JLC Qwen Engine] Skipping lm_head weight patch for quantized Qwen load "
                    f"(quantization={quantization})."
                )

            if not effective_device_map and quantization == "none":
                self.model.to(device)

            self.model.eval()

            ####################################
            ####################################

            register_model(
                cache_key,
                {
                    "processor": self.processor,
                    "model": self.model,
                    "local_path": str(local_path),
                },
                family="qwen",
                model_path=str(local_path),
                device=device,
                quantization=quantization,
                role="caption",
                unload_fn=_unload_qwen_bundle,
                keep=cache_policy == "keep_this_model",
            )

            print("[JLC Qwen Engine] Model loaded.")

        finally:
            if self.config.quiet_transformers_load:
                logging.getLogger("transformers").setLevel(old_level)

    def _maybe_patch_lm_head_weight(self) -> None:
        if self.model is None:
            return

        try:
            lm_head = getattr(self.model, "lm_head", None)
            embeddings = self.model.get_input_embeddings()
            if lm_head is None or embeddings is None:
                return

            if hasattr(lm_head, "weight") and hasattr(embeddings, "weight"):
                if tuple(lm_head.weight.shape) == tuple(embeddings.weight.shape):
                    lm_head.weight = embeddings.weight
        except Exception:
            # Non-fatal compatibility patch.
            return

    def unload(self) -> None:
        if self.local_model_path is not None:
            device = self._resolve_device(self.config.device)
            torch_dtype = self._resolve_dtype(self.config.dtype, device)

            quantization = self._resolve_quantization(self.config.quantization)

            cache_key = make_cache_key(
                role="caption",
                family="qwen",
                model_path=str(self.local_model_path.resolve()),
                device=device,
                quantization=quantization,
                dtype=str(torch_dtype),
            )

            unload_after_run(cache_key, enabled=True)

        self.processor = None
        self.model = None

    @torch.inference_mode()
    def caption_pil(self, image: Image.Image) -> tuple[str, str]:
        """
        Return (final_caption, raw_caption).
        """
        if self.model is None or self.processor is None:
            raise RuntimeError("Model is not loaded. Call load() first.")

        if self.generation.seed is not None:
            set_seed(self.generation.seed)

        prompt = self.config.prompt.replace("\\n", "\n").strip() or DEFAULT_PROMPT
        image = image.convert("RGB")
        image_for_model = resize_for_model(image, self.config.max_size)

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image_for_model},
                    {"type": "text", "text": prompt},
                ],
            }
        ]

        text = self.processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

        # Prefer qwen-vl-utils because it follows Qwen's official multimodal message handling.
        try:
            from qwen_vl_utils import process_vision_info
            image_inputs, video_inputs = process_vision_info(messages)
            inputs = self.processor(
                text=[text],
                images=image_inputs,
                videos=video_inputs,
                padding=True,
                return_tensors="pt",
            )
        except Exception:
            # Fallback to the simpler path used by the original CLI.
            inputs = self.processor(
                text=[text],
                images=[image_for_model],
                padding=True,
                return_tensors="pt",
            )

        try:
            device = next(self.model.parameters()).device
            inputs = inputs.to(device)
        except Exception:
            pass

        generation_kwargs: dict[str, Any] = {
            "max_new_tokens": int(self.generation.max_new_tokens),
            "repetition_penalty": float(self.generation.repetition_penalty),
        }

        tokenizer = getattr(self.processor, "tokenizer", None)
        if tokenizer is not None:
            if getattr(tokenizer, "pad_token_id", None) is not None:
                generation_kwargs["pad_token_id"] = tokenizer.pad_token_id
            elif getattr(tokenizer, "eos_token_id", None) is not None:
                generation_kwargs["pad_token_id"] = tokenizer.eos_token_id

            if getattr(tokenizer, "eos_token_id", None) is not None:
                generation_kwargs["eos_token_id"] = tokenizer.eos_token_id

        if self.generation.temperature > 0:
            generation_kwargs.update(
                {
                    "do_sample": True,
                    "temperature": float(self.generation.temperature),
                    "top_p": float(self.generation.top_p),
                }
            )
            if self.generation.top_k > 0:
                generation_kwargs["top_k"] = int(self.generation.top_k)
        else:
            generation_kwargs["do_sample"] = False

        generated_ids = self.model.generate(
            **inputs,
            **generation_kwargs,
        )

        generated_ids_trimmed = [
            out_ids[len(in_ids):]
            for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]

        decoded = self.processor.batch_decode(
            generated_ids_trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )

        raw_caption = decoded[0].strip() if decoded else ""
        final_caption = cleanup_caption(raw_caption, self.cleanup)
        return final_caption, raw_caption

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
            seed=self.generation.seed,
            temperature=self.generation.temperature,
            top_p=self.generation.top_p,
            top_k=self.generation.top_k,
            max_new_tokens=self.generation.max_new_tokens,
            max_size=self.config.max_size,
            timestamp=iso_timestamp(),
            captionforge_pass="A",
            model_family="qwen",
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
            print(f"[JLC Qwen Engine] No images found in: {batch.input_path}")
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
            run_config_name = batch.run_config_filename.strip() or f"jlc_qwen_caption_run_config_{timestamp()}.json"
            write_run_config_json(
                base_dir / run_config_name,
                self.build_run_config(batch),
                dry_run=batch.dry_run,
            )

        print(f"[JLC Qwen Engine] Found {len(images)} image(s).")

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
                print(f"[JLC Qwen Engine] ERROR on {image_path}: {exc}")

                error_record = CaptionRecord(
                    image=str(image_path),
                    caption="",
                    raw_caption="",
                    model_name=self.config.model_name,
                    model_path=str(self.local_model_path or self.config.model_path),
                    prompt=self.config.prompt,
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
                    model_family="qwen",
                    ensemble_run_index=0,
                    image_key=str(image_path.resolve()),
                )

                result.records.append(error_record)

                if jsonl_path is not None:
                    append_jsonl_records(jsonl_path, [error_record], dry_run=batch.dry_run)

                if also_jsonl_path is not None:
                    append_jsonl_records(also_jsonl_path, [error_record], dry_run=batch.dry_run)


        return result
    
    
    def build_run_config(self, batch: Optional[BatchCaptionConfig] = None) -> dict[str, Any]:
        return {
            "timestamp": iso_timestamp(),
            "engine": "JLC Qwen Caption Engine",
            "qwen_config": asdict(self.config),
            "generation": asdict(self.generation),
            "cleanup": {
                **asdict(self.cleanup),
                # Tuples serialize as lists anyway; explicit for readability.
                "replacement_rules": [list(rule) for rule in self.cleanup.replacement_rules],
            },
            "batch": json_safe(asdict(batch)) if batch is not None else None,
        }


def caption_one_image(
    image_path: str | Path,
    qwen_config: QwenCaptionConfig,
    generation: Optional[GenerationConfig] = None,
    cleanup: Optional[CleanupConfig] = None,
) -> CaptionRecord:
    engine = QwenCaptionEngine(qwen_config, generation=generation, cleanup=cleanup)
    engine.load()
    try:
        return engine.caption_path(image_path)
    finally:
        cache_policy = getattr(
            qwen_config,
            "cache_policy",
            "evict_other_caption_models" if getattr(qwen_config, "keep_loaded", True) else "unload_after_run",
        )

        if cache_policy == "unload_after_run":
            engine.unload()
