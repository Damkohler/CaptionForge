"""
JLC CaptionForge Qwen Caption — ComfyUI Node Wrapper

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
    - The **JLC CaptionForge Qwen Caption** node provides a ComfyUI
      frontend for Qwen-family Hugging Face vision-language image captioning
      inside CaptionForge.

    - This file is the **ComfyUI-facing wrapper**, not the Qwen model
      implementation. It is responsible for:
            • ComfyUI INPUT_TYPES / widget definitions
            • optional direct IMAGE tensor input
            • IMAGE tensor conversion to PIL images
            • optional Pipeline Planner file/folder `input_path` routing
            • Qwen model dropdown and local model-root handling
            • optional model download probing
            • clear template-vs-custom prompt controls
            • CaptionForge Pipeline Planner consumption through `pipeline_plan`
            • CaptionForge Template Options consumption through `template_options`
            • TXT audit sidecar writing in planned runs
            • shared JSONL audit output in planned runs
            • direct ComfyUI caption and resolved-prompt string outputs
            • IMAGE and pipeline-plan passthrough for clean graph chaining
            • node display name, category, and mapping registration

    - Model loading and generation are delegated to the reusable
      `jlc_qwen_caption_engine.py` engine.

- CaptionForge Pipeline Role
    - This node participates in **Pass A** of the CaptionForge pipeline.

    - Pass A generates auditable caption evidence records from one or more
      captioning engines.

    - The Qwen Caption node contributes Qwen-family raw caption evidence
      compatible with downstream CaptionForge distillation, validation, and
      final caption construction.

    - CaptionForge audit fields include:
            • captionforge_pass
            • model_family
            • ensemble_run_index
            • image_key
            • raw_caption
            • final cleaned caption
            • generation parameters
            • prompt metadata
            • model metadata

- Node Workflow Model
    - The node supports two image sources:
            • direct ComfyUI IMAGE input
            • file/folder routing supplied by the CaptionForge Pipeline Planner

    - These may be used independently or together.

    - In standalone mode, the node behaves as a direct IMAGE captioning node:
            • captions the connected IMAGE or IMAGE batch
            • returns caption text and resolved prompt text
            • remains file-silent from the user's point of view
            • passes the IMAGE through unchanged

    - When connected to the CaptionForge Pipeline Planner, this node consumes
      the shared CAPTIONFORGE_PIPELINE_PLAN through the `pipeline_plan` pin and
      uses the Qwen caption family plan to determine:
            • whether Qwen captioning participates in the run
            • how many Qwen raw-caption records to generate per image
            • shared output directory
            • optional shared input path
            • recursive folder traversal
            • filename filtering
            • per-caption-instance seed values
            • per-caption-instance sampling values
            • shared max image size
            • shared max token budget
            • shared LoRA trigger word

    - In planned mode, the node can write:
            • TXT audit sidecar captions
            • JSONL audit records
            • run-configuration JSON files

- Prompting Model
    - The node supports two explicit prompt modes:
            • caption_template_mode
            • custom_prompt_mode

    - `caption_template_mode` uses caption_type, caption_length, and optional
      Template Options supplied by the `template_options` pin.

    - `custom_prompt_mode` uses custom_prompt when provided, otherwise a local
      prompt_preset fallback.

    - If both booleans are enabled, custom_prompt_mode intentionally takes
      precedence. If both are disabled, the node falls back to template mode.

    - Local extra-option widgets are not duplicated. Shared template modifiers
      live only in the CaptionForge Template Options sidecar node.

- Model and Dependency Notes
    - Qwen model choices refer to Hugging Face-style model directories under:
      ComfyUI/models/LLM/JLC_QwenCaption/

    - The node supports CaptionForge's Qwen quantization selector, including
      the Balanced 8-bit mode used to reduce VRAM pressure.

    - Runtime behavior depends on PyTorch, Transformers, Pillow, NumPy,
      optional bitsandbytes support, and the local Qwen model files.

- Design Philosophy
    - This node keeps Qwen-family captioning as one independent captioning
      voice inside CaptionForge while presenting the same workflow shape as
      the Joy and Ollama Caption nodes.

    - CaptionForge is engine-democratic: Qwen captions contribute evidence to
      the broader audit and consensus pipeline alongside Joy, Ollama VLM
      witnesses, cleanup LLMs, validators, or other robustness engines.

    - The node prioritizes reproducibility, auditability, low UI clutter, and
      clean separation between the ComfyUI interface and the backend model
      engine.

- ⚠️ Development Status
    - This is early CaptionForge raw-caption infrastructure.
    - The UI, model registry, prompt behavior, and output audit fields may
      evolve as the multi-pass CaptionForge pipeline matures.
    - The node is intended for local dataset preparation and controlled caption
      audit workflows.

- Attribution & License
  - Concept and implementation by **J. L. Córdova**
    with development assistance from **ChatGPT (OpenAI)**.

  - Designed for use with:
    https://github.com/comfyanonymous/ComfyUI

  - Copyright (c) 2026 J. L. Córdova

  - Released under the **MIT License**.
"""

from __future__ import annotations
from ...captionforge_version import CAPTIONFORGE_VERSION

MANIFEST = {
    "name": "JLC CaptionForge Qwen Caption",
    "version": CAPTIONFORGE_VERSION,
    "author": "J. L. Córdova",
    "description": (
        "Qwen-family CaptionForge frontend for local Hugging Face vision-language "
        "image captioning. Provides direct IMAGE captioning while also consuming "
        "CAPTIONFORGE_PIPELINE_PLAN objects from the CaptionForge Pipeline Planner "
        "through the pipeline_plan input. Template extras are consumed only through "
        "the template_options input from the CaptionForge Template Options sidecar. "
        "The UI separates caption_template_mode and custom_prompt_mode for clearer "
        "prompt routing while delegating model loading, quantization, generation, "
        "cleanup, TXT sidecars, and JSONL audit records to the Qwen caption engine."
    ),
}

import json
from datetime import datetime
from pathlib import Path
import time
from typing import Any

import numpy as np
import torch
from PIL import Image

import folder_paths

from ...engines.jlc_qwen_caption_engine import (
    CaptionRecord,
    CleanupConfig,
    GenerationConfig,
    MODEL_REGISTRY,
    QwenCaptionConfig,
    QwenCaptionEngine,
    append_jsonl_records,
    probe_registry_model_download,
    timestamp,
    write_run_config_json,
    write_text_sidecar,
)
from ...engines.captionforge_pipeline_planner_engine import expand_captionforge_runs
from ...engines.captionforge_caption_prompt_kit import (
    CAPTION_LENGTH_CHOICES,
    CAPTION_TYPE_CHOICES,
    build_caption_prompt,
)
from ..jlc_captionforge_template_options import resolve_effective_extra_options


JLC_QWEN_MODEL_ROOT = Path(folder_paths.models_dir) / "LLM" / "JLC_QwenCaption"
DEFAULT_JSONL_FILENAME = "captions.jsonl"
_SUPPORTED_IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff"}

DEFAULT_SYSTEM_PROMPT = (
    "You are a helpful image-captioning assistant. Describe only what is visible "
    "in the image. Do not invent unseen context."
)

DEFAULT_QWEN_PROMPT = (
    "Describe the image in a highly detailed, literal, and visually grounded way. "
    "Focus only on visible details. Include subject appearance, clothing, pose, "
    "body position, hands, facial expression, hairstyle, accessories, lighting, "
    "background, textures, colors, and spatial relationships. Write a dense "
    "descriptive prompt suitable for image captioning. Avoid speculation, avoid "
    "backstory, avoid opinions, and avoid mentioning things not clearly visible."
)

QWEN_PROMPT_PRESETS = {
    "default_literal": DEFAULT_QWEN_PROMPT,
    "dense_lora_literal": (
        "Describe the image as a dense, literal LoRA dataset caption. Focus only on visible facts: "
        "subject, clothing, body position, pose, hands, face, hair, expression, accessories, background, "
        "composition, lighting, colors, textures, and spatial relationships. Do not roleplay, continue "
        "a conversation, add safety commentary, infer backstory, or mention anything not visible."
    ),
    "concise_literal": (
        "Describe the image literally and concisely. Include the main visible subject, pose, clothing, "
        "facial expression, hairstyle, background, lighting, and important visual details. Avoid speculation, "
        "backstory, opinions, and conversation."
    ),
}


def _first_available(preferred: str, fallback: str | None, choices: dict[str, Any]) -> str:
    if preferred in choices:
        return preferred
    if fallback and fallback in choices:
        return fallback
    return next(iter(choices.keys()))


def _resolve_qwen_prompt(prompt_preset: str, custom_prompt: str) -> str:
    text = str(custom_prompt or "").strip()
    if text:
        return text
    return QWEN_PROMPT_PRESETS.get(str(prompt_preset or ""), DEFAULT_QWEN_PROMPT)


def _combine_qwen_system_and_prompt(system_prompt: str, prompt: str) -> str:
    """Qwen engine exposes a single prompt string, so system text is folded in."""
    system = str(system_prompt or "").strip()
    user = str(prompt or "").strip()
    if not system:
        return user
    return f"System instruction: {system}\n\nCaption prompt: {user}"


def _format_resolved_prompt(system_prompt: str, prompt: str) -> str:
    system = str(system_prompt or "").strip()
    user = str(prompt or "").strip()
    if not system:
        return user
    return f"SYSTEM:\n{system}\n\nUSER:\n{user}"


def _tensor_to_pil(image_tensor) -> list[Image.Image]:
    """Convert ComfyUI IMAGE tensor [B,H,W,C] float 0..1 into PIL RGB images."""
    if image_tensor is None:
        return []

    if isinstance(image_tensor, torch.Tensor):
        image_tensor = image_tensor.detach().cpu()

    if image_tensor.ndim == 3:
        image_tensor = image_tensor.unsqueeze(0)

    images: list[Image.Image] = []
    for img in image_tensor:
        arr = img.numpy()
        arr = np.clip(arr * 255.0, 0, 255).astype(np.uint8)
        images.append(Image.fromarray(arr).convert("RGB"))

    return images


def _safe_source_name(value: str) -> str:
    cleaned = []
    for ch in value.replace("\\", "/"):
        if ch.isalnum() or ch in {"-", "_", "."}:
            cleaned.append(ch)
        elif ch == "/":
            cleaned.append("__")
        else:
            cleaned.append("_")
    out = "".join(cleaned).strip("._")
    return out or "image"


def _iter_input_path_images(input_path: str, recursive: bool, filename_glob: str) -> list[tuple[str, Path]]:
    root = Path(str(input_path or "").strip())
    if not root:
        return []
    if not root.exists():
        raise RuntimeError(f"CaptionForge input_path does not exist: {root}")

    glob_text = (filename_glob or "*").strip() or "*"

    if root.is_file():
        if root.suffix.lower() not in _SUPPORTED_IMAGE_SUFFIXES:
            raise RuntimeError(f"CaptionForge input_path is not a supported image file: {root}")
        return [(_safe_source_name(root.stem), root)]

    pattern_iter = root.rglob(glob_text) if recursive else root.glob(glob_text)
    paths = sorted(
        p for p in pattern_iter
        if p.is_file() and p.suffix.lower() in _SUPPORTED_IMAGE_SUFFIXES
    )

    items: list[tuple[str, Path]] = []
    for path in paths:
        rel = path.relative_to(root).with_suffix("")
        items.append((_safe_source_name(str(rel)), path))
    return items


def _open_image(path: Path) -> Image.Image:
    with Image.open(path) as im:
        return im.convert("RGB")


def _normalize_pipeline_plan(config):
    if config is None:
        return {}
    if isinstance(config, dict):
        return dict(config)
    if isinstance(config, str):
        text = config.strip()
        if not text:
            return {}
        try:
            obj = json.loads(text)
            return obj if isinstance(obj, dict) else {}
        except Exception:
            return {}
    return {}


def _planned_caption_jsonl_path(pipeline_plan) -> Path | None:
    """Return Planner-owned shared Pass A JSONL path, if present."""
    cfg = _normalize_pipeline_plan(pipeline_plan)
    paths = cfg.get("paths") if isinstance(cfg.get("paths"), dict) else {}
    for key in ("caption_jsonl", "pass_a_jsonl"):
        value = str(paths.get(key) or "").strip()
        if value:
            return Path(value)
    return None


def _run_txt_path(output_dir: Path, source_name: str, run_count: int, run_index: int) -> Path:
    if run_count <= 1:
        return output_dir / f"{source_name}.txt"
    return output_dir / f"{source_name}__cf_run_{run_index:02d}.txt"


def _use_template_mode(caption_template_mode: bool, custom_prompt_mode: bool) -> bool:
    if bool(custom_prompt_mode):
        return False
    if bool(caption_template_mode):
        return True
    print("[CaptionForge] No caption mode was selected; falling back to caption_template_mode.")
    return True


def _resolve_template_options(template_options):
    options, name_input, metadata = resolve_effective_extra_options(
        payload=template_options,
        local_options=[],
        local_name="",
    )
    return options, name_input, metadata


def _parse_forbidden_lines(value: str) -> list[str]:
    value = (value or "").strip()
    if not value:
        return []
    return [line.strip() for line in value.splitlines() if line.strip()]


def _parse_replace_pairs(value: str) -> list[tuple[str, str]]:
    """Parse ComfyUI-friendly replacement rules in old=>new form."""
    value = (value or "").strip()
    if not value:
        return []

    rules: list[tuple[str, str]] = []
    for line in value.splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=>" not in line:
            continue
        old, new = line.split("=>", 1)
        old = old.strip()
        new = new.strip()
        if old:
            rules.append((old, new))
    return rules


class JLC_CaptionForgeQwen:
    """Canonical Qwen raw-caption witness CapNode for CaptionForge."""

    @classmethod
    def INPUT_TYPES(cls):
        qwen_default_model = _first_available(
            "Qwen2.5-VL-7B-Instruct",
            "Qwen2.5-VL-3B-Instruct",
            MODEL_REGISTRY,
        )
        caption_type_default = (
            "LoRA Literal"
            if "LoRA Literal" in CAPTION_TYPE_CHOICES
            else CAPTION_TYPE_CHOICES[0]
        )

        return {
            "required": {
                "model": (
                    list(MODEL_REGISTRY.keys()),
                    {
                        "default": qwen_default_model,
                        "tooltip": (
                            "Qwen vision-language model. Models are loaded from "
                            "ComfyUI/models/LLM/JLC_QwenCaption/. Missing models may be "
                            "downloaded automatically unless download_probe_only is enabled."
                        ),
                    },
                ),
                "qwen_quantization": (
                    ["Default", "Balanced (8-bit)"],
                    {
                        "default": "Balanced (8-bit)",
                        "tooltip": (
                            "Qwen model load mode. Balanced (8-bit) uses bitsandbytes 8-bit loading "
                            "to reduce VRAM pressure, especially for Qwen2.5-VL 7B variants."
                        ),
                    },
                ),
                "keep_loaded": (
                    "BOOLEAN",
                    {
                        "default": True,
                        "tooltip": (
                            "Keep the model cached after captioning for faster repeated runs. "
                            "CaptionForge cache policy may still evict it when another caption model must load."
                        ),
                    },
                ),

                "caption_template_mode": (
                    "BOOLEAN",
                    {
                        "default": True,
                        "tooltip": (
                            "Use the structured CaptionForge template path: caption_type, caption_length, "
                            "and optional Template Options from the template_options pin. If custom_prompt_mode "
                            "is also enabled, custom_prompt_mode takes precedence."
                        ),
                    },
                ),
                "caption_type": (
                    CAPTION_TYPE_CHOICES,
                    {
                        "default": caption_type_default,
                        "tooltip": "Caption template style used when caption_template_mode is active.",
                    },
                ),
                "caption_length": (
                    CAPTION_LENGTH_CHOICES,
                    {
                        "default": "any",
                        "tooltip": "Target caption length used when caption_template_mode is active.",
                    },
                ),

                "custom_prompt_mode": (
                    "BOOLEAN",
                    {
                        "default": False,
                        "tooltip": (
                            "Use custom_prompt when non-empty, otherwise use prompt_preset. "
                            "This overrides caption_template_mode when both toggles are enabled."
                        ),
                    },
                ),
                "prompt_preset": (
                    list(QWEN_PROMPT_PRESETS.keys()),
                    {
                        "default": "default_literal",
                        "tooltip": "Built-in prompt preset used only in custom_prompt_mode when custom_prompt is blank.",
                    },
                ),
                "system_prompt": (
                    "STRING",
                    {
                        "default": DEFAULT_SYSTEM_PROMPT,
                        "multiline": True,
                        "tooltip": (
                            "Qwen engine accepts a single prompt string, so this system instruction is "
                            "folded above the resolved caption prompt. Kept next to custom_prompt for clarity."
                        ),
                    },
                ),
                "custom_prompt": (
                    "STRING",
                    {
                        "default": "",
                        "multiline": True,
                        "tooltip": "Custom prompt used only when custom_prompt_mode is enabled. Overrides prompt_preset when non-empty.",
                    },
                ),

                "max_new_tokens": (
                    "INT",
                    {
                        "default": 384,
                        "min": 16,
                        "max": 4096,
                        "step": 8,
                        "tooltip": (
                            "Standalone token budget. When a Pipeline Planner is connected, this is "
                            "overridden by the Planner's shared max_new_tokens."
                        ),
                    },
                ),
                "temperature": (
                    "FLOAT",
                    {
                        "default": 0.75,
                        "min": 0.0,
                        "max": 2.0,
                        "step": 0.01,
                        "tooltip": (
                            "Standalone sampling temperature. When a Pipeline Planner is connected, "
                            "this is overridden by the Planner temperature schedule."
                        ),
                    },
                ),
                "top_p": (
                    "FLOAT",
                    {
                        "default": 0.90,
                        "min": 0.0,
                        "max": 1.0,
                        "step": 0.01,
                        "tooltip": (
                            "Standalone top-p sampling value. When a Pipeline Planner is connected, "
                            "this is overridden by the Planner top-p schedule."
                        ),
                    },
                ),
                "top_k": (
                    "INT",
                    {
                        "default": 50,
                        "min": 0,
                        "max": 500,
                        "step": 1,
                        "tooltip": (
                            "Standalone top-k sampling limit. When a Pipeline Planner is connected, "
                            "this is overridden by the Planner top-k schedule."
                        ),
                    },
                ),
                "repetition_penalty": (
                    "FLOAT",
                    {
                        "default": 1.08,
                        "min": 1.0,
                        "max": 2.0,
                        "step": 0.01,
                        "tooltip": (
                            "Penalty applied to repeated tokens. Kept with the core captioning parameters. "
                            "This is not currently overridden by the Pipeline Planner."
                        ),
                    },
                ),
                "max_size": (
                    "INT",
                    {
                        "default": 1024,
                        "min": 0,
                        "max": 4096,
                        "step": 64,
                        "tooltip": (
                            "Maximum longest-side image size for standalone captioning. The image is resized "
                            "in memory only. Pipeline Planner overrides this in planned runs."
                        ),
                    },
                ),

                "forbidden_phrases": (
                    "STRING",
                    {
                        "default": "",
                        "multiline": True,
                        "advanced": True,
                        "tooltip": "Optional cleanup filter: remove lines/captions containing any listed phrase, one per line.",
                    },
                ),
                "replace_pairs": (
                    "STRING",
                    {
                        "default": "",
                        "multiline": True,
                        "advanced": True,
                        "tooltip": "Optional cleanup replacements, one per line: old=>new.",
                    },
                ),
                "download_probe_only": (
                    "BOOLEAN",
                    {
                        "default": False,
                        "tooltip": (
                            "At the very bottom by design. Probe/download lightweight model metadata only, "
                            "then return a status message without captioning."
                        ),
                    },
                ),
            },
            "optional": {
                "image": (
                    "IMAGE",
                    {
                        "tooltip": (
                            "Image or batch of images to caption. The image is passed through unchanged "
                            "for clean node-to-node pipeline chaining."
                        ),
                    },
                ),
                "pipeline_plan": (
                    "CAPTIONFORGE_PIPELINE_PLAN",
                    {
                        "tooltip": (
                            "Connect the CaptionForge Pipeline Planner output here. When connected, this "
                            "node switches into Pass A evidence mode: Planner image routing, per-run seeds, "
                            "sampling schedules, shared output paths, and internal JSONL evidence append."
                        ),
                    },
                ),
                "template_options": (
                    "CAPTIONFORGE_EXTRA_OPTIONS",
                    {
                        "tooltip": (
                            "Connect the CaptionForge Template Options node here. Works in standalone and "
                            "Pipeline modes. This is the only source for template modifiers and name input."
                        ),
                    },
                ),
                "seed": (
                    "INT",
                    {
                        "forceInput": True,
                        "tooltip": "Optional standalone seed input. Ignored when a Pipeline Planner supplies a seed schedule.",
                    },
                ),
            },
        }

    RETURN_TYPES = (
        "IMAGE",
        "CAPTIONFORGE_PIPELINE_PLAN",
        "CAPTIONFORGE_EXTRA_OPTIONS",
        "STRING",
        "STRING",
    )
    RETURN_NAMES = (
        "image_out",
        "pipeline_plan_out",
        "template_options_out",
        "caption",
        "resolved_prompt",
    )
    FUNCTION = "caption"
    CATEGORY = "Captioning/CaptionForge/Captioning Nodes"

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        return float("NaN")

    def caption(
        self,
        model,
        qwen_quantization,
        keep_loaded,
        caption_template_mode,
        caption_type,
        caption_length,
        custom_prompt_mode,
        prompt_preset,
        system_prompt,
        custom_prompt,
        max_new_tokens,
        temperature,
        top_p,
        top_k,
        repetition_penalty,
        max_size,
        forbidden_phrases,
        replace_pairs,
        download_probe_only,
        image=None,
        pipeline_plan=None,
        template_options=None,
        seed=None,
    ):
        run_plan_connected = bool(pipeline_plan)
        use_caption_template = _use_template_mode(caption_template_mode, custom_prompt_mode)
        effective_extra_options, effective_person_name, _ = _resolve_template_options(template_options)

        if use_caption_template:
            user_prompt = build_caption_prompt(
                caption_type=caption_type,
                caption_length=caption_length,
                extra_options=effective_extra_options,
                name_input=effective_person_name,
                dialect="qwen",
            )
        else:
            user_prompt = _resolve_qwen_prompt(prompt_preset, custom_prompt)

        prompt = _combine_qwen_system_and_prompt(system_prompt, user_prompt)
        resolved_prompt = _format_resolved_prompt(system_prompt, user_prompt)

        if download_probe_only:
            result = probe_registry_model_download(model, JLC_QWEN_MODEL_ROOT)
            return (image, pipeline_plan, result, resolved_prompt)

        effective_seed = -1 if seed is None else int(seed)

        run_plan = expand_captionforge_runs(
            pipeline_plan,
            model_key="qwen",
            widget_captions_per_image=1,
            widget_seed=effective_seed,
            widget_temperature=float(temperature),
            widget_top_p=float(top_p),
            widget_top_k=int(top_k),
            widget_max_new_tokens=int(max_new_tokens),
            widget_max_size=int(max_size),
            widget_trigger_word="",
            widget_output_dir="",
            widget_input_path="",
            widget_recursive=True,
            widget_filename_glob="*",
        )

        if run_plan_connected and not run_plan:
            status = "[CaptionForge] Qwen disabled by Pipeline Planner."
            print(status)
            return (image, pipeline_plan, status, resolved_prompt)

        first_run = run_plan[0]
        qwen_quantization_value = "bnb_8bit" if qwen_quantization == "Balanced (8-bit)" else "none"

        generation = GenerationConfig(
            max_new_tokens=int(first_run.max_new_tokens),
            temperature=float(first_run.temperature),
            top_p=float(first_run.top_p),
            top_k=int(first_run.top_k),
            repetition_penalty=float(repetition_penalty),
            seed=first_run.seed,
        )

        cleanup = CleanupConfig(
            trigger="",
            prefix=(f"{first_run.trigger_word}," if first_run.trigger_word else ""),
            suffix="",
            forbidden_phrases=_parse_forbidden_lines(forbidden_phrases),
            replacement_rules=_parse_replace_pairs(replace_pairs),
        )

        qwen_config = QwenCaptionConfig(
            model_name=model,
            model_path="",
            model_root=str(JLC_QWEN_MODEL_ROOT),
            dtype="auto",
            device="auto",
            device_map="auto",
            quantization=qwen_quantization_value,
            trust_remote_code=True,
            keep_loaded=bool(keep_loaded),
            quiet_transformers_load=True,
            patch_lm_head_weight=True,
            ignore_mismatched_sizes=True,
            max_size=int(first_run.max_size),
            prompt=prompt,
            allow_download=True,
        )

        engine = QwenCaptionEngine(config=qwen_config, generation=generation, cleanup=cleanup)

        direct_images = [(f"comfy_image_{i:04d}", pil) for i, pil in enumerate(_tensor_to_pil(image))]
        file_images: list[tuple[str, Path]] = []
        if first_run.input_path:
            file_images = _iter_input_path_images(first_run.input_path, first_run.recursive, first_run.filename_glob)

        if direct_images and file_images:
            print(
                "[CaptionForge] IMAGE input and Pipeline Planner input_path are both active; "
                "captioning both sources."
            )

        if not direct_images and not file_images:
            raise RuntimeError(
                "No image input found. Connect an IMAGE input or provide input_path in the CaptionForge Pipeline Planner."
            )

        output_dir: Path | None = None
        jsonl_path: Path | None = None
        if run_plan_connected:
            planned_caption_jsonl_path = _planned_caption_jsonl_path(pipeline_plan)
            output_dir = (
                planned_caption_jsonl_path.parent
                if planned_caption_jsonl_path is not None
                else Path(first_run.output_dir or (Path(folder_paths.get_output_directory()) / "CaptionForge"))
            )
            output_dir.mkdir(parents=True, exist_ok=True)
            jsonl_path = planned_caption_jsonl_path or (output_dir / DEFAULT_JSONL_FILENAME)

        all_records: list[CaptionRecord] = []

        engine.load()

        if run_plan_connected and output_dir is not None:
            write_run_config_json(
                output_dir / f"jlc_captionforge_qwen_run_config_{timestamp()}.json",
                engine.build_run_config(),
                dry_run=False,
            )

        def process_one(source_name: str, pil: Image.Image):
            for run in run_plan:
                engine.generation = GenerationConfig(
                    max_new_tokens=int(run.max_new_tokens),
                    temperature=float(run.temperature),
                    top_p=float(run.top_p),
                    top_k=int(run.top_k),
                    repetition_penalty=float(repetition_penalty),
                    seed=run.seed,
                )
                engine.cleanup = CleanupConfig(
                    trigger="",
                    prefix=(f"{run.trigger_word}," if run.trigger_word else ""),
                    suffix="",
                    forbidden_phrases=_parse_forbidden_lines(forbidden_phrases),
                    replacement_rules=_parse_replace_pairs(replace_pairs),
                )
                engine.config.max_size = int(run.max_size)

                t0 = time.perf_counter()
                final_caption, raw_caption = engine.caption_pil(pil)
                dt = time.perf_counter() - t0
                print(f"[JLC CaptionForge Qwen] Generation time run {run.ensemble_run_index}: {dt:.2f}s")

                record = CaptionRecord(
                    image=source_name,
                    caption=final_caption,
                    raw_caption=raw_caption,
                    model_name=model,
                    model_path=str(engine.local_model_path or ""),
                    prompt=prompt,
                    seed=run.seed,
                    temperature=run.temperature,
                    top_p=run.top_p,
                    top_k=run.top_k,
                    max_new_tokens=run.max_new_tokens,
                    max_size=int(run.max_size),
                    timestamp=datetime.now().isoformat(timespec="seconds"),
                    captionforge_pass="A",
                    model_family="qwen",
                    ensemble_run_index=run.ensemble_run_index,
                    image_key=source_name,
                )
                all_records.append(record)

                if run_plan_connected and jsonl_path is not None and output_dir is not None:
                    append_jsonl_records(jsonl_path, [record], dry_run=False)
                    write_text_sidecar(
                        _run_txt_path(output_dir, source_name, len(run_plan), run.ensemble_run_index),
                        record.caption,
                        overwrite=True,
                        backup_existing=False,
                        dry_run=False,
                    )
                print(
                    f"[JLC CaptionForge Qwen] Captioned {source_name} "
                    f"run {run.ensemble_run_index + 1}/{len(run_plan)}"
                )

        for source_name, pil in direct_images:
            process_one(source_name, pil)

        for source_name, path in file_images:
            process_one(source_name, _open_image(path))

        if not keep_loaded:
            engine.unload()

        caption_string = "\n\n".join(r.caption for r in all_records if r.status == "ok")
        return (image, pipeline_plan, template_options, caption_string, resolved_prompt)


NODE_CLASS_MAPPINGS = {
    "JLC_CaptionForgeQwen": JLC_CaptionForgeQwen,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "JLC_CaptionForgeQwen": "\u2003JLC CaptionForge Qwen Caption",
}
