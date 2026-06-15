"""
JLC Qwen Caption — ComfyUI Node Wrapper

- CaptionForge
  - This node is part of **CaptionForge**, a model-agnostic captioning
    framework for ComfyUI developed by **J. L. Córdova**.

  - Repository
    https://github.com/Damkohler/CaptionForge

  - CaptionForge focuses on practical dataset-captioning infrastructure for:
        • LoRA dataset preparation
        • multi-engine caption generation
        • JSONL audit trails
        • claim extraction and refinement
        • consensus-oriented caption improvement

- Node Purpose
    - The **JLC Qwen Caption** node provides a single-node ComfyUI frontend for
      Qwen-family vision-language captioning inside CaptionForge.

    - This file is the **ComfyUI-facing wrapper**, not the reusable captioning
      engine. It is responsible for:
            • ComfyUI INPUT_TYPES / widget definitions
            • optional direct IMAGE tensor input
            • IMAGE tensor conversion to PIL images
            • optional file/folder `input_path` routing
            • prompt preset / prompt file / custom prompt selection
            • fixed model-root integration under `models/LLM/`
            • download-probe exposure through the node UI
            • TXT / JSONL / run-config routing controls
            • ComfyUI output strings for captions and JSONL records
            • node display name, category, and mapping registration
            • passing user settings into the shared Qwen caption engine

    - The actual reusable captioning implementation lives in:
            jlc_qwen_caption_engine.py

      That engine handles:
            • Qwen model registry lookup
            • Hugging Face model probing/downloading
            • processor and model loading
            • Qwen2 / Qwen2.5 model-class selection
            • CaptionForge shared model-cache integration
            • optional bitsandbytes 8-bit loading
            • Accelerate-aware unload behavior
            • compatibility patching for selected Qwen variants
            • prompt resolution
            • generation settings
            • caption cleanup
            • image folder traversal
            • TXT sidecar writing
            • JSONL audit output
            • run-configuration export

- CaptionForge Pass Role
    - This node participates in **Pass A** of the CaptionForge pipeline.

    - Pass A generates auditable caption evidence records from one or more
      captioning engines.

    - The Qwen node writes records compatible with downstream CaptionForge
      processing, including Pass B claim extraction.

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
    - The node supports two input paths:
            • direct ComfyUI IMAGE input
            • file/folder captioning through `input_path`

    - These may be used independently or together.

    - For direct IMAGE input with no output directory, sidecar files are written
      to:
            ComfyUI/output/jlc_qwen_caption/

    - For file/folder input with no output directory, sidecar files are written
      beside the source images.

    - The node can write:
            • TXT sidecar captions
            • JSONL audit records
            • run-configuration JSON files
            • direct ComfyUI string outputs

- Prompting Model
        prompt = custom_prompt
              or prompt_file
              or prompt_preset
              or default_literal

    where:
        • `custom_prompt` overrides all other prompt sources
        • `prompt_file` enables reusable external prompt files
        • `prompt_preset` provides built-in captioning styles
        • `default_literal` provides the conservative dataset baseline

- Model and Dependency Notes
    - Qwen VLM repositories are multi-file Hugging Face model directories,
      not single checkpoint files.

    - Full model downloads may require several gigabytes of disk space.

    - The download-probe mode intentionally skips large weight files and only
      verifies folder creation, Hugging Face access, and lightweight metadata
      download.

    - A metadata-probed folder is not a complete usable model unless full
      weights are later downloaded or copied into place.

    - Runtime behavior depends on the active ComfyUI Python environment,
      PyTorch, Transformers, Accelerate, Pillow, qwen-vl-utils, bitsandbytes
      when 8-bit mode is selected, and Hugging Face Hub dependencies.

- Design Philosophy
    - This node preserves Qwen-family VLMs as one independent captioning voice
      inside CaptionForge rather than treating any single model as canonical.

    - CaptionForge is engine-democratic: Qwen captions contribute evidence to the
      broader audit and consensus pipeline alongside Joy, future local VLMs,
      cleanup LLMs, validators, or other robustness engines.

    - The node therefore prioritizes reproducibility, auditability, and clean
      separation between the ComfyUI interface and the shared model-specific
      captioning engine.

- ⚠️ Development Status
    - This is early CaptionForge Pass A ComfyUI wrapper infrastructure.
    - The UI, prompt presets, model registry, and output audit fields may evolve
      as the multi-pass CaptionForge pipeline matures.
    - The node is intended for local dataset preparation and controlled caption
      audit workflows.

- Attribution & License
  - Concept and implementation by **J. L. Córdova**
    with development assistance from **ChatGPT (OpenAI)**.

  - Qwen-family model loading is designed around compatible Hugging Face
    Transformers interfaces and publicly available Qwen-family checkpoints.

  - Designed for use with:
    https://github.com/comfyanonymous/ComfyUI

  - Copyright (c) 2026 J. L. Córdova

  - Released under the **MIT License**.
"""

from __future__ import annotations

MANIFEST = {
    "name": "JLC Qwen Caption",
    "version": (1, 1, 0),
    "author": "J. L. Córdova",
    "description": (
        "Single-node ComfyUI frontend for Qwen-family vision-language captioning inside "
        "CaptionForge Pass A. Provides widget schema, direct IMAGE tensor handling, "
        "file/folder routing, prompt preset/file/custom prompt selection, fixed models/LLM "
        "integration, download-probe exposure, TXT sidecar controls, JSONL audit output, "
        "run-config export, and caption/JSONL string returns. Delegates reusable model "
        "loading, Qwen2/Qwen2.5 generation, optional 8-bit loading, cleanup, cache integration, "
        "batch traversal, and audit-record creation to jlc_qwen_caption_engine.py so Qwen "
        "captions can contribute auditable evidence to downstream model-agnostic claim "
        "extraction and consensus refinement passes."
    ),
}

import json
from pathlib import Path
from datetime import datetime

import torch
import numpy as np
from PIL import Image
import time

import folder_paths

from ..engines.jlc_qwen_caption_engine import (
    BatchCaptionConfig,
    CaptionRecord,
    CleanupConfig,
    GenerationConfig,
    MODEL_REGISTRY,
    PROMPT_PRESETS,
    QwenCaptionConfig,
    QwenCaptionEngine,
    append_jsonl_records,
    load_existing_jsonl_images,
    probe_registry_model_download,
    record_to_json,
    resolve_prompt,
    timestamp,
    write_run_config_json,
    write_text_sidecar,
)

from ..engines.captionforge_pipeline_planner_engine import expand_captionforge_runs

from ..engines.captionforge_caption_prompt_kit import (
    CAPTION_LENGTH_CHOICES,
    CAPTION_TYPE_CHOICES,
    build_caption_prompt,
)
from .jlc_captionforge_template_options import (
    CAPTIONFORGE_EXTRA_OPTIONS_CHOICES,
    resolve_effective_extra_options,
)
    

# -------------------------------------------------------------------------
# Fixed JLC model root
# -------------------------------------------------------------------------
# Not user-changeable from the node UI.
# Models are stored under:
#   ComfyUI/models/LLM/JLC_QwenCaption/<model-folder>
# -------------------------------------------------------------------------

JLC_QWEN_MODEL_ROOT = Path(folder_paths.models_dir) / "LLM" / "JLC_QwenCaption"


# -------------------------------------------------------------------------
# ComfyUI helpers
# -------------------------------------------------------------------------

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


def _parse_forbidden_lines(value: str) -> list[str]:
    value = (value or "").strip()
    if not value:
        return []
    return [line.strip() for line in value.splitlines() if line.strip()]


def _parse_replace_pairs(value: str) -> list[tuple[str, str]]:
    """
    ComfyUI-friendly replacement format, matching the previous node:
      old=>new
      old phrase=>new phrase
    """
    value = (value or "").strip()
    if not value:
        return []

    rules: list[tuple[str, str]] = []
    for line in value.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if "=>" not in line:
            continue
        old, new = line.split("=>", 1)
        old = old.strip()
        new = new.strip()
        if old:
            rules.append((old, new))
    return rules




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


def _planned_caption_jsonl_path(captionforge_run_config) -> Path | None:
    """Return Planner-owned shared Pass A JSONL path, if present.

    The Planner owns the canonical caption evidence file name. Caption witness
    nodes should append to this exact path in planned mode so the final
    CaptionForge node can read the same file without widget duplication.
    """
    cfg = _normalize_pipeline_plan(captionforge_run_config)
    paths = cfg.get("paths") if isinstance(cfg.get("paths"), dict) else {}
    for key in ("caption_jsonl", "pass_a_jsonl"):
        value = str(paths.get(key) or "").strip()
        if value:
            return Path(value)
    return None

def _make_jsonl_string(records: list[CaptionRecord]) -> str:
    return "\n".join(json.dumps(record_to_json(r), ensure_ascii=False) for r in records)


def _extra_options_from_widgets(*items: str) -> list[str]:
    return [str(item).strip() for item in items if str(item).strip()]


class JLC_QwenCaption:
    """
    Single-node Qwen VLM captioner.

    This node preserves the original JLC Qwen Caption ComfyUI interface,
    but delegates the implementation to `jlc_qwen_caption_engine.py`.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": (
                    list(MODEL_REGISTRY.keys()),
                    {
                        "default": "Qwen2.5-VL-3B-Instruct",
                        "tooltip": (
                            "Select which supported Qwen vision-language model to use for captioning. "
                            "The node loads it from the fixed JLC_QwenCaption model folder, downloading "
                            "it if needed."
                        ),
                    },
                ),

                "qwen_quantization": (
                    ["Default", "Balanced (8-bit)"],
                    {
                        "default": "Balanced (8-bit)",
                        "tooltip": (
                            "Qwen model load mode. Balanced (8-bit) uses bitsandbytes 8-bit loading "
                            "to reduce VRAM pressure, especially for Qwen2.5-VL 7B variants. This is "
                            "the recommended CaptionForge default on 16 GB VRAM systems."
                        ),
                    },
                ),

                "input_path": (
                    "STRING",
                    {
                        "default": "",
                        "multiline": False,
                        "tooltip": (
                            "Optional standalone file or folder path. If this points to a single image "
                            "file, that file is captioned. If it points to a folder, all supported "
                            "images in that folder are captioned. Can be used instead of, or together "
                            "with, the IMAGE input."
                        ),
                    },
                ),

                "recursive": (
                    "BOOLEAN",
                    {
                        "default": True,
                        "tooltip": (
                            "Standalone folder mode: if input_path points to a folder, search "
                            "subfolders too. Disable to process only images directly inside the "
                            "selected folder."
                        ),
                    },
                ),

                "filename_glob": (
                    "STRING",
                    {
                        "default": "*",
                        "multiline": False,
                        "tooltip": (
                            "Standalone folder mode filename filter. Examples: *, *.png, *.jpg, "
                            "JessJenn_*.webp, *_closeup.*"
                        ),
                    },
                ),

                "prompt_mode": (
                    ["custom_or_preset", "caption_template"],
                    {
                        "default": "caption_template",
                        "tooltip": (
                            "custom_or_preset uses custom_prompt, prompt_file, or prompt_preset. "
                            "caption_template uses caption_type, caption_length, local extra options, "
                            "or an optional CAPTIONFORGE_EXTRA_OPTIONS sidecar input."
                        ),
                    },
                ),

                "caption_type": (
                    CAPTION_TYPE_CHOICES,
                    {
                        "default": "LoRA Literal" if "LoRA Literal" in CAPTION_TYPE_CHOICES else CAPTION_TYPE_CHOICES[0],
                        "tooltip": "Caption template style used in caption_template mode.",
                    },
                ),

                "caption_length": (
                    CAPTION_LENGTH_CHOICES,
                    {
                        "default": "any",
                        "tooltip": "Target caption length for caption_template mode. Numeric values are interpreted as word limits.",
                    },
                ),

                "extra_option1": (
                    CAPTIONFORGE_EXTRA_OPTIONS_CHOICES,
                    {"default": "", "tooltip": "Optional instruction appended to the generated prompt."},
                ),

                "extra_option2": (
                    CAPTIONFORGE_EXTRA_OPTIONS_CHOICES,
                    {"default": "", "tooltip": "Optional instruction appended to the generated prompt.", "advanced": True},
                ),

                "extra_option3": (
                    CAPTIONFORGE_EXTRA_OPTIONS_CHOICES,
                    {"default": "", "tooltip": "Optional instruction appended to the generated prompt.", "advanced": True},
                ),

                "extra_option4": (
                    CAPTIONFORGE_EXTRA_OPTIONS_CHOICES,
                    {"default": "", "tooltip": "Optional instruction appended to the generated prompt.", "advanced": True},
                ),

                "extra_option5": (
                    CAPTIONFORGE_EXTRA_OPTIONS_CHOICES,
                    {"default": "", "tooltip": "Optional instruction appended to the generated prompt.", "advanced": True},
                ),

                "extra_option6": (
                    CAPTIONFORGE_EXTRA_OPTIONS_CHOICES,
                    {"default": "", "tooltip": "Optional instruction appended to the generated prompt.", "advanced": True},
                ),

                "person_name": (
                    "STRING",
                    {"default": "", "multiline": False, "tooltip": "Replacement value for the {name} placeholder in matching extra options.", "advanced": True},
                ),

                "prompt_preset": (
                    list(PROMPT_PRESETS.keys()),
                    {
                        "default": "default_literal" if "default_literal" in PROMPT_PRESETS else sorted(PROMPT_PRESETS.keys())[0],
                        "tooltip": (
                            "Built-in CaptionForge prompt preset. Used when custom_prompt is blank "
                            "and no prompt_file is provided."
                        ),
                    },
                ),

                "custom_prompt": (
                    "STRING",
                    {
                        "default": "",
                        "multiline": True,
                        "tooltip": (
                            "Optional custom prompt text. If provided, this overrides both prompt_file "
                            "and prompt_preset. This remains model-specific and is not currently "
                            "overridden by the CaptionForge Pipeline Planner."
                        ),
                    },
                ),

                "prompt_file": (
                    "STRING",
                    {
                        "default": "",
                        "multiline": False,
                        "tooltip": (
                            "Optional path to a text file containing the caption prompt. Used only "
                            "when custom_prompt is blank. If both custom_prompt and prompt_file are "
                            "blank, prompt_preset is used."
                        ),
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
                            "Maximum number of new tokens for standalone captioning. When a "
                            "CaptionForge Pipeline Planner is connected, this is overridden by the Run "
                            "Plan's shared constant token budget."
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
                            "Sampling temperature for standalone captioning. When a CaptionForge "
                            "Pipeline Planner is connected, this is overridden by the Pipeline Planner temperature "
                            "schedule."
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
                            "Top-p sampling value for standalone captioning. When a CaptionForge "
                            "Pipeline Planner is connected, this is overridden by the Pipeline Planner top-p "
                            "schedule."
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
                            "Top-k sampling value for standalone captioning. When a CaptionForge "
                            "Pipeline Planner is connected, this is overridden by the Pipeline Planner top-k "
                            "schedule."
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
                            "Penalty applied to repeated tokens. Values slightly above 1.0 can "
                            "reduce repetitive captions. This is not currently overridden by the "
                            "CaptionForge Pipeline Planner."
                        ),
                    },
                ),

                "captions_per_image": (
                    "INT",
                    {
                        "default": 1,
                        "min": 1,
                        "max": 100,
                        "step": 1,
                        "tooltip": (
                            "Number of caption records per image for standalone captioning. When "
                            "a CaptionForge Pipeline Planner is connected, this is overridden by the Run "
                            "Plan."
                        ),
                    },
                ),

                "seed": (
                    "INT",
                    {
                        "default": -1,
                        "min": -1,
                        "max": 0xFFFFFFFF,
                        "step": 1,
                        "tooltip": (
                            "Random seed for standalone captioning. When a CaptionForge Pipeline Planner "
                            "is connected, this widget is overridden by per-caption-instance seeds "
                            "derived from the Pipeline Planner."
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
                            "Maximum longest-side image size for standalone captioning. The image "
                            "is resized in memory only. When a CaptionForge Pipeline Planner is connected, "
                            "this is overridden by the Pipeline Planner's shared workload guard."
                        ),
                    },
                ),

                "output_dir": (
                    "STRING",
                    {
                        "default": "",
                        "multiline": False,
                        "tooltip": (
                            "Output folder for standalone captioning. When a CaptionForge Pipeline Planner "
                            "is connected, this is overridden by the Pipeline Planner output_dir so all "
                            "engines write to one shared evidence pool."
                        ),
                    },
                ),

                "write_txt": (
                    "BOOLEAN",
                    {
                        "default": True,
                        "tooltip": (
                            "If enabled, write one TXT audit sidecar per image or per ensemble run. "
                            "When a CaptionForge Pipeline Planner is connected, TXT files are audit artifacts; "
                            "JSONL remains the primary evidence output."
                        ),
                    },
                ),

                "write_jsonl": (
                    "BOOLEAN",
                    {
                        "default": False,
                        "tooltip": (
                            "If enabled, append caption records to a JSONL file. When a CaptionForge "
                            "Pipeline Planner is connected, this is forced ON because JSONL evidence is the "
                            "primary CaptionForge Pass A output."
                        ),
                    },
                ),

                "also_jsonl": (
                    "BOOLEAN",
                    {
                        "default": False,
                        "tooltip": (
                            "Standalone compatibility toggle for writing JSONL in addition to TXT. "
                            "When a CaptionForge Pipeline Planner is connected, this is forced OFF because "
                            "write_jsonl is already forced ON."
                        ),
                    },
                ),

                "write_run_config": (
                    "BOOLEAN",
                    {
                        "default": True,
                        "tooltip": (
                            "If enabled, save a small JSON file recording model, prompt, cleanup, "
                            "and generation settings for reproducibility."
                        ),
                    },
                ),

                "jsonl_filename": (
                    "STRING",
                    {
                        "default": "captions.jsonl",
                        "multiline": False,
                        "tooltip": (
                            "Filename to use for JSONL evidence output. In CaptionForge Pipeline Planner "
                            "mode, connected captioners should normally share the same filename "
                            "inside the shared output_dir."
                        ),
                    },
                ),

                "overwrite": (
                    "BOOLEAN",
                    {
                        "default": False,
                        "tooltip": (
                            "Standalone mode: allow existing TXT sidecars to be overwritten. When "
                            "a CaptionForge Pipeline Planner is connected, this is forced ON because TXT "
                            "files are audit artifacts and must not block JSONL evidence generation."
                        ),
                    },
                ),

                "backup_existing": (
                    "BOOLEAN",
                    {
                        "default": True,
                        "tooltip": (
                            "Standalone mode: if overwrite behavior is triggered, existing TXT "
                            "files are first renamed to a timestamped backup. In CaptionForge mode, "
                            "TXT sidecars are audit artifacts and JSONL evidence is primary."
                        ),
                    },
                ),

                "dry_run": (
                    "BOOLEAN",
                    {
                        "default": False,
                        "tooltip": (
                            "If enabled, the node goes through workflow logic without writing TXT, "
                            "JSONL, or run-config files."
                        ),
                    },
                ),

                "limit": (
                    "INT",
                    {
                        "default": 0,
                        "min": 0,
                        "max": 100000,
                        "step": 1,
                        "tooltip": (
                            "Standalone folder mode: maximum number of images to process from "
                            "input_path. Set to 0 for no limit."
                        ),
                    },
                ),

                "skip_existing_txt": (
                    "BOOLEAN",
                    {
                        "default": True,
                        "tooltip": (
                            "Standalone mode: skip images whose TXT sidecar already exists unless "
                            "overwrite is enabled. When a CaptionForge Pipeline Planner is connected, this "
                            "is forced OFF so existing audit TXT files cannot suppress required "
                            "JSONL evidence generation."
                        ),
                    },
                ),

                "skip_existing_jsonl_images": (
                    "BOOLEAN",
                    {
                        "default": False,
                        "tooltip": (
                            "Standalone mode: skip images already present in the target JSONL file. "
                            "When a CaptionForge Pipeline Planner is connected, this is forced OFF so reruns "
                            "with new seeds or schedules append fresh evidence records."
                        ),
                    },
                ),

                "prefix": (
                    "STRING",
                    {
                        "default": "",
                        "multiline": False,
                        "tooltip": (
                            "Optional standalone text prefix for final captions. When a CaptionForge "
                            "Pipeline Planner includes a trigger_word, that trigger is injected ahead of "
                            "this prefix."
                        ),
                    },
                ),

                "suffix": (
                    "STRING",
                    {
                        "default": "",
                        "multiline": False,
                        "tooltip": "Optional text to append to every final caption.",
                    },
                ),

                "forbidden_phrases": (
                    "STRING",
                    {
                        "default": "",
                        "multiline": True,
                        "tooltip": "Optional list of phrases to remove from captions, one per line.",
                    },
                ),

                "replace_pairs": (
                    "STRING",
                    {
                        "default": "",
                        "multiline": True,
                        "tooltip": "Optional search-and-replace rules, one per line, using the format old=>new.",
                    },
                ),

                "keep_loaded": (
                    "BOOLEAN",
                    {
                        "default": True,
                        "tooltip": (
                            "If enabled, keep the loaded model cached in memory for faster repeated "
                            "runs. CaptionForge cache policy may still evict this model when another "
                            "captioning model must load."
                        ),
                    },
                ),

                "download_probe_only": (
                    "BOOLEAN",
                    {
                        "default": False,
                        "tooltip": (
                            "If enabled, do not load the model or caption images. Instead, create "
                            "the expected model folder and download lightweight metadata/config "
                            "files only, skipping large weight files."
                        ),
                    },
                ),
            },
            "optional": {
                "image": (
                    "IMAGE",
                    {
                        "tooltip": (
                            "Optional IMAGE input from the ComfyUI graph. If connected, the node "
                            "captions these image tensors directly. Can be used instead of, or "
                            "together with, input_path."
                        ),
                    },
                ),
                "captionforge_run_config": (
                    "CAPTIONFORGE_PIPELINE_PLAN",
                    {
                        "tooltip": (
                            "Optional shared CaptionForge Pipeline Planner. When connected, it overrides "
                            "captions_per_image, seed behavior, sampling settings, max_size, "
                            "max_new_tokens, trigger prefix, and output_dir. It also forces JSONL "
                            "evidence output and disables TXT/JSONL skip behavior that could "
                            "suppress required CaptionForge evidence records."
                        ),
                    },
                ),
                "extra_options": (
                    "CAPTIONFORGE_EXTRA_OPTIONS",
                    {"tooltip": "Optional shared CaptionForge Extra Options payload. Overrides local extra_option widgets when non-empty."},
                ),
            },
        }

    RETURN_TYPES = ("CAPTIONFORGE_PIPELINE_PLAN", "STRING", "STRING", "STRING")
    RETURN_NAMES = ("captionforge_run_config_out", "caption", "jsonl_records", "resolved_prompt")
    FUNCTION = "caption"
    CATEGORY = "JLC/Captioning"

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        # Captioning should re-run when executed.
        return float("NaN")

    def caption(
        self,
        model,
        qwen_quantization,
        input_path,
        recursive,
        filename_glob,
        prompt_mode,
        caption_type,
        caption_length,
        extra_option1,
        extra_option2,
        extra_option3,
        extra_option4,
        extra_option5,
        extra_option6,
        person_name,
        prompt_preset,
        custom_prompt,
        prompt_file,
        max_new_tokens,
        temperature,
        top_p,
        top_k,
        repetition_penalty,
        captions_per_image,
        seed,
        max_size,
        output_dir,
        write_txt,
        write_jsonl,
        also_jsonl,
        write_run_config,
        jsonl_filename,
        overwrite,
        backup_existing,
        dry_run,
        limit,
        skip_existing_txt,
        skip_existing_jsonl_images,
        prefix,
        suffix,
        forbidden_phrases,
        replace_pairs,
        keep_loaded,
        download_probe_only,
        image=None,
        captionforge_run_config=None,
        extra_options=None,
    ):
        if download_probe_only:
            result = probe_registry_model_download(model, JLC_QWEN_MODEL_ROOT)
            return (captionforge_run_config, result, "", "")

        local_extra_options = _extra_options_from_widgets(
            extra_option1, extra_option2, extra_option3, extra_option4, extra_option5, extra_option6
        )
        effective_extra_options, effective_person_name, extra_options_metadata = resolve_effective_extra_options(
            payload=extra_options,
            local_options=local_extra_options,
            local_name=person_name,
        )
        use_caption_template = prompt_mode == "caption_template"

        if use_caption_template:
            prompt = build_caption_prompt(
                caption_type=caption_type,
                caption_length=caption_length,
                extra_options=effective_extra_options,
                name_input=effective_person_name,
                dialect="qwen",
            )
        else:
            prompt = resolve_prompt(
                prompt=custom_prompt,
                prompt_file=prompt_file,
                prompt_preset=prompt_preset,
            )

        run_plan = expand_captionforge_runs(
            captionforge_run_config,
            model_key="qwen",
            widget_captions_per_image=int(captions_per_image),
            widget_seed=int(seed),
            widget_temperature=float(temperature),
            widget_top_p=float(top_p),
            widget_top_k=int(top_k),
            widget_max_new_tokens=int(max_new_tokens),
            widget_max_size=int(max_size),
            widget_trigger_word="",
            widget_output_dir=output_dir,
            widget_input_path=input_path,
            widget_recursive=bool(recursive),
            widget_filename_glob=filename_glob,
        )

        run_plan_connected = bool(captionforge_run_config)
        if run_plan_connected and not run_plan:
            status = "[CaptionForge] Qwen Caption disabled by Pipeline Planner."
            print(status)
            return (captionforge_run_config, status, "", prompt)

        first_run = run_plan[0]

        if run_plan_connected:
            write_jsonl = True
            also_jsonl = False
            skip_existing_txt = False
            skip_existing_jsonl_images = False
            overwrite = True

        effective_prefix = prefix
        if first_run.trigger_word:
            effective_prefix = f"{first_run.trigger_word}, {prefix}".strip(", ")

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
            prefix=effective_prefix,
            suffix=suffix,
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
            quantization="bnb_8bit" if qwen_quantization == "Balanced (8-bit)" else "none",
            trust_remote_code=True,
            keep_loaded=bool(keep_loaded),
            quiet_transformers_load=True,
            patch_lm_head_weight=True,
            ignore_mismatched_sizes=True,
            max_size=int(first_run.max_size),
            prompt=prompt,
            allow_download=True,
        )

        engine = QwenCaptionEngine(
            config=qwen_config,
            generation=generation,
            cleanup=cleanup,
        )

        input_path = (input_path or "").strip()
        output_dir = (output_dir or "").strip()
        jsonl_filename = (jsonl_filename or "captions.jsonl").strip() or "captions.jsonl"
        planned_caption_jsonl_path = _planned_caption_jsonl_path(captionforge_run_config) if run_plan_connected else None

        if first_run.output_dir:
            output_dir = first_run.output_dir
        if first_run.input_path:
            input_path = first_run.input_path
        if run_plan_connected:
            recursive = bool(first_run.recursive)
            filename_glob = first_run.filename_glob or "*"
            if planned_caption_jsonl_path is not None:
                output_dir = str(planned_caption_jsonl_path.parent)
                jsonl_filename = planned_caption_jsonl_path.name

        use_jsonl = bool(write_jsonl or also_jsonl)

        all_records: list[CaptionRecord] = []

        # -------------------------------------------------------------
        # Direct ComfyUI IMAGE input
        # -------------------------------------------------------------
        pil_images = _tensor_to_pil(image)

        if pil_images and input_path:
            print(
                "[CaptionForge] IMAGE input and input_path are both active; "
                "captioning both sources."
            )

        if pil_images:
            engine.load()

            fallback_dir = Path(folder_paths.get_output_directory()) / "jlc_qwen_caption"
            image_output_dir = Path(output_dir) if output_dir else fallback_dir
            image_output_dir.mkdir(parents=True, exist_ok=True)

            jsonl_path = image_output_dir / jsonl_filename
            seen_jsonl_images: set[str] = set()
            if use_jsonl and skip_existing_jsonl_images:
                seen_jsonl_images = load_existing_jsonl_images(jsonl_path)

            if write_run_config and not input_path:
                write_run_config_json(
                    image_output_dir / f"jlc_qwen_caption_run_config_{timestamp()}.json",
                    engine.build_run_config(),
                    dry_run=bool(dry_run),
                )

            records_to_jsonl: list[CaptionRecord] = []

            for index, pil in enumerate(pil_images):
                source_name = f"comfy_image_{index:04d}"
                txt_path = image_output_dir / f"{source_name}.txt"

                if len(run_plan) == 1:
                    if skip_existing_txt and write_txt and txt_path.exists() and not overwrite:
                        print(f"[JLC Qwen Caption] Skipping existing TXT: {txt_path}")
                        continue

                if use_jsonl and skip_existing_jsonl_images:
                    if source_name in seen_jsonl_images:
                        print(f"[JLC Qwen Caption] Skipping existing JSONL image: {source_name}")
                        continue

                for run in run_plan:
                    engine.generation = GenerationConfig(
                        max_new_tokens=int(run.max_new_tokens),
                        temperature=float(run.temperature),
                        top_p=float(run.top_p),
                        top_k=int(run.top_k),
                        repetition_penalty=float(repetition_penalty),
                        seed=run.seed,
                    )

                    effective_prefix = prefix
                    if run.trigger_word:
                        effective_prefix = f"{run.trigger_word}, {prefix}".strip(", ")

                    engine.cleanup = CleanupConfig(
                        trigger="",
                        prefix=effective_prefix,
                        suffix=suffix,
                        forbidden_phrases=_parse_forbidden_lines(forbidden_phrases),
                        replacement_rules=_parse_replace_pairs(replace_pairs),
                    )
                    engine.config.max_size = int(run.max_size)

                    run_source_name = source_name
                    run_txt_path = txt_path
                    if len(run_plan) > 1:
                        run_txt_path = image_output_dir / f"{source_name}__cf_run_{run.ensemble_run_index:02d}.txt"

                    if skip_existing_txt and write_txt and run_txt_path.exists() and not overwrite:
                        print(f"[JLC Qwen Caption] Skipping existing TXT: {run_txt_path}")
                        continue

                    t0 = time.perf_counter()
                    final_caption, raw_caption = engine.caption_pil(pil)
                    dt = time.perf_counter() - t0
                    print(f"[JLC Qwen Caption] Generation time run {run.ensemble_run_index}: {dt:.2f}s")

                    record = CaptionRecord(
                        image=run_source_name,
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
                        image_key=run_source_name,
                    )

                    all_records.append(record)
                    records_to_jsonl.append(record)

                    if use_jsonl:
                        append_jsonl_records(jsonl_path, [record], dry_run=bool(dry_run))

                    if write_txt:
                        write_text_sidecar(
                            run_txt_path,
                            record.caption,
                            overwrite=bool(overwrite),
                            backup_existing=bool(backup_existing),
                            dry_run=bool(dry_run),
                        )

                    print(
                        f"[JLC Qwen Caption] Captioned IMAGE {index + 1}/{len(pil_images)} "
                        f"run {run.ensemble_run_index + 1}/{len(run_plan)}: {source_name}"
                    )

        # -------------------------------------------------------------
        # File/folder input path
        # -------------------------------------------------------------
        batch_result = None
        if input_path:
            # Match the old node's `also_jsonl` behavior: it writes to the same
            # jsonl_filename target when write_jsonl is off. Avoid duplicate appends
            # when write_jsonl is already true.
            also_jsonl_path = ""
            if also_jsonl and not write_jsonl:
                base_dir = Path(output_dir) if output_dir else (
                    Path(input_path).parent if Path(input_path).is_file() else Path(input_path)
                )
                also_jsonl_path = str(base_dir / jsonl_filename)

            batch = BatchCaptionConfig(
                input_path=input_path,
                recursive=bool(recursive),
                filename_glob=(filename_glob or "*").strip() or "*",
                output_dir=output_dir,
                write_txt=bool(write_txt),
                write_jsonl=bool(write_jsonl),
                jsonl_filename=jsonl_filename,
                also_jsonl_path=also_jsonl_path,
                write_run_config=bool(write_run_config),
                overwrite=bool(overwrite),
                backup_existing=bool(backup_existing),
                dry_run=bool(dry_run),
                limit=int(limit),
                skip_existing_txt=bool(skip_existing_txt),
                skip_existing_jsonl_images=bool(skip_existing_jsonl_images),
            )

            batch_result = engine.caption_batch(batch)
            all_records.extend(batch_result.records)

        if not pil_images and batch_result is None:
            raise RuntimeError(
                "No image input found. Connect an IMAGE input or provide input_path "
                "pointing to an image file or folder."
            )

        if not keep_loaded:
            engine.unload()

        caption_string = "\n\n".join(r.caption for r in all_records if r.status == "ok")
        jsonl_string = _make_jsonl_string(all_records)

        return (captionforge_run_config, caption_string, jsonl_string, prompt)


NODE_CLASS_MAPPINGS = {
    "JLC_QwenCaption": JLC_QwenCaption,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "JLC_QwenCaption": "\u2003JLC Qwen Caption",
}
