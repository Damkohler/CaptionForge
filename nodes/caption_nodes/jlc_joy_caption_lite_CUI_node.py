"""
JLC CaptionForge Joy (Lite) — ComfyUI Node Wrapper

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
    - The **JLC CaptionForge Joy (Lite)** node provides a compact ComfyUI
      frontend for JoyCaption/LLaVA-family image captioning inside CaptionForge.

    - This file is the **ComfyUI-facing wrapper**, not the reusable captioning
      engine. It is responsible for:
            • compact ComfyUI INPUT_TYPES / widget definitions
            • optional direct IMAGE tensor input
            • IMAGE tensor conversion to PIL images
            • optional Pipeline Planner file/folder `input_path` routing
            • fixed model-root integration under `models/LLM/`
            • minimal model and memory-mode controls
            • clear template-vs-custom prompt controls
            • CaptionForge Pipeline Planner consumption through `pipeline_plan`
            • CaptionForge Template Options consumption through `template_options`
            • TXT audit sidecar writing in planned runs
            • shared JSONL audit output in planned runs
            • ComfyUI output strings for captions and JSONL records
            • node display name, category, and mapping registration
            • passing user settings into the shared Joy caption engine

    - The actual reusable captioning implementation lives in:
            jlc_joy_caption_engine.py

      That engine handles:
            • JoyCaption model registry lookup
            • Hugging Face model probing/downloading
            • processor and model loading
            • Joy/LLaVA generation behavior
            • CaptionForge shared model-cache integration
            • memory-efficient 8-bit loading
            • ComfyUI model-management bridge behavior
            • generation settings
            • caption cleanup
            • TXT sidecar writing
            • JSONL audit output
            • run-configuration export

- CaptionForge Pipeline Role
    - This node participates as a **raw-caption witness** in the CaptionForge
      pipeline.

    - Raw-caption witnesses generate auditable caption evidence records from one
      or more captioning engines.

    - The Joy Lite node contributes JoyCaption/LLaVA-family caption evidence
      compatible with downstream CaptionForge claim extraction, semantic
      synthesis, and final caption construction.

    - CaptionForge audit fields include:
            • captionforge_pass
            • model_family
            • ensemble_run_index
            • image_key
            • raw_caption
            • final cleaned caption
            • generation parameters
            • system prompt and prompt metadata
            • model metadata

- Node Workflow Model
    - The node supports two image sources:
            • direct ComfyUI IMAGE input
            • file/folder routing supplied by the CaptionForge Pipeline Planner

    - These may be used independently or together.

    - In standalone mode, the node behaves as a compact direct-IMAGE captioning
      node and writes to:
            ComfyUI/output/jlc_captionforge_joy_lite/

    - When connected to the CaptionForge Pipeline Planner, this node consumes
      the shared CAPTIONFORGE_PIPELINE_PLAN through the `pipeline_plan` pin and
      uses the Joy model-family plan to determine:
            • whether Joy captioning participates in the run
            • how many Joy raw-caption witnesses to generate per image
            • shared output directory
            • optional shared input path
            • recursive folder traversal
            • filename filtering
            • per-caption-instance seed values
            • per-caption-instance sampling values
            • shared max image size
            • shared max token budget
            • shared LoRA trigger word

    - The node can write:
            • TXT audit sidecar captions
            • JSONL audit records
            • run-configuration JSON files
            • direct ComfyUI string outputs

- Prompting Model
    - The Lite node supports two explicit prompt modes:
            • caption_template_mode
            • custom_prompt_mode

    - `caption_template_mode` uses caption_type, caption_length, and optional
      Template Options supplied by the `template_options` pin.

    - `custom_prompt_mode` uses custom_prompt when provided, otherwise a local
      prompt_preset fallback.

    - If both booleans are enabled, custom_prompt_mode intentionally takes
      precedence. If both are disabled, the node falls back to template mode.

    - Local extra-option widgets have been removed. Shared template modifiers
      now live only in the CaptionForge Template Options sidecar node.

- Model and Dependency Notes
    - JoyCaption/LLaVA repositories are multi-file Hugging Face model
      directories, not single checkpoint files.

    - Runtime behavior depends on the active ComfyUI Python environment,
      PyTorch, Transformers, Accelerate, Pillow, bitsandbytes when 8-bit mode
      is selected, and Hugging Face Hub dependencies.

- Design Philosophy
    - The Lite node is intended to be a compact production-friendly raw-caption
      witness node.

    - It keeps model-family configuration local to the caption node while taking
      shared workflow policy from the CaptionForge Pipeline Planner.

    - CaptionForge is engine-democratic: Joy captions contribute evidence to the
      broader audit and consensus pipeline alongside Qwen, SmolVLM, future local
      VLMs, cleanup LLMs, validators, or other robustness engines.

    - The node prioritizes reproducibility, auditability, low UI clutter, and
      clean separation between the ComfyUI interface and the shared
      model-specific captioning engine.

- ⚠️ Development Status
    - This is early CaptionForge raw-caption witness infrastructure.
    - The UI, model registry, prompt behavior, and output audit fields may
      evolve as the multi-pass CaptionForge pipeline matures.
    - The node is intended for local dataset preparation and controlled caption
      audit workflows.

- Attribution & License
  - Concept and implementation by **J. L. Córdova**
    with development assistance from **ChatGPT (OpenAI)**.

  - CaptionForge's template-option workflow is locally adapted and was inspired
    in part by the practical template interface pattern used by the public
    JoyCaption Beta One Hugging Face Space.

  - JoyCaption/LLaVA model loading is designed around compatible Hugging Face
    Transformers interfaces and publicly available JoyCaption-family
    checkpoints.

  - Designed for use with:
    https://github.com/comfyanonymous/ComfyUI

  - Copyright (c) 2026 J. L. Córdova

  - Released under the **MIT License**.
"""

from __future__ import annotations

MANIFEST = {
    "name": "JLC CaptionForge Joy (Lite)",
    "version": (1, 3, 0),
    "author": "J. L. Córdova",
    "description": (
        "Compact CaptionForge frontend for JoyCaption/LLaVA-family image captioning. "
        "Provides direct IMAGE captioning while also consuming CAPTIONFORGE_PIPELINE_PLAN "
        "objects from the CaptionForge Pipeline Planner through the pipeline_plan input. "
        "Template extras are no longer duplicated as local widgets; they are consumed only "
        "through the template_options input from the CaptionForge Template Options sidecar. "
        "The UI now separates caption_template_mode and custom_prompt_mode for clearer prompt "
        "routing while delegating reusable Joy/LLaVA model loading, generation, memory-efficient "
        "loading, ComfyUI model-management integration, cleanup, cache integration, and audit-record "
        "creation to jlc_joy_caption_engine.py."
    ),
}

import json
from datetime import datetime
from pathlib import Path
import time

import numpy as np
import torch
from PIL import Image

import folder_paths

from ...engines.jlc_joy_caption_engine import (
    CAPTION_LENGTH_CHOICES,
    CAPTION_TYPE_MAP,
    CaptionRecord,
    CleanupConfig,
    GenerationConfig,
    JoyCaptionConfig,
    JoyCaptionEngine,
    MEMORY_EFFICIENT_CONFIGS,
    PROMPT_PRESETS,
    resolve_prompt,
    MODEL_REGISTRY,
    append_jsonl_records,
    record_to_json,
    timestamp,
    write_run_config_json,
    write_text_sidecar,
)
from ...engines.captionforge_pipeline_planner_engine import expand_captionforge_runs
from ..jlc_captionforge_template_options import resolve_effective_extra_options


JLC_JOY_MODEL_ROOT = Path(folder_paths.models_dir) / "LLM" / "JLC_JoyCaption"
DEFAULT_JSONL_FILENAME = "captions.jsonl"
_SUPPORTED_IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff"}

LITE_SYSTEM_PROMPT = (
    "You are a helpful image-captioning assistant. Describe only what is visible "
    "in the image. Do not invent unseen context."
)


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


def _make_jsonl_string(records: list[CaptionRecord]) -> str:
    return "\n".join(json.dumps(record_to_json(r), ensure_ascii=False) for r in records)


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


class JLC_JoyCaptionLite:
    """Compact JoyCaption witness node for CaptionForge."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": (
                    list(MODEL_REGISTRY.keys()),
                    {
                        "default": "llama-joycaption-beta-one-hf-llava",
                        "tooltip": (
                            "Select the JoyCaption/LLaVA-family model. Models are loaded from "
                            "ComfyUI/models/LLM/JLC_JoyCaption/."
                        ),
                    },
                ),
                "memory_mode": (
                    list(MEMORY_EFFICIENT_CONFIGS.keys()),
                    {
                        "default": (
                            "Balanced (8-bit)"
                            if "Balanced (8-bit)" in MEMORY_EFFICIENT_CONFIGS
                            else list(MEMORY_EFFICIENT_CONFIGS.keys())[0]
                        ),
                        "tooltip": (
                            "Joy model memory mode. Balanced (8-bit) uses bitsandbytes load-time "
                            "quantization and is the recommended CaptionForge default for 16 GB "
                            "VRAM systems."
                        ),
                    },
                ),
                "keep_loaded": (
                    "BOOLEAN",
                    {
                        "default": True,
                        "tooltip": (
                            "Keep the model cached after captioning for faster repeated runs. "
                            "CaptionForge cache policy may still evict it when another caption "
                            "model must load."
                        ),
                    },
                ),
                "caption_template_mode": (
                    "BOOLEAN",
                    {
                        "default": True,
                        "tooltip": (
                            "Use the structured CaptionForge template path: caption_type, "
                            "caption_length, and optional Template Options from the template_options pin. "
                            "If custom_prompt_mode is also enabled, custom_prompt_mode takes precedence."
                        ),
                    },
                ),
                "caption_type": (
                    list(CAPTION_TYPE_MAP.keys()),
                    {
                        "default": "JLC LoRA Literal" if "JLC LoRA Literal" in CAPTION_TYPE_MAP else list(CAPTION_TYPE_MAP.keys())[0],
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
                    list(PROMPT_PRESETS.keys()),
                    {
                        "default": "default_literal" if "default_literal" in PROMPT_PRESETS else sorted(PROMPT_PRESETS.keys())[0],
                        "tooltip": "Built-in prompt preset used only in custom_prompt_mode when custom_prompt is blank.",
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
                            "Standalone token budget. When a CaptionForge Pipeline Planner is connected, "
                            "this is overridden by the Pipeline Planner's shared max_new_tokens."
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
                            "Standalone sampling temperature. When a CaptionForge Pipeline Planner is "
                            "connected, this is overridden by the Pipeline Planner temperature schedule."
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
                            "Standalone top-p sampling value. When a CaptionForge Pipeline Planner is "
                            "connected, this is overridden by the Pipeline Planner top-p schedule."
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
                            "Standalone top-k sampling limit. When a CaptionForge Pipeline Planner is "
                            "connected, this is overridden by the Pipeline Planner top-k schedule."
                        ),
                    },
                ),
            },
            "optional": {
                "image": (
                    "IMAGE",
                    {
                        "tooltip": (
                            "Image or batch of images to caption. When a CaptionForge Pipeline Planner "
                            "also supplies input_path, both sources are captioned."
                        ),
                    },
                ),
                "pipeline_plan": (
                    "CAPTIONFORGE_PIPELINE_PLAN",
                    {
                        "tooltip": (
                            "Connect the CaptionForge Pipeline Planner output here. When connected, "
                            "this node switches into Pass A evidence mode: shared input_path/output_dir, "
                            "per-run seeds and schedules, TXT audit sidecars, common captions.jsonl, "
                            "and run config export."
                        ),
                    },
                ),
                "template_options": (
                    "CAPTIONFORGE_EXTRA_OPTIONS",
                    {
                        "tooltip": (
                            "Connect the CaptionForge Template Options node here. This supplies shared "
                            "caption-template modifiers and optional name text. Local extra-option widgets "
                            "were removed so template modifiers have one source of truth."
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

    RETURN_TYPES = ("CAPTIONFORGE_PIPELINE_PLAN", "STRING", "STRING", "STRING")
    RETURN_NAMES = ("pipeline_plan_out", "caption", "jsonl_records", "resolved_prompt")
    FUNCTION = "caption"
    CATEGORY = "JLC/CaptionForge/Caption Nodes"

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        return float("NaN")

    def caption(
        self,
        model,
        memory_mode,
        keep_loaded,
        caption_template_mode,
        caption_type,
        caption_length,
        custom_prompt_mode,
        prompt_preset,
        custom_prompt,
        max_new_tokens,
        temperature,
        top_p,
        top_k,
        image=None,
        pipeline_plan=None,
        template_options=None,
        seed=None,
    ):
        run_plan_connected = bool(pipeline_plan)
        use_caption_template = _use_template_mode(caption_template_mode, custom_prompt_mode)
        effective_extra_options, effective_person_name, _ = _resolve_template_options(template_options)

        prompt = resolve_prompt(
            prompt=custom_prompt if not use_caption_template else "",
            prompt_file="",
            prompt_preset=prompt_preset,
            caption_type=caption_type if use_caption_template else "",
            caption_length=caption_length,
            extra_options=effective_extra_options if use_caption_template else [],
            name_input=effective_person_name if use_caption_template else "",
        )
        effective_seed = -1 if seed is None else int(seed)

        run_plan = expand_captionforge_runs(
            pipeline_plan,
            model_key="joy",
            widget_captions_per_image=1,
            widget_seed=effective_seed,
            widget_temperature=float(temperature),
            widget_top_p=float(top_p),
            widget_top_k=int(top_k),
            widget_max_new_tokens=int(max_new_tokens),
            widget_max_size=1024,
            widget_trigger_word="",
            widget_output_dir="",
            widget_input_path="",
            widget_recursive=True,
            widget_filename_glob="*",
        )

        if run_plan_connected and not run_plan:
            status = "[CaptionForge] Joy Caption Lite disabled by Pipeline Planner."
            print(status)
            return (pipeline_plan, status, "", prompt)

        first_run = run_plan[0]

        generation = GenerationConfig(
            max_new_tokens=int(first_run.max_new_tokens),
            temperature=float(first_run.temperature),
            top_p=float(first_run.top_p),
            top_k=int(first_run.top_k),
            repetition_penalty=1.0,
            seed=first_run.seed,
        )

        cleanup = CleanupConfig(
            trigger="",
            prefix=(f"{first_run.trigger_word}," if first_run.trigger_word else ""),
            suffix="",
            forbidden_phrases=[],
            replacement_rules=[],
        )

        joy_config = JoyCaptionConfig(
            model_name=model,
            model_path="",
            model_root=str(JLC_JOY_MODEL_ROOT),
            memory_mode=memory_mode,
            dtype="bf16",
            device="auto",
            trust_remote_code=True,
            keep_loaded=bool(keep_loaded),
            quiet_transformers_load=True,
            max_size=int(first_run.max_size),
            system_prompt=LITE_SYSTEM_PROMPT,
            prompt=prompt,
            allow_download=True,
            space_compatible_mode=bool(use_caption_template),
            use_comfy_model_management=True,
        )

        engine = JoyCaptionEngine(config=joy_config, generation=generation, cleanup=cleanup)

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
                "No image input found. Connect an IMAGE input or provide input_path in the "
                "CaptionForge Pipeline Planner."
            )

        planned_caption_jsonl_path = _planned_caption_jsonl_path(pipeline_plan) if run_plan_connected else None
        output_dir = Path(first_run.output_dir) if first_run.output_dir else Path(folder_paths.get_output_directory()) / "jlc_captionforge_joy_lite"
        if planned_caption_jsonl_path is not None:
            output_dir = planned_caption_jsonl_path.parent
        if run_plan_connected:
            output_dir.mkdir(parents=True, exist_ok=True)

        jsonl_path = planned_caption_jsonl_path or (output_dir / DEFAULT_JSONL_FILENAME)
        all_records: list[CaptionRecord] = []

        engine.load()

        if run_plan_connected:
            write_run_config_json(
                output_dir / f"jlc_captionforge_joy_lite_run_config_{timestamp()}.json",
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
                    repetition_penalty=1.0,
                    seed=run.seed,
                )
                engine.cleanup = CleanupConfig(
                    trigger="",
                    prefix=(f"{run.trigger_word}," if run.trigger_word else ""),
                    suffix="",
                    forbidden_phrases=[],
                    replacement_rules=[],
                )
                engine.config.max_size = int(run.max_size)

                t0 = time.perf_counter()
                final_caption, raw_caption = engine.caption_pil(pil)
                dt = time.perf_counter() - t0
                print(f"[JLC CaptionForge Joy Lite] Generation time run {run.ensemble_run_index}: {dt:.2f}s")

                record = CaptionRecord(
                    image=source_name,
                    caption=final_caption,
                    raw_caption=raw_caption,
                    model_name=model,
                    model_path=str(engine.local_model_path or ""),
                    prompt=prompt,
                    system_prompt=LITE_SYSTEM_PROMPT,
                    seed=run.seed,
                    temperature=run.temperature,
                    top_p=run.top_p,
                    top_k=run.top_k,
                    max_new_tokens=run.max_new_tokens,
                    max_size=int(run.max_size),
                    timestamp=datetime.now().isoformat(timespec="seconds"),
                    captionforge_pass="A",
                    model_family="joy",
                    ensemble_run_index=run.ensemble_run_index,
                    image_key=source_name,
                )
                all_records.append(record)

                if run_plan_connected:
                    append_jsonl_records(jsonl_path, [record], dry_run=False)
                    write_text_sidecar(
                        _run_txt_path(output_dir, source_name, len(run_plan), run.ensemble_run_index),
                        record.caption,
                        overwrite=True,
                        backup_existing=False,
                        dry_run=False,
                    )
                print(
                    f"[JLC CaptionForge Joy Lite] Captioned {source_name} "
                    f"run {run.ensemble_run_index + 1}/{len(run_plan)}"
                )

        for source_name, pil in direct_images:
            process_one(source_name, pil)

        for source_name, path in file_images:
            process_one(source_name, _open_image(path))

        if not keep_loaded:
            engine.unload()

        caption_string = "\n\n".join(r.caption for r in all_records if r.status == "ok")
        jsonl_string = _make_jsonl_string(all_records)
        return (pipeline_plan, caption_string, jsonl_string, prompt)


NODE_CLASS_MAPPINGS = {
    "JLC_JoyCaptionLite": JLC_JoyCaptionLite,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "JLC_JoyCaptionLite": "\u2003JLC CaptionForge Joy (Lite)",
}
