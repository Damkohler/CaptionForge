"""
JLC Qwen Caption (Lite) — ComfyUI Node Wrapper

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
    - The **JLC Qwen Caption (Lite)** node provides a compact ComfyUI frontend
      for Qwen-family vision-language captioning inside CaptionForge.

    - This file is the **ComfyUI-facing wrapper**, not the reusable captioning
      engine. It is responsible for:
            • compact ComfyUI INPUT_TYPES / widget definitions
            • optional direct IMAGE tensor input
            • IMAGE tensor conversion to PIL images
            • optional Pipeline Planner file/folder `input_path` routing
            • fixed model-root integration under `models/LLM/`
            • minimal model and quantization controls
            • Lite prompt control
            • CaptionForge Pipeline Planner consumption
            • TXT audit sidecar writing in planned runs
            • shared JSONL audit output in planned runs
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

    - The Qwen Lite node contributes Qwen-family caption evidence compatible
      with downstream CaptionForge claim extraction, semantic synthesis, and
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

    - In standalone mode, the node behaves as a compact direct-IMAGE captioning
      node and writes to:
            ComfyUI/output/jlc_qwen_caption_lite/

    - When connected to the CaptionForge Pipeline Planner, this node consumes
      the shared CAPTIONFORGE_PIPELINE_PLAN and uses the Qwen model-family plan
      to determine:
            • whether Qwen captioning participates in the run
            • how many Qwen raw-caption witnesses to generate per image
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
    - The Lite node exposes one compact prompt field.

    - The Lite prompt remains model-specific and belongs to this captioning
      node, while the Pipeline Planner controls shared routing, witness counts,
      seed/sampling policy, image-size guard, token budget, and trigger-word
      routing.

- Model and Dependency Notes
    - Qwen VLM repositories are multi-file Hugging Face model directories, not
      single checkpoint files.

    - Runtime behavior depends on the active ComfyUI Python environment,
      PyTorch, Transformers, Accelerate, Pillow, qwen-vl-utils, bitsandbytes
      when 8-bit mode is selected, and Hugging Face Hub dependencies.

- Design Philosophy
    - The Lite node is intended to be a compact production-friendly raw-caption
      witness node.

    - It keeps model-family configuration local to the caption node while taking
      shared workflow policy from the CaptionForge Pipeline Planner.

    - CaptionForge is engine-democratic: Qwen captions contribute evidence to
      the broader audit and consensus pipeline alongside Joy, future local VLMs,
      cleanup LLMs, validators, or other robustness engines.

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

  - Qwen-family model loading is designed around compatible Hugging Face
    Transformers interfaces and publicly available Qwen-family checkpoints.

  - Designed for use with:
    https://github.com/comfyanonymous/ComfyUI

  - Copyright (c) 2026 J. L. Córdova

  - Released under the **MIT License**.
"""

from __future__ import annotations

MANIFEST = {
    "name": "JLC Qwen Caption (Lite)",
    "version": (1, 1, 1),
    "author": "J. L. Córdova",
    "description": (
        "Compact ComfyUI frontend for Qwen-family vision-language captioning inside "
        "CaptionForge. Provides a minimal direct-IMAGE captioning interface while also "
        "consuming CAPTIONFORGE_PIPELINE_PLAN objects from the CaptionForge Pipeline "
        "Planner. Uses the Qwen model-family plan to determine raw-caption witness count, "
        "shared routing, seed and sampling schedules, image-size guard, token budget, "
        "and trigger-word routing. Writes TXT audit sidecars and shared JSONL evidence "
        "records in planned runs, while delegating reusable Qwen-family model loading, "
        "generation, optional 8-bit loading, cleanup, cache integration, and audit-record "
        "creation to jlc_qwen_caption_engine.py."
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

from ..engines.jlc_qwen_caption_engine import (
    CaptionRecord,
    CleanupConfig,
    GenerationConfig,
    MODEL_REGISTRY,
    QwenCaptionConfig,
    QwenCaptionEngine,
    append_jsonl_records,
    record_to_json,
    timestamp,
    write_run_config_json,
    write_text_sidecar,
)
from ..engines.captionforge_pipeline_planner_engine import expand_captionforge_runs


JLC_QWEN_MODEL_ROOT = Path(folder_paths.models_dir) / "LLM" / "JLC_QwenCaption"
DEFAULT_JSONL_FILENAME = "captions.jsonl"
_SUPPORTED_IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff"}

DEFAULT_LITE_PROMPT = (
    "You are an image captioning assistant. Describe the image in a highly "
    "detailed, literal, and visually grounded way. Focus only on visible details. "
    "Include subject appearance, clothing, pose, body position, hands, facial "
    "expression, hairstyle, accessories, lighting, background, textures, colors, "
    "and spatial relationships. Write a dense descriptive prompt suitable for "
    "image captioning. Avoid speculation, avoid backstory, avoid opinions, and "
    "avoid mentioning things not clearly visible."
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


def _make_jsonl_string(records: list[CaptionRecord]) -> str:
    return "\n".join(json.dumps(record_to_json(r), ensure_ascii=False) for r in records)


def _run_txt_path(output_dir: Path, source_name: str, run_count: int, run_index: int) -> Path:
    if run_count <= 1:
        return output_dir / f"{source_name}.txt"
    return output_dir / f"{source_name}__cf_run_{run_index:02d}.txt"


class JLC_QwenCaptionLite:
    """Minimal Qwen VLM node; Run Plan turns it into a CaptionForge evidence node."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": (
                    list(MODEL_REGISTRY.keys()),
                    {
                        "default": "Qwen2.5-VL-3B-Instruct",
                        "tooltip": (
                            "Select the Qwen vision-language model. Models are loaded from "
                            "ComfyUI/models/LLM/JLC_QwenCaption/."
                        ),
                    },
                ),
                "qwen_quantization": (
                    ["Default", "Balanced (8-bit)"],
                    {
                        "default": "Balanced (8-bit)",
                        "tooltip": (
                            "Qwen model load mode. Balanced (8-bit) uses bitsandbytes 8-bit "
                            "loading to reduce VRAM pressure, especially for Qwen2.5-VL 7B "
                            "variants."
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
                "system_prompt": (
                    "STRING",
                    {
                        "default": DEFAULT_LITE_PROMPT,
                        "multiline": True,
                        "tooltip": (
                            "Lite caption instruction. For Qwen this is passed as the caption "
                            "prompt consumed by the shared engine. Run Plan controls routing, "
                            "seed/sampling schedules, size, token budget, and evidence output; "
                            "this prompt remains the Lite node's model-specific caption style."
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
                            "Standalone token budget. When a CaptionForge Run Plan is connected, "
                            "this is overridden by the Run Plan's shared max_new_tokens."
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
                            "Standalone sampling temperature. When a CaptionForge Run Plan is "
                            "connected, this is overridden by the Run Plan temperature schedule."
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
                            "Standalone top-p sampling value. When a CaptionForge Run Plan is "
                            "connected, this is overridden by the Run Plan top-p schedule."
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
                            "Standalone top-k sampling limit. When a CaptionForge Run Plan is "
                            "connected, this is overridden by the Run Plan top-k schedule."
                        ),
                    },
                ),
            },
            "optional": {
                "image": (
                    "IMAGE",
                    {
                        "tooltip": (
                            "Image or batch of images to caption. When a CaptionForge Run Plan "
                            "also supplies input_path, both sources are captioned."
                        ),
                    },
                ),
                "captionforge_run_config": (
                    "CAPTIONFORGE_PIPELINE_PLAN",
                    {
                        "tooltip": (
                            "Optional shared CaptionForge Run Plan. When connected, this Lite node "
                            "switches into Pass A evidence mode: shared input_path/output_dir, "
                            "per-run seeds and schedules, TXT audit sidecars, common captions.jsonl, "
                            "and run config export."
                        ),
                    },
                ),
            },
        }

    RETURN_TYPES = ("STRING", "STRING", "STRING")
    RETURN_NAMES = ("caption", "jsonl_records", "resolved_prompt")
    FUNCTION = "caption"
    CATEGORY = "JLC/Captioning"

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        return float("NaN")

    def caption(
        self,
        model,
        qwen_quantization,
        keep_loaded,
        system_prompt,
        max_new_tokens,
        temperature,
        top_p,
        top_k,
        image=None,
        captionforge_run_config=None,
    ):
        run_plan_connected = bool(captionforge_run_config)
        prompt = (system_prompt or DEFAULT_LITE_PROMPT).strip() or DEFAULT_LITE_PROMPT

        run_plan = expand_captionforge_runs(
            captionforge_run_config,
            model_key="qwen",
            widget_captions_per_image=1,
            widget_seed=-1,
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
            status = "[CaptionForge] Qwen Caption Lite disabled by Pipeline Planner."
            print(status)
            return (status, "", prompt)

        first_run = run_plan[0]

        qwen_quantization_value = "bnb_8bit" if qwen_quantization == "Balanced (8-bit)" else "none"

        generation = GenerationConfig(
            max_new_tokens=int(first_run.max_new_tokens),
            temperature=float(first_run.temperature),
            top_p=float(first_run.top_p),
            top_k=int(first_run.top_k),
            repetition_penalty=1.08,
            seed=first_run.seed,
        )

        cleanup = CleanupConfig(
            trigger="",
            prefix=(f"{first_run.trigger_word}," if first_run.trigger_word else ""),
            suffix="",
            forbidden_phrases=[],
            replacement_rules=[],
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
                "[CaptionForge] IMAGE input and Run Plan input_path are both active; "
                "captioning both sources."
            )

        if not direct_images and not file_images:
            raise RuntimeError(
                "No image input found. Connect an IMAGE input or provide input_path in the "
                "CaptionForge Run Plan."
            )

        output_dir = Path(first_run.output_dir) if first_run.output_dir else Path(folder_paths.get_output_directory()) / "jlc_qwen_caption_lite"
        if run_plan_connected:
            output_dir.mkdir(parents=True, exist_ok=True)

        jsonl_path = output_dir / DEFAULT_JSONL_FILENAME
        all_records: list[CaptionRecord] = []
        records_to_jsonl: list[CaptionRecord] = []

        engine.load()

        if run_plan_connected:
            write_run_config_json(
                output_dir / f"jlc_qwen_caption_lite_run_config_{timestamp()}.json",
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
                    repetition_penalty=1.08,
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
                print(f"[JLC Qwen Caption Lite] Generation time run {run.ensemble_run_index}: {dt:.2f}s")

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
                
                if run_plan_connected:
                    records_to_jsonl.append(record)
                    append_jsonl_records(jsonl_path, [record], dry_run=False)
                    write_text_sidecar(
                        _run_txt_path(output_dir, source_name, len(run_plan), run.ensemble_run_index),
                        record.caption,
                        overwrite=True,
                        backup_existing=False,
                        dry_run=False,
                    )
                print(
                    f"[JLC Qwen Caption Lite] Captioned {source_name} "
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
        return (caption_string, jsonl_string, prompt)


NODE_CLASS_MAPPINGS = {
    "JLC_QwenCaptionLite": JLC_QwenCaptionLite,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "JLC_QwenCaptionLite": "\u2003JLC Qwen Caption (Lite)",
}
