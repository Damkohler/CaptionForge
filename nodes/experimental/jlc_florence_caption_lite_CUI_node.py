"""
JLC Florence Caption (Lite) — ComfyUI Node Wrapper

Drop target:
    CaptionForge/nodes/jlc_florence_caption_lite_CUI_node.py

Compact Florence-2 raw-caption witness node for CaptionForge.
"""

from __future__ import annotations

MANIFEST = {
    "name": "JLC Florence Caption (Lite)",
    "version": (0, 1, 0),
    "author": "J. L. Córdova",
    "description": (
        "Compact ComfyUI frontend for Florence-2-family CaptionForge Pass A captioning. "
        "Consumes CAPTIONFORGE_PIPELINE_PLAN objects, writes TXT audit sidecars and shared "
        "JSONL evidence in planned runs, and delegates model loading/generation/cache behavior "
        "to jlc_florence_caption_engine.py."
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

from ..engines.jlc_florence_caption_engine import (
    CaptionRecord,
    CleanupConfig,
    FlorenceCaptionConfig,
    FlorenceCaptionEngine,
    GenerationConfig,
    MEMORY_MODES,
    MODEL_REGISTRY,
    TASK_PROMPTS,
    append_jsonl_records,
    record_to_json,
    resolve_task_prompt,
    timestamp,
    write_run_config_json,
    write_text_sidecar,
)
from ..engines.captionforge_pipeline_planner_engine import expand_captionforge_runs


JLC_FLORENCE_MODEL_ROOT = Path(folder_paths.models_dir) / "LLM" / "JLC_FlorenceCaption"
DEFAULT_JSONL_FILENAME = "captions.jsonl"
_SUPPORTED_IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff"}


def _tensor_to_pil(image_tensor) -> list[Image.Image]:
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
    return "".join(cleaned).strip("._") or "image"


def _iter_input_path_images(input_path: str, recursive: bool, filename_glob: str) -> list[tuple[str, Path]]:
    text = str(input_path or "").strip()
    if not text:
        return []
    root = Path(text)
    if not root.exists():
        raise RuntimeError(f"CaptionForge input_path does not exist: {root}")
    glob_text = (filename_glob or "*").strip() or "*"
    if root.is_file():
        if root.suffix.lower() not in _SUPPORTED_IMAGE_SUFFIXES:
            raise RuntimeError(f"CaptionForge input_path is not a supported image file: {root}")
        return [(_safe_source_name(root.stem), root)]
    pattern_iter = root.rglob(glob_text) if recursive else root.glob(glob_text)
    paths = sorted(p for p in pattern_iter if p.is_file() and p.suffix.lower() in _SUPPORTED_IMAGE_SUFFIXES)
    return [(_safe_source_name(str(p.relative_to(root).with_suffix(""))), p) for p in paths]


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


def _planned_caption_jsonl_path(captionforge_run_config) -> Path | None:
    cfg = _normalize_pipeline_plan(captionforge_run_config)
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


class JLC_FlorenceCaptionLite:
    """Minimal Florence caption node; Run Plan turns it into a CaptionForge evidence node."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": (
                    list(MODEL_REGISTRY.keys()),
                    {
                        "default": "Florence-2-base-ft",
                        "tooltip": "Select Florence-2 model. Models are loaded from ComfyUI/models/LLM/JLC_FlorenceCaption/.",
                    },
                ),
                "memory_mode": (
                    list(MEMORY_MODES.keys()),
                    {
                        "default": "Default",
                        "tooltip": "Florence v1 uses normal Transformers loading; no bitsandbytes path is exposed.",
                    },
                ),
                "keep_loaded": (
                    "BOOLEAN",
                    {
                        "default": True,
                        "tooltip": "Keep Florence cached after captioning. The shared CaptionForge cache may still evict it when another caption model loads.",
                    },
                ),
                "task": (
                    list(TASK_PROMPTS.keys()),
                    {
                        "default": "Detailed Caption",
                        "tooltip": "Florence caption task. Detailed Caption is the suggested CaptionForge Lite default.",
                    },
                ),
                "custom_task_prompt": (
                    "STRING",
                    {
                        "default": "",
                        "multiline": False,
                        "tooltip": "Optional raw Florence task prompt override, such as <MORE_DETAILED_CAPTION>. Leave blank to use the task dropdown.",
                        "advanced": True,
                    },
                ),
                "max_new_tokens": (
                    "INT",
                    {
                        "default": 256,
                        "min": 16,
                        "max": 2048,
                        "step": 8,
                        "tooltip": "Standalone token budget. Run Plan overrides this when connected.",
                    },
                ),
                "num_beams": (
                    "INT",
                    {
                        "default": 3,
                        "min": 1,
                        "max": 16,
                        "step": 1,
                        "tooltip": "Beam count for deterministic Florence generation. 3 is the common Florence example value.",
                    },
                ),
                "temperature": (
                    "FLOAT",
                    {
                        "default": 0.0,
                        "min": 0.0,
                        "max": 2.0,
                        "step": 0.01,
                        "tooltip": "Standalone sampling temperature. 0.0 keeps Florence deterministic. Run Plan overrides this when connected.",
                    },
                ),
                "top_p": (
                    "FLOAT",
                    {
                        "default": 0.90,
                        "min": 0.0,
                        "max": 1.0,
                        "step": 0.01,
                        "tooltip": "Top-p used only when temperature > 0. Run Plan overrides this when connected.",
                    },
                ),
                "top_k": (
                    "INT",
                    {
                        "default": 50,
                        "min": 0,
                        "max": 500,
                        "step": 1,
                        "tooltip": "Top-k used only when temperature > 0. Run Plan overrides this when connected.",
                    },
                ),
            },
            "optional": {
                "image": ("IMAGE", {"tooltip": "Image or batch of images to caption."}),
                "captionforge_run_config": (
                    "CAPTIONFORGE_PIPELINE_PLAN",
                    {
                        "tooltip": (
                            "Optional shared CaptionForge Run Plan. When connected, this Lite node uses "
                            "the florence model-family plan for witness count, routing, seeds/schedules, "
                            "size, token budget, trigger word, and shared JSONL evidence."
                        ),
                    },
                ),
            },
        }

    RETURN_TYPES = ("CAPTIONFORGE_PIPELINE_PLAN", "STRING", "STRING", "STRING")
    RETURN_NAMES = ("captionforge_run_config_out", "caption", "jsonl_records", "resolved_prompt")
    FUNCTION = "caption"
    CATEGORY = "JLC/Captioning"

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        return float("NaN")

    def caption(
        self,
        model,
        memory_mode,
        keep_loaded,
        task,
        custom_task_prompt,
        max_new_tokens,
        num_beams,
        temperature,
        top_p,
        top_k,
        image=None,
        captionforge_run_config=None,
    ):
        run_plan_connected = bool(captionforge_run_config)
        task_prompt = resolve_task_prompt(task, custom_task_prompt)

        run_plan = expand_captionforge_runs(
            captionforge_run_config,
            model_key="florence",
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
            status = "[CaptionForge] Florence Caption Lite disabled by Pipeline Planner."
            print(status)
            return (captionforge_run_config, status, "", task_prompt)

        first_run = run_plan[0]
        generation = GenerationConfig(
            max_new_tokens=int(first_run.max_new_tokens),
            num_beams=int(num_beams),
            temperature=float(first_run.temperature),
            top_p=float(first_run.top_p),
            top_k=int(first_run.top_k),
            seed=first_run.seed,
        )
        cleanup = CleanupConfig(
            trigger="",
            prefix=(f"{first_run.trigger_word}," if first_run.trigger_word else ""),
            suffix="",
            forbidden_phrases=[],
            replacement_rules=[],
        )
        florence_config = FlorenceCaptionConfig(
            model_name=model,
            model_path="",
            model_root=str(JLC_FLORENCE_MODEL_ROOT),
            memory_mode=memory_mode,
            dtype="bf16",
            device="auto",
            trust_remote_code=True,
            keep_loaded=bool(keep_loaded),
            quiet_transformers_load=True,
            max_size=int(first_run.max_size),
            task_prompt=task_prompt,
            allow_download=True,
            use_comfy_model_management=True,
        )
        engine = FlorenceCaptionEngine(config=florence_config, generation=generation, cleanup=cleanup)

        direct_images = [(f"comfy_image_{i:04d}", pil) for i, pil in enumerate(_tensor_to_pil(image))]
        file_images: list[tuple[str, Path]] = []
        if first_run.input_path:
            file_images = _iter_input_path_images(first_run.input_path, first_run.recursive, first_run.filename_glob)

        if direct_images and file_images:
            print("[CaptionForge] IMAGE input and Run Plan input_path are both active; captioning both sources.")
        if not direct_images and not file_images:
            raise RuntimeError("No image input found. Connect an IMAGE input or provide input_path in the CaptionForge Run Plan.")

        planned_caption_jsonl_path = _planned_caption_jsonl_path(captionforge_run_config) if run_plan_connected else None
        output_dir = Path(first_run.output_dir) if first_run.output_dir else Path(folder_paths.get_output_directory()) / "jlc_florence_caption_lite"
        if planned_caption_jsonl_path is not None:
            output_dir = planned_caption_jsonl_path.parent
        if run_plan_connected:
            output_dir.mkdir(parents=True, exist_ok=True)
        jsonl_path = planned_caption_jsonl_path or (output_dir / DEFAULT_JSONL_FILENAME)

        all_records: list[CaptionRecord] = []
        engine.load()

        if run_plan_connected:
            write_run_config_json(output_dir / f"jlc_florence_caption_lite_run_config_{timestamp()}.json", engine.build_run_config(), dry_run=False)

        def process_one(source_name: str, pil: Image.Image):
            for run in run_plan:
                engine.generation = GenerationConfig(
                    max_new_tokens=int(run.max_new_tokens),
                    num_beams=int(num_beams),
                    temperature=float(run.temperature),
                    top_p=float(run.top_p),
                    top_k=int(run.top_k),
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
                engine.config.task_prompt = task_prompt

                t0 = time.perf_counter()
                final_caption, raw_caption = engine.caption_pil(pil)
                dt = time.perf_counter() - t0
                print(f"[JLC Florence Caption Lite] Generation time run {run.ensemble_run_index}: {dt:.2f}s")

                record = CaptionRecord(
                    image=source_name,
                    caption=final_caption,
                    raw_caption=raw_caption,
                    model_name=model,
                    model_path=str(engine.local_model_path or ""),
                    task_prompt=task_prompt,
                    seed=run.seed,
                    temperature=run.temperature,
                    top_p=run.top_p,
                    top_k=run.top_k,
                    num_beams=int(num_beams),
                    max_new_tokens=run.max_new_tokens,
                    max_size=int(run.max_size),
                    timestamp=datetime.now().isoformat(timespec="seconds"),
                    captionforge_pass="A",
                    model_family="florence",
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
                print(f"[JLC Florence Caption Lite] Captioned {source_name} run {run.ensemble_run_index + 1}/{len(run_plan)}")

        for source_name, pil in direct_images:
            process_one(source_name, pil)
        for source_name, path in file_images:
            process_one(source_name, _open_image(path))

        if not keep_loaded:
            engine.unload()

        caption_string = "\n\n".join(r.caption for r in all_records if r.status == "ok")
        jsonl_string = _make_jsonl_string(all_records)
        return (captionforge_run_config, caption_string, jsonl_string, task_prompt)


NODE_CLASS_MAPPINGS = {
    "JLC_FlorenceCaptionLite": JLC_FlorenceCaptionLite,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "JLC_FlorenceCaptionLite": "\u2003JLC Florence Caption (Lite)",
}
