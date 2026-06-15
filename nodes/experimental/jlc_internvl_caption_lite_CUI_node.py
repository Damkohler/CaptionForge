"""
JLC InternVL Caption (Lite) — ComfyUI Node Wrapper
"""
from __future__ import annotations

MANIFEST = {
    "name": "JLC InternVL Caption (Lite)",
    "version": (0, 1, 0),
    "author": "J. L. Córdova",
    "description": "Compact ComfyUI frontend for InternVL-family CaptionForge Pass A captioning.",
}

import json
from datetime import datetime
from pathlib import Path
import time

import numpy as np
import torch
from PIL import Image

import folder_paths

from ..engines.jlc_internvl_caption_engine import (
    CaptionRecord,
    CleanupConfig,
    InternVLCaptionConfig,
    InternVLCaptionEngine,
    GenerationConfig,
    MEMORY_MODES,
    MODEL_REGISTRY,
    PROMPT_PRESETS,
    append_jsonl_records,
    record_to_json,
    resolve_prompt,
    timestamp,
    write_run_config_json,
    write_text_sidecar,
)
from ..engines.captionforge_pipeline_planner_engine import expand_captionforge_runs

JLC_INTERVLVL_MODEL_ROOT = Path(folder_paths.models_dir) / "LLM" / "JLC_InternVLCaption"
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
        arr = img.numpy(); arr = np.clip(arr * 255.0, 0, 255).astype(np.uint8)
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
    paths = sorted(p for p in pattern_iter if p.is_file() and p.suffix.lower() in _SUPPORTED_IMAGE_SUFFIXES)
    items = []
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

class JLC_InternVLCaptionLite:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": (list(MODEL_REGISTRY.keys()), {"default": "InternVL2.5-2B"}),
                "memory_mode": (list(MEMORY_MODES.keys()), {"default": "Default"}),
                "keep_loaded": ("BOOLEAN", {"default": True}),
                "prompt_preset": (list(PROMPT_PRESETS.keys()), {"default": "detailed"}),
                "custom_prompt": ("STRING", {"default": "", "multiline": True}),
                "max_new_tokens": ("INT", {"default": 256, "min": 16, "max": 2048, "step": 8}),
                "num_beams": ("INT", {"default": 3, "min": 1, "max": 16, "step": 1}),
                "temperature": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 2.0, "step": 0.01}),
                "top_p": ("FLOAT", {"default": 0.90, "min": 0.0, "max": 1.0, "step": 0.01}),
                "top_k": ("INT", {"default": 50, "min": 0, "max": 500, "step": 1}),
            },
            "optional": {
                "image": ("IMAGE", {}),
                "captionforge_run_config": ("CAPTIONFORGE_PIPELINE_PLAN", {}),
            },
        }
    RETURN_TYPES = ("CAPTIONFORGE_PIPELINE_PLAN", "STRING", "STRING", "STRING")
    RETURN_NAMES = ("captionforge_run_config_out", "caption", "jsonl_records", "resolved_prompt")
    FUNCTION = "caption"
    CATEGORY = "JLC/Captioning"
    @classmethod
    def IS_CHANGED(cls, **kwargs):
        return float("NaN")

    def caption(self, model, memory_mode, keep_loaded, prompt_preset, custom_prompt, max_new_tokens, num_beams, temperature, top_p, top_k, image=None, captionforge_run_config=None):
        run_plan_connected = bool(captionforge_run_config)
        prompt = resolve_prompt(prompt_preset, custom_prompt)
        run_plan = expand_captionforge_runs(captionforge_run_config, model_key="internvl", widget_captions_per_image=1, widget_seed=-1, widget_temperature=float(temperature), widget_top_p=float(top_p), widget_top_k=int(top_k), widget_max_new_tokens=int(max_new_tokens), widget_max_size=768, widget_trigger_word="", widget_output_dir="", widget_input_path="", widget_recursive=True, widget_filename_glob="*")
        if run_plan_connected and not run_plan:
            status = "[CaptionForge] JLC InternVL Caption (Lite) disabled by Pipeline Planner."
            return (captionforge_run_config, status, "", prompt)
        first_run = run_plan[0]
        generation = GenerationConfig(max_new_tokens=int(first_run.max_new_tokens), num_beams=int(num_beams), temperature=float(first_run.temperature), top_p=float(first_run.top_p), top_k=int(first_run.top_k), repetition_penalty=1.0, seed=first_run.seed)
        cleanup = CleanupConfig(trigger="", prefix=(f"{first_run.trigger_word}," if first_run.trigger_word else ""), suffix="", forbidden_phrases=[], replacement_rules=[])
        cfg = InternVLCaptionConfig(model_name=model, model_path="", model_root=str(JLC_INTERVLVL_MODEL_ROOT), memory_mode=memory_mode, dtype="bf16", device="auto", trust_remote_code=True, keep_loaded=bool(keep_loaded), quiet_transformers_load=True, max_size=int(first_run.max_size), prompt=prompt, allow_download=True, use_comfy_model_management=True)
        engine = InternVLCaptionEngine(config=cfg, generation=generation, cleanup=cleanup)
        direct_images = [(f"comfy_image_{i:04d}", pil) for i, pil in enumerate(_tensor_to_pil(image))]
        file_images = _iter_input_path_images(first_run.input_path, first_run.recursive, first_run.filename_glob) if first_run.input_path else []
        if not direct_images and not file_images:
            raise RuntimeError("No image input found. Connect an IMAGE input or provide input_path in the CaptionForge Run Plan.")
        planned_caption_jsonl_path = _planned_caption_jsonl_path(captionforge_run_config) if run_plan_connected else None
        output_dir = Path(first_run.output_dir) if first_run.output_dir else Path(folder_paths.get_output_directory()) / "jlc_internvl_caption_lite"
        if planned_caption_jsonl_path is not None:
            output_dir = planned_caption_jsonl_path.parent
        if run_plan_connected:
            output_dir.mkdir(parents=True, exist_ok=True)
        jsonl_path = planned_caption_jsonl_path or (output_dir / DEFAULT_JSONL_FILENAME)
        all_records: list[CaptionRecord] = []
        engine.load()
        if run_plan_connected:
            write_run_config_json(output_dir / f"jlc_internvl_caption_lite_run_config_{timestamp()}.json", engine.build_run_config(), dry_run=False)
        def process_one(source_name: str, pil: Image.Image):
            for run in run_plan:
                engine.generation = GenerationConfig(max_new_tokens=int(run.max_new_tokens), num_beams=int(num_beams), temperature=float(run.temperature), top_p=float(run.top_p), top_k=int(run.top_k), repetition_penalty=1.0, seed=run.seed)
                engine.cleanup = CleanupConfig(trigger="", prefix=(f"{run.trigger_word}," if run.trigger_word else ""), suffix="", forbidden_phrases=[], replacement_rules=[])
                engine.config.max_size = int(run.max_size); engine.config.prompt = prompt
                t0 = time.perf_counter(); final_caption, raw_caption = engine.caption_pil(pil); _ = time.perf_counter() - t0
                rec = CaptionRecord(image=source_name, caption=final_caption, raw_caption=raw_caption, model_name=model, model_path=str(engine.local_model_path or ""), prompt=prompt, seed=run.seed, temperature=run.temperature, top_p=run.top_p, top_k=run.top_k, num_beams=int(num_beams), max_new_tokens=run.max_new_tokens, max_size=int(run.max_size), timestamp=datetime.now().isoformat(timespec="seconds"), captionforge_pass="A", model_family="internvl", ensemble_run_index=run.ensemble_run_index, image_key=source_name)
                all_records.append(rec)
                if run_plan_connected:
                    append_jsonl_records(jsonl_path, [rec], dry_run=False)
                    write_text_sidecar(_run_txt_path(output_dir, source_name, len(run_plan), run.ensemble_run_index), rec.caption, overwrite=True, backup_existing=False, dry_run=False)
        for source_name, pil in direct_images:
            process_one(source_name, pil)
        for source_name, path in file_images:
            process_one(source_name, _open_image(path))
        if not keep_loaded:
            engine.unload()
        caption_string = "\n\n".join(r.caption for r in all_records if r.status == "ok")
        jsonl_string = _make_jsonl_string(all_records)
        return (captionforge_run_config, caption_string, jsonl_string, prompt)

NODE_CLASS_MAPPINGS = {"JLC_InternVLCaptionLite": JLC_InternVLCaptionLite}
NODE_DISPLAY_NAME_MAPPINGS = {"JLC_InternVLCaptionLite": " JLC InternVL Caption (Lite)"}
