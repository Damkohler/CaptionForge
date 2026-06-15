"""
JLC Florence Caption — ComfyUI Node Wrapper

Drop target:
    CaptionForge/nodes/jlc_florence_caption_CUI_node.py

Full Florence-2 CaptionForge Pass A node.
"""

from __future__ import annotations

MANIFEST = {
    "name": "JLC Florence Caption",
    "version": (0, 1, 0),
    "author": "J. L. Córdova",
    "description": (
        "ComfyUI frontend for Florence-2-family CaptionForge Pass A captioning. Provides "
        "model/task controls, direct IMAGE and file/folder routing, TXT/JSONL/run-config "
        "outputs, cleanup rules, download probe, Pipeline Planner consumption, and delegates "
        "model loading/generation/cache behavior to jlc_florence_caption_engine.py."
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
    BatchCaptionConfig,
    CaptionRecord,
    CleanupConfig,
    FlorenceCaptionConfig,
    FlorenceCaptionEngine,
    GenerationConfig,
    MEMORY_MODES,
    MODEL_REGISTRY,
    TASK_PROMPTS,
    append_jsonl_records,
    load_existing_jsonl_images,
    probe_registry_model_download,
    record_to_json,
    resolve_task_prompt,
    timestamp,
    write_run_config_json,
    write_text_sidecar,
)
from ..engines.captionforge_pipeline_planner_engine import expand_captionforge_runs


JLC_FLORENCE_MODEL_ROOT = Path(folder_paths.models_dir) / "LLM" / "JLC_FlorenceCaption"


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


def _parse_forbidden_lines(value: str) -> list[str]:
    value = (value or "").strip()
    if not value:
        return []
    return [line.strip() for line in value.splitlines() if line.strip()]


def _parse_replace_pairs(value: str) -> list[tuple[str, str]]:
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


class JLC_FlorenceCaption:
    """Full Florence-2 captioner for CaptionForge."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": (
                    list(MODEL_REGISTRY.keys()),
                    {
                        "default": "Florence-2-base-ft",
                        "tooltip": "Select Florence-2 model. Models live under ComfyUI/models/LLM/JLC_FlorenceCaption/.",
                    },
                ),
                "memory_mode": (
                    list(MEMORY_MODES.keys()),
                    {
                        "default": "Default",
                        "tooltip": "Florence v1 uses normal Transformers loading; no bitsandbytes path is exposed.",
                    },
                ),
                "input_path": (
                    "STRING",
                    {"default": "", "multiline": False, "tooltip": "Optional standalone image file or folder path."},
                ),
                "recursive": (
                    "BOOLEAN",
                    {"default": True, "tooltip": "Standalone folder mode: search subfolders."},
                ),
                "filename_glob": (
                    "STRING",
                    {"default": "*", "multiline": False, "tooltip": "Standalone folder mode filename filter."},
                ),
                "task": (
                    list(TASK_PROMPTS.keys()),
                    {"default": "Detailed Caption", "tooltip": "Florence caption task."},
                ),
                "custom_task_prompt": (
                    "STRING",
                    {
                        "default": "",
                        "multiline": False,
                        "tooltip": "Optional raw Florence task prompt override, e.g. <MORE_DETAILED_CAPTION>.",
                    },
                ),
                "max_new_tokens": (
                    "INT",
                    {"default": 384, "min": 16, "max": 2048, "step": 8, "tooltip": "Max generated tokens. Planner overrides when connected."},
                ),
                "num_beams": (
                    "INT",
                    {"default": 3, "min": 1, "max": 16, "step": 1, "tooltip": "Beam count for Florence generation."},
                ),
                "temperature": (
                    "FLOAT",
                    {"default": 0.0, "min": 0.0, "max": 2.0, "step": 0.01, "tooltip": "Sampling temperature. 0.0 is deterministic."},
                ),
                "top_p": (
                    "FLOAT",
                    {"default": 0.90, "min": 0.0, "max": 1.0, "step": 0.01, "tooltip": "Top-p when sampling."},
                ),
                "top_k": (
                    "INT",
                    {"default": 50, "min": 0, "max": 500, "step": 1, "tooltip": "Top-k when sampling."},
                ),
                "captions_per_image": (
                    "INT",
                    {"default": 1, "min": 1, "max": 100, "step": 1, "tooltip": "Standalone captions per image. Planner overrides when connected."},
                ),
                "seed": (
                    "INT",
                    {"default": -1, "min": -1, "max": 0xFFFFFFFF, "step": 1, "tooltip": "Standalone seed. -1 means unseeded/planner-style behavior."},
                ),
                "max_size": (
                    "INT",
                    {"default": 1024, "min": 0, "max": 4096, "step": 64, "tooltip": "Maximum longest-side image size."},
                ),
                "output_dir": (
                    "STRING",
                    {"default": "", "multiline": False, "tooltip": "Standalone output folder. Planner overrides when connected."},
                ),
                "write_txt": (
                    "BOOLEAN",
                    {"default": True, "tooltip": "Write TXT sidecar captions."},
                ),
                "write_jsonl": (
                    "BOOLEAN",
                    {"default": False, "tooltip": "Append JSONL evidence records."},
                ),
                "also_jsonl": (
                    "BOOLEAN",
                    {"default": False, "tooltip": "Standalone compatibility toggle for JSONL in addition to TXT."},
                ),
                "write_run_config": (
                    "BOOLEAN",
                    {"default": True, "tooltip": "Write run-config JSON."},
                ),
                "jsonl_filename": (
                    "STRING",
                    {"default": "captions.jsonl", "multiline": False, "tooltip": "JSONL evidence filename."},
                ),
                "overwrite": (
                    "BOOLEAN",
                    {"default": False, "tooltip": "Allow overwriting TXT sidecars."},
                ),
                "backup_existing": (
                    "BOOLEAN",
                    {"default": True, "tooltip": "Backup existing TXT before overwrite."},
                ),
                "dry_run": (
                    "BOOLEAN",
                    {"default": False, "tooltip": "Run logic without writing files."},
                ),
                "limit": (
                    "INT",
                    {"default": 0, "min": 0, "max": 100000, "step": 1, "tooltip": "Standalone folder image limit; 0 means no limit."},
                ),
                "skip_existing_txt": (
                    "BOOLEAN",
                    {"default": True, "tooltip": "Skip images whose TXT exists unless overwrite is enabled."},
                ),
                "skip_existing_jsonl_images": (
                    "BOOLEAN",
                    {"default": False, "tooltip": "Skip images already present in JSONL."},
                ),
                "prefix": (
                    "STRING",
                    {"default": "", "multiline": False, "tooltip": "Optional final caption prefix."},
                ),
                "suffix": (
                    "STRING",
                    {"default": "", "multiline": False, "tooltip": "Optional final caption suffix."},
                ),
                "forbidden_phrases": (
                    "STRING",
                    {"default": "", "multiline": True, "tooltip": "Phrases to remove, one per line."},
                ),
                "replace_pairs": (
                    "STRING",
                    {"default": "", "multiline": True, "tooltip": "Search-replace rules, one per line, old=>new."},
                ),
                "keep_loaded": (
                    "BOOLEAN",
                    {"default": True, "tooltip": "Keep Florence cached after run."},
                ),
                "download_probe_only": (
                    "BOOLEAN",
                    {"default": False, "tooltip": "Download metadata/config only; skip large weights and do not caption."},
                ),
            },
            "optional": {
                "image": ("IMAGE", {"tooltip": "Optional ComfyUI IMAGE input."}),
                "captionforge_run_config": (
                    "CAPTIONFORGE_PIPELINE_PLAN",
                    {"tooltip": "Optional shared CaptionForge Pipeline Planner."},
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
        input_path,
        recursive,
        filename_glob,
        task,
        custom_task_prompt,
        max_new_tokens,
        num_beams,
        temperature,
        top_p,
        top_k,
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
    ):
        if download_probe_only:
            result = probe_registry_model_download(model, JLC_FLORENCE_MODEL_ROOT)
            return (captionforge_run_config, result, "", "")

        task_prompt = resolve_task_prompt(task, custom_task_prompt)

        run_plan = expand_captionforge_runs(
            captionforge_run_config,
            model_key="florence",
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
            status = "[CaptionForge] Florence Caption disabled by Pipeline Planner."
            print(status)
            return (captionforge_run_config, status, "", task_prompt)

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
            num_beams=int(num_beams),
            temperature=float(first_run.temperature),
            top_p=float(first_run.top_p),
            top_k=int(first_run.top_k),
            seed=first_run.seed,
        )
        cleanup = CleanupConfig(
            trigger="",
            prefix=effective_prefix,
            suffix=suffix,
            forbidden_phrases=_parse_forbidden_lines(forbidden_phrases),
            replacement_rules=_parse_replace_pairs(replace_pairs),
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

        # Direct IMAGE input
        pil_images = _tensor_to_pil(image)
        if pil_images and input_path:
            print("[CaptionForge] IMAGE input and input_path are both active; captioning both sources.")

        if pil_images:
            engine.load()
            fallback_dir = Path(folder_paths.get_output_directory()) / "jlc_florence_caption"
            image_output_dir = Path(output_dir) if output_dir else fallback_dir
            image_output_dir.mkdir(parents=True, exist_ok=True)
            jsonl_path = image_output_dir / jsonl_filename
            seen_jsonl_images: set[str] = set()
            if use_jsonl and skip_existing_jsonl_images:
                seen_jsonl_images = load_existing_jsonl_images(jsonl_path)

            if write_run_config and not input_path:
                write_run_config_json(
                    image_output_dir / f"jlc_florence_caption_run_config_{timestamp()}.json",
                    engine.build_run_config(),
                    dry_run=bool(dry_run),
                )

            for index, pil in enumerate(pil_images):
                source_name = f"comfy_image_{index:04d}"
                txt_path = image_output_dir / f"{source_name}.txt"
                if len(run_plan) == 1 and skip_existing_txt and write_txt and txt_path.exists() and not overwrite:
                    print(f"[JLC Florence Caption] Skipping existing TXT: {txt_path}")
                    continue
                if use_jsonl and skip_existing_jsonl_images and source_name in seen_jsonl_images:
                    print(f"[JLC Florence Caption] Skipping existing JSONL image: {source_name}")
                    continue

                for run in run_plan:
                    engine.generation = GenerationConfig(
                        max_new_tokens=int(run.max_new_tokens),
                        num_beams=int(num_beams),
                        temperature=float(run.temperature),
                        top_p=float(run.top_p),
                        top_k=int(run.top_k),
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
                    engine.config.task_prompt = task_prompt

                    run_txt_path = txt_path if len(run_plan) <= 1 else image_output_dir / f"{source_name}__cf_run_{run.ensemble_run_index:02d}.txt"
                    if skip_existing_txt and write_txt and run_txt_path.exists() and not overwrite:
                        print(f"[JLC Florence Caption] Skipping existing TXT: {run_txt_path}")
                        continue

                    t0 = time.perf_counter()
                    final_caption, raw_caption = engine.caption_pil(pil)
                    dt = time.perf_counter() - t0
                    print(f"[JLC Florence Caption] Generation time run {run.ensemble_run_index}: {dt:.2f}s")

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

                    if use_jsonl:
                        append_jsonl_records(jsonl_path, [record], dry_run=bool(dry_run))
                    if write_txt:
                        write_text_sidecar(run_txt_path, record.caption, overwrite=bool(overwrite), backup_existing=bool(backup_existing), dry_run=bool(dry_run))
                    print(f"[JLC Florence Caption] Captioned IMAGE {index + 1}/{len(pil_images)} run {run.ensemble_run_index + 1}/{len(run_plan)}: {source_name}")

        # File/folder input path. Note: engine.caption_batch is single-caption-per-image.
        # For planned multi-run file mode, use direct iteration logic in a later patch if needed.
        batch_result = None
        if input_path:
            also_jsonl_path = ""
            if also_jsonl and not write_jsonl:
                base_dir = Path(output_dir) if output_dir else (Path(input_path).parent if Path(input_path).is_file() else Path(input_path))
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
            raise RuntimeError("No image input found. Connect an IMAGE input or provide input_path pointing to an image file or folder.")

        if not keep_loaded:
            engine.unload()

        caption_string = "\n\n".join(r.caption for r in all_records if r.status == "ok")
        jsonl_string = _make_jsonl_string(all_records)
        return (captionforge_run_config, caption_string, jsonl_string, task_prompt)


NODE_CLASS_MAPPINGS = {
    "JLC_FlorenceCaption": JLC_FlorenceCaption,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "JLC_FlorenceCaption": "\u2003JLC Florence Caption",
}
