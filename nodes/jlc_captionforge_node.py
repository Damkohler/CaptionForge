"""
JLC CaptionForge — ComfyUI Capstone Node Wrapper

- CaptionForge
  - This node is part of CaptionForge, a model-agnostic captioning and
    caption-refinement framework for ComfyUI developed by J. L. Córdova.

  - Repository
    https://github.com/Damkohler/CaptionForge

  - CaptionForge focuses on practical dataset-captioning infrastructure for:
        • LoRA dataset preparation
        • multi-engine raw caption generation
        • JSONL audit trails
        • text-LLM caption distillation
        • image-aware VLM validation
        • final TXT sidecar export for training datasets

- Node Purpose
    - The JLC CaptionForge capstone node orchestrates the current final pipeline:
            A_RAW_CAPTIONS JSONL
              -> CaptionForge Distiller Engine
              -> CaptionForge VLM Validator Engine
              -> deterministic final TXT/JSONL export

    - This file is the ComfyUI-facing wrapper, not the reusable distiller or VLM
      validator engine. It is responsible for:
            • ComfyUI INPUT_TYPES / widget definitions
            • standalone captions JSONL + image-root execution
            • CAPTIONFORGE_PIPELINE_PLAN consumption
            • planner-owned override resolution
            • optional direct IMAGE tensor handoff for validator resolution
            • Distiller config construction
            • VLM Validator config construction
            • final caption selection and TXT/JSONL export
            • output path derivation and run audit status strings

    - The reusable implementation stages live in:
            captionforge_distiller_engine.py
            captionforge_vlm_validator_engine.py
            captionforge_pipeline_planner_engine.py

- Ollama Model Dropdowns
    - Distiller and Validator dropdown values are explicit Ollama model tags.
    - Family aliases and shorthand substitutions are intentionally not used.
    - Dropdown choices are loaded at node-import time from:
            config/captionforge_ollama_models.json
    - If the JSON file is missing or malformed, the node falls back to:
            Distiller: llama3.1:8b
            Validator: gemma4:e4b
    - The optional custom choice lets users enter any installed Ollama model tag
      without editing Python.

- CaptionForge Pipeline Role
    - In planned mode, this node consumes the Pipeline Planner object and lets
      planner-owned values override matching visible widgets.
    - In standalone mode, it can run directly from a Pass A captions JSONL and
      an image file/folder root.
    - Final TXT sidecars preserve user-facing source filename stems when
      possible, while intermediate audit keys may remain sanitized.

- Design Philosophy
    - CaptionForge is an original concept and implementation, not derived from
      or based on another ComfyUI workflow.
    - The capstone node keeps orchestration and ComfyUI UI concerns separate
      from reusable engine logic.
    - The node prioritizes auditable local caption refinement, explicit model
      selection, deterministic output paths, and user-editable validation policy.

- ⚠️ Development Status
    - This is release-candidate CaptionForge capstone infrastructure.
    - Validator backend reliability and output schema details may evolve as the
      release candidate is tested across local Ollama/VLM installations.

- Attribution & License
  - Concept and implementation by J. L. Córdova
    with development assistance from ChatGPT (OpenAI).
  - Designed for use with:
    https://github.com/comfyanonymous/ComfyUI
  - Copyright (c) 2026 J. L. Córdova
  - Released under the MIT License.
"""

from __future__ import annotations

MANIFEST = {
    "name": "JLC CaptionForge Node",
    "version": (0, 1, 7),
    "author": "J. L. Córdova",
    "description": (
        "Heavy ComfyUI capstone node for CaptionForge. Consumes the optional "
        "pipeline_plan pin when connected, then runs captions JSONL "
        "through the pollster/copywriter distiller engine, VLM validator engine, "
        "and deterministic final export. Loads explicit Ollama Distiller/Validator dropdown tags "
        "from config/captionforge_ollama_models.json, with no family aliases "
        "or shorthand model substitutions."
    ),
}

import json
import random
import re
from datetime import datetime
from pathlib import Path, PureWindowsPath

import numpy as np
import torch
from PIL import Image
from typing import Any

try:
    from .captionforge_ollama_model_dropdowns import load_ollama_model_dropdowns
except Exception:  # pragma: no cover - useful for direct local smoke tests
    from captionforge_ollama_model_dropdowns import load_ollama_model_dropdowns

try:
    import folder_paths
except Exception:  # pragma: no cover - useful outside ComfyUI tests
    folder_paths = None

from ..engines.captionforge_distiller_engine import (
    BatchConfig as DistillerBatchConfig,
    DEFAULT_DISTILLER_INSTRUCTIONS,
    DistillerConfig,
    process_batch as run_distiller_batch,
)
from ..engines.captionforge_vlm_validator_engine import (
    BatchVLMValidatorConfig,
    VLMValidatorConfig,
    extract_validate_batch,
    read_jsonl as read_validator_jsonl,
)

try:
    from ..engines.captionforge_pipeline_planner_engine import (
        MAX_SEED_32,
        normalize_captionforge_pipeline_plan,
    )
except Exception:  # pragma: no cover
    MAX_SEED_32 = 0xFFFFFFFF

    def normalize_captionforge_pipeline_plan(config: Any) -> dict[str, Any]:
        if isinstance(config, dict):
            return dict(config)
        if isinstance(config, str) and config.strip():
            try:
                obj = json.loads(config)
                return obj if isinstance(obj, dict) else {}
            except Exception:
                return {}
        return {}


CAPTIONFORGE_NODE_VERSION = "0.1.7"
SEED_MODES = ["fixed", "increment", "decrement", "random"]

_MODEL_DROPDOWNS = load_ollama_model_dropdowns(__file__)
DISTILLER_MODEL_CHOICES = _MODEL_DROPDOWNS["distiller_models"]
VALIDATOR_MODEL_CHOICES = _MODEL_DROPDOWNS["validator_models"]
DEFAULT_DISTILLER_MODEL = _MODEL_DROPDOWNS["distiller_default"]
DEFAULT_VALIDATOR_MODEL = _MODEL_DROPDOWNS["validator_default"]

DISTILLER_STRATEGIES = ["single_pass", "by_model_then_global"]
FINAL_CAPTION_STYLES = ["narrative", "comma", "both"]

DEFAULT_VALIDATOR_PROMPT = (
    "You are CaptionForge Pass C: an image-grounded rich-caption validator and copywriter. "
    "You see the actual image and a Pass B pollster record. Pass B contains accepted claims, "
    "plausible singleton candidates, rejected or unresolved conflicts, and rich/taggy draft captions. "
    "Your job is not to summarize. Your job is to ground the evidence against the image, keep all "
    "supported useful details, correct visibly wrong details, visually confirm useful singletons when "
    "possible, add clearly visible missing details when they improve LoRA training value, and write a "
    "rich final caption. Use no deterministic semantic taxonomy. Judge natural-language claims only. "
    "Reject details only when they are visibly false, contradicted, not visible enough, or inappropriate "
    "as a training caption claim. Preserve accurate outfit construction, materials, jewelry, accessories, "
    "makeup, hair, eye details, body pose, hand placement, crop, lighting, background, and visible texture. "
    "Do not over-prune. Do not compress detailed jewelry, makeup, fabric, pose, or material evidence into "
    "generic phrases. Treat trigger words as metadata and do not prepend them yourself; the engine will "
    "prepend trigger and anchor after parsing. Treat the user caption anchor as training guidance: preserve "
    "it when compatible with the image and reject it only when clearly contradicted. The final narrative "
    "caption should be rich LoRA-training prose. The final comma caption should be a dense taggy caption "
    "with nearly the same visual content."
)


# -----------------------------------------------------------------------------
# Generic helpers
# -----------------------------------------------------------------------------


def _now_stamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def _default_output_dir() -> str:
    if folder_paths is not None:
        try:
            return str(Path(folder_paths.get_output_directory()) / "CaptionForge")
        except Exception:
            pass
    return str(Path.cwd() / "output" / "CaptionForge")


def _normalize_text(value: Any) -> str:
    return re.sub(r"\s+", " ", str(value or "").replace("\n", " ")).strip()


def _clean_run_name(value: Any) -> str:
    text = str(value or "captionforge_run").strip()
    text = re.sub(r"[^A-Za-z0-9_.-]+", "_", text).strip("._")
    return text or "captionforge_run"


def _safe_bool(value: Any, default: bool = False) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return default
    text = str(value).strip().lower()
    if text in {"1", "true", "yes", "y", "on"}:
        return True
    if text in {"0", "false", "no", "n", "off"}:
        return False
    return default


def _json_safe(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    return value


def _deep_get(data: Any, dotted_key: str, default: Any = None) -> Any:
    cur = data
    for part in dotted_key.split("."):
        if not isinstance(cur, dict) or part not in cur:
            return default
        cur = cur[part]
    return cur


def _plan_get(plan: dict[str, Any], *keys: str, default: Any = None) -> Any:
    if not isinstance(plan, dict):
        return default
    for key in keys:
        if "." in key:
            value = _deep_get(plan, key, default=None)
        else:
            value = plan.get(key)
        if value not in (None, ""):
            return value
    return default


def _planner_overrides(plan: dict[str, Any]) -> bool:
    return bool(plan and str(plan.get("captionforge_config_type", "")).lower() in {"captionforge_pipeline_plan", "captionforge_run_config"})


def _coerce_int(value: Any, default: int, min_value: int | None = None, max_value: int | None = None) -> int:
    try:
        out = int(value)
    except Exception:
        out = int(default)
    if min_value is not None:
        out = max(min_value, out)
    if max_value is not None:
        out = min(max_value, out)
    return out


def _coerce_float(value: Any, default: float, min_value: float | None = None, max_value: float | None = None) -> float:
    try:
        out = float(value)
    except Exception:
        out = float(default)
    if min_value is not None:
        out = max(min_value, out)
    if max_value is not None:
        out = min(max_value, out)
    return out


def _seed_for_stage(base_seed: Any, seed_mode: Any, stage_index: int) -> int | None:
    base = _coerce_int(base_seed, -1, -1, MAX_SEED_32)
    mode = str(seed_mode or "fixed").strip().lower()
    if mode not in {"fixed", "increment", "decrement", "random"}:
        mode = "fixed"
    if base < 0:
        return None
    if mode == "fixed":
        return base
    if mode == "increment":
        return min(MAX_SEED_32, base + stage_index)
    if mode == "decrement":
        return max(0, base - stage_index)
    rng = random.Random(base)
    out = base
    for _ in range(stage_index + 1):
        out = rng.randint(0, MAX_SEED_32)
    return out


def _set_if_present(obj: Any, attr_names: tuple[str, ...], value: Any) -> None:
    if value is None:
        return
    for name in attr_names:
        if hasattr(obj, name):
            setattr(obj, name, value)


def _resolve_ollama_model_name(value: Any, custom_value: Any, fallback: str) -> str:
    """Resolve a concrete Ollama dropdown value or a custom model tag.

    Dropdown values are explicit Ollama tags and are used exactly as written.
    No family aliases or shorthand substitutions are applied.
    """
    text = str(value or "").strip()
    custom = str(custom_value or "").strip()
    if text.lower() == "custom":
        return custom or fallback
    return text or custom or fallback


def _load_prompt_file_if_path(text: str) -> str:
    """Treat a non-empty existing file path as prompt-file shorthand; otherwise return text."""
    text = str(text or "")
    stripped = text.strip()
    if not stripped:
        return ""
    try:
        p = Path(stripped)
        if p.exists() and p.is_file():
            return p.read_text(encoding="utf-8")
    except Exception:
        pass
    return text


def _upgrade_legacy_distiller_prompt(text: str) -> str:
    """Replace old saved default prompt text with the current pollster contract.

    ComfyUI workflows persist widget text. Without this guard, an old dropped
    CaptionForge node can keep using the pre-v0.2 summarizing distiller prompt
    even after the Python default changes.
    """
    value = str(text or "")
    legacy_markers = (
        "two deliberately over-complete captions",
        "distilled_caption_narrative",
        "distilled_caption_comma",
    )
    if all(marker in value for marker in legacy_markers):
        return DEFAULT_DISTILLER_INSTRUCTIONS
    return value


def _upgrade_legacy_validator_prompt(text: str) -> str:
    """Replace old saved default validator prompt text with the current rich validator contract."""
    value = str(text or "")
    legacy_markers = (
        "two over-complete draft captions produced by a text distiller",
        "clean, useful final caption candidates",
        "removed_or_rejected_details field",
    )
    if all(marker in value for marker in legacy_markers):
        return DEFAULT_VALIDATOR_PROMPT
    return value


def _tensor_to_pil_images(image_tensor: Any) -> list[Image.Image]:
    """Convert an optional ComfyUI IMAGE tensor to PIL RGB images for validator file handoff."""
    if image_tensor is None:
        return []
    if isinstance(image_tensor, torch.Tensor):
        image_tensor = image_tensor.detach().cpu()
    if getattr(image_tensor, "ndim", 0) == 3:
        image_tensor = image_tensor.unsqueeze(0)
    images: list[Image.Image] = []
    try:
        for img in image_tensor:
            arr = img.numpy()
            arr = np.clip(arr * 255.0, 0, 255).astype(np.uint8)
            images.append(Image.fromarray(arr).convert("RGB"))
    except Exception:
        return []
    return images


def _save_single_image_inputs_for_validator(image_tensor: Any, output_dir: Path, run_name: str) -> str:
    """
    Save optional ComfyUI IMAGE input(s) into the run output folder so the
    file-based VLM validator can resolve direct-IMAGE Pass A records.

    Current Joy/Qwen direct tensor records use image/image_key values like
    ``comfy_image_0000``. The validator resolves exact basenames under
    ``image_root``, so this helper writes an extensionless PNG payload with
    that exact stem for compatibility, plus a human-viewable .png sibling.

    Only the extensionless file is required for resolver compatibility; the
    .png file is an audit/convenience copy.
    """
    images = _tensor_to_pil_images(image_tensor)
    if not images:
        return ""

    image_dir = output_dir / "opt_images"
    image_dir.mkdir(parents=True, exist_ok=True)

    for index, pil in enumerate(images):
        stem = f"comfy_image_{index:04d}"
        resolver_target = image_dir / stem
        audit_target = image_dir / f"{stem}.png"
        pil.save(resolver_target, format="PNG")
        pil.save(audit_target, format="PNG")

    print(f"[JLC CaptionForge Node] Saved optional IMAGE input(s) for VLM validator: {image_dir}", flush=True)
    return str(image_dir)



# -----------------------------------------------------------------------------
# Plan resolution helpers
# -----------------------------------------------------------------------------


def _resolve_output_dir(widget_value: str, plan: dict[str, Any]) -> Path:
    if _planner_overrides(plan):
        planned = _plan_get(plan, "paths.output_dir", "shared.output_dir", "output_dir", default="")
        if str(planned or "").strip():
            return Path(str(planned).strip())
    widget_value = str(widget_value or "").strip()
    return Path(widget_value or _default_output_dir())


def _resolve_run_name(widget_value: str, plan: dict[str, Any]) -> str:
    if _planner_overrides(plan):
        planned = _plan_get(plan, "paths.run_name", "shared.run_name", "run_name", default="")
        if str(planned or "").strip():
            return _clean_run_name(planned)
    return _clean_run_name(widget_value or "captionforge_run")


def _derive_paths(output_dir: Path, run_name: str, plan: dict[str, Any]) -> dict[str, str]:
    planned_paths = _plan_get(plan, "paths", default={}) if _planner_overrides(plan) else {}
    planned_paths = planned_paths if isinstance(planned_paths, dict) else {}

    def pick(name: str, fallback: Path) -> str:
        value = planned_paths.get(name)
        return str(value).strip() if str(value or "").strip() else str(fallback)

    return {
        "output_dir": pick("output_dir", output_dir),
        "run_name": run_name,
        "caption_jsonl": pick("caption_jsonl", output_dir / f"{run_name}__A_RAW_CAPTIONS.jsonl"),
        "pass_a_jsonl": pick("pass_a_jsonl", output_dir / f"{run_name}__A_RAW_CAPTIONS.jsonl"),
        "distiller_jsonl": pick("distiller_jsonl", output_dir / f"{run_name}__B_DISTILL.jsonl"),
        "distiller_readable_jsonl": pick("distiller_readable_jsonl", output_dir / f"{run_name}__B_DISTILL_readable.jsonl"),
        "distiller_readable_json": pick("distiller_readable_json", output_dir / f"{run_name}__B_DISTILL_readable.json"),
        "distiller_prompt_jsonl": pick("distiller_prompt_jsonl", output_dir / f"{run_name}__B_DISTILL_prompts.jsonl"),
        "validator_jsonl": pick("validator_jsonl", output_dir / f"{run_name}__C_VLM_VALIDATED.jsonl"),
        "validator_prompt_jsonl": pick("validator_prompt_jsonl", output_dir / f"{run_name}__C_VLM_VALIDATOR_prompts.jsonl"),
        "validator_readable_dir": pick("validator_readable_dir", output_dir / f"{run_name}__C_VLM_VALIDATED_readable"),
        "final_jsonl": pick("final_jsonl", output_dir / f"{run_name}__D_FINAL_EXPORT.jsonl"),
        "final_txt_dir": pick("final_txt_dir", output_dir / f"{run_name}__TXT"),
        "output_paths_json": pick("output_paths_json", output_dir / f"{run_name}__output_paths.json"),
    }


def _resolve_caption_jsonl(widget_value: str, plan: dict[str, Any], paths: dict[str, str]) -> str:
    if _planner_overrides(plan):
        planned = _plan_get(
            plan,
            "paths.caption_jsonl",
            "paths.pass_a_jsonl",
            "shared.caption_jsonl",
            "shared.pass_a_jsonl",
            default="",
        )
        if str(planned or "").strip():
            return str(planned).strip()
    widget_value = str(widget_value or "").strip()
    if widget_value:
        widget_path = Path(widget_value)
        if widget_path.is_absolute():
            return str(widget_path)
        output_root = Path(str(paths.get("output_dir") or "").strip() or ".")
        return str(output_root / widget_path)
    return paths["caption_jsonl"] or paths["pass_a_jsonl"]


def _resolve_image_root(widget_value: str, plan: dict[str, Any], caption_jsonl: str) -> str:
    if _planner_overrides(plan):
        planned = _plan_get(
            plan,
            "paths.image_root",
            "shared.image_root",
            "shared.input_path",
            "input_path",
            default="",
        )
        if str(planned or "").strip():
            return str(planned).strip()
    widget_value = str(widget_value or "").strip()
    if widget_value:
        return widget_value
    try:
        return str(Path(caption_jsonl).resolve().parent)
    except Exception:
        return ""


def _resolve_prompt(widget_value: str, plan: dict[str, Any], plan_keys: tuple[str, ...], default_prompt: str) -> str:
    if _planner_overrides(plan):
        planned = _plan_get(plan, *plan_keys, default="")
        if str(planned or "").strip():
            return _load_prompt_file_if_path(str(planned))
    widget_prompt = _load_prompt_file_if_path(str(widget_value or ""))
    return widget_prompt if widget_prompt.strip() else default_prompt


def _resolve_setting(plan: dict[str, Any], widget_value: Any, *plan_keys: str, default: Any = None) -> Any:
    if _planner_overrides(plan):
        value = _plan_get(plan, *plan_keys, default=None)
        if value not in (None, ""):
            return value
    return widget_value if widget_value not in (None, "") else default


# -----------------------------------------------------------------------------
# Final export helpers
# -----------------------------------------------------------------------------


def _basename_cross_platform(value: Any) -> str:
    text = str(value or "").strip()
    if not text:
        return ""
    return PureWindowsPath(text).name if "\\" in text else Path(text).name



def _safe_txt_stem_for_source_filename(stem: Any) -> str:
    """
    Preserve ordinary user filename stems for LoRA TXT sidecars.

    Spaces, commas, parentheses, and similar common filename characters are kept
    so final TXT sidecars can match source image stems. Only Windows-forbidden
    path characters/control characters are replaced.
    """
    text = str(stem or "").strip()
    if not text:
        return "image"
    text = re.sub(r'[<>:"/\\|?*\x00-\x1f]+', "_", text)
    text = text.rstrip(" .")
    return text or "image"


def _txt_stem_from_record(record: dict[str, Any]) -> str:
    candidates = [
        record.get("image_resolved_path"),
        record.get("image"),
        record.get("image_key"),
    ]
    source = record.get("source_distiller")
    if isinstance(source, dict):
        candidates.extend([source.get("image"), source.get("image_key")])
    for candidate in candidates:
        base = _basename_cross_platform(candidate)
        if base:
            stem = Path(base).stem or base
            clean = _safe_txt_stem_for_source_filename(stem)
            if clean:
                return clean
    return "image"


def _select_validated_caption(record: dict[str, Any], caption_style: str) -> str:
    narrative = _normalize_text(record.get("validated_caption_narrative"))
    comma = _normalize_text(record.get("validated_caption_comma"))
    style = str(caption_style or "narrative").strip().lower()
    if style == "comma":
        return comma or narrative
    if style == "both":
        if narrative and comma and narrative != comma:
            return f"{narrative}\n{comma}"
        return narrative or comma
    return narrative or comma


def _write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(str(text).rstrip() + "\n", encoding="utf-8")


def _write_final_outputs(
    *,
    validator_jsonl: Path,
    final_jsonl: Path,
    final_txt_dir: Path,
    caption_style: str,
    write_txt: bool,
    write_final_jsonl: bool,
) -> tuple[list[dict[str, Any]], str, int, int]:
    records = read_validator_jsonl(validator_jsonl)
    final_records: list[dict[str, Any]] = []
    final_caption_blocks: list[str] = []
    ok = 0
    failed = 0

    if write_final_jsonl:
        final_jsonl.parent.mkdir(parents=True, exist_ok=True)
        final_jsonl.write_text("", encoding="utf-8")

    for record in records:
        status = str(record.get("status") or "").lower()
        image_key = str(record.get("image_key") or record.get("image") or "image")
        final_caption = _select_validated_caption(record, caption_style)
        is_ok = status in {"ok", "prompt_only"} and bool(final_caption)
        ok += int(is_ok)
        failed += int(not is_ok)

        out_record = {
            "captionforge_pass": "D_FINAL_EXPORT",
            "engine": "jlc_captionforge_node",
            "engine_version": CAPTIONFORGE_NODE_VERSION,
            "image_key": image_key,
            "image": record.get("image", image_key),
            "status": "ok" if is_ok else "error",
            "caption_style": caption_style,
            "final_caption": final_caption if is_ok else "",
            "validated_caption_narrative": record.get("validated_caption_narrative", ""),
            "validated_caption_comma": record.get("validated_caption_comma", ""),
            "trigger_word": record.get("trigger_word", ""),
            "user_caption_anchor": record.get("user_caption_anchor", ""),
            "source_vlm_validator": {
                "captionforge_pass": record.get("captionforge_pass", ""),
                "status": record.get("status", ""),
                "image_resolved_path": record.get("image_resolved_path", ""),
                "removed_or_rejected_details": record.get("removed_or_rejected_details", []),
                "corrected_details": record.get("corrected_details", []),
                "uncertain_details": record.get("uncertain_details", []),
                "visual_validation_notes": record.get("visual_validation_notes", []),
                "params": record.get("params", {}),
                "metrics": record.get("metrics", {}),
            },
            "timestamp": datetime.now().isoformat(timespec="seconds"),
        }
        final_records.append(out_record)

        if is_ok:
            final_caption_blocks.append(final_caption)
            if write_txt:
                _write_text(final_txt_dir / f"{_txt_stem_from_record(record)}.txt", final_caption)

        if write_final_jsonl:
            with final_jsonl.open("a", encoding="utf-8", newline="\n") as f:
                f.write(json.dumps(_json_safe(out_record), ensure_ascii=False) + "\n")

    return final_records, "\n\n".join(final_caption_blocks), ok, failed


# -----------------------------------------------------------------------------
# ComfyUI node
# -----------------------------------------------------------------------------


class JLC_CaptionForge:
    """Heavy CaptionForge capstone node."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                # Inputs first.
                "Input - captions JSONL": (
                    "STRING",
                    {
                        "default": "",
                        "multiline": False,
                        "tooltip": "Raw caption JSONL produced by CaptionForge captioning nodes.",
                    },
                ),
                "Input - image path": (
                    "STRING",
                    {
                        "default": "",
                        "multiline": False,
                        "tooltip": "Image file/folder root used by the VLM validator to resolve source images.",
                    },
                ),


                # Outputs next.
                "Output - folder": (
                    "STRING",
                    {
                        "default": "",
                        "multiline": False,
                        "tooltip": "Output folder. Planner value overrides this when pipeline_plan is connected.",
                    },
                ),
                "Output - run name": (
                    "STRING",
                    {
                        "default": "captionforge_run",
                        "multiline": False,
                        "tooltip": "Run-root used for B/C/D JSONL and prompt/output path files.",
                    },
                ),
                "Output - overwrite outputs": (
                    "BOOLEAN",
                    {"default": True},
                ),

                # LoRA metadata.
                "LoRA - trigger word": (
                    "STRING",
                    {"default": "", "multiline": False},
                ),
                "LoRA - user caption anchor": (
                    "STRING",
                    {"default": "", "multiline": False},
                ),

                # Distiller controls.
                "Distiller - enabled": (
                    "BOOLEAN",
                    {
                        "default": True,
                        "tooltip": (
                            "Run the Distiller stage. Disable this to reuse an existing "
                            "<run>__B_DISTILL.jsonl from the current Output folder and run name."
                        ),
                    },
                ),
                "Distiller - model": (
                    DISTILLER_MODEL_CHOICES,
                    {"default": DEFAULT_DISTILLER_MODEL},
                ),
                "Distiller - custom Ollama model": (
                    "STRING",
                    {
                        "default": "",
                        "multiline": False,
                        "tooltip": "Used only when Distiller - model is custom. Enter an installed Ollama text model tag, for example gpt-oss:20b.",
                    },
                ),
                "Distiller - base seed": (
                    "INT",
                    {"default": -1, "min": -1, "max": MAX_SEED_32, "step": 1},
                ),
                "Distiller - seed mode": (
                    SEED_MODES,
                    {"default": "fixed"},
                ),
                "Distiller - strategy": (
                    DISTILLER_STRATEGIES,
                    {"default": "single_pass"},
                ),
                "Distiller - prompt": (
                    "STRING",
                    {
                        "default": DEFAULT_DISTILLER_INSTRUCTIONS,
                        "multiline": True,
                        "tooltip": "Distiller prompt/instructions. Existing file path is also accepted as shorthand.",
                    },
                ),
                "Distiller - max caption chars for LLM": (
                    "INT",
                    {"default": 1536, "min": 0, "max": 12000, "step": 64},
                ),
                "Distiller - num predict": (
                    "INT",
                    {"default": 3096, "min": 64, "max": 12000, "step": 64},
                ),
                "Distiller - temperature": (
                    "FLOAT",
                    {"default": 0.24, "min": 0.0, "max": 2.0, "step": 0.01},
                ),
                "Distiller - top p": (
                    "FLOAT",
                    {"default": 0.90, "min": 0.0, "max": 1.0, "step": 0.01},
                ),
                "Distiller - top k": (
                    "INT",
                    {"default": 60, "min": 0, "max": 500, "step": 1},
                ),
                "Distiller - write prompt JSONL": (
                    "BOOLEAN",
                    {"default": False},
                ),
                "Distiller - preserve raw response": (
                    "BOOLEAN",
                    {"default": False},
                ),

                # Validator controls, same order pattern.
                "Validator - enabled": (
                    "BOOLEAN",
                    {
                        "default": True,
                        "tooltip": (
                            "Run the image-aware Validator stage. Disable this to stop after the "
                            "Distiller stage without producing final validated TXT/JSONL output."
                        ),
                    },
                ),
                "Validator - model": (
                    VALIDATOR_MODEL_CHOICES,
                    {"default": DEFAULT_VALIDATOR_MODEL},
                ),
                "Validator - custom Ollama model": (
                    "STRING",
                    {
                        "default": "",
                        "multiline": False,
                        "tooltip": "Used only when Validator - model is custom. Enter an installed Ollama vision model tag, for example gemma4:e4b.",
                    },
                ),

                "Validator - base seed": (
                    "INT",
                    {"default": -1, "min": -1, "max": MAX_SEED_32, "step": 1},
                ),
                "Validator - seed mode": (
                    SEED_MODES,
                    {"default": "fixed"},
                ),
                "Validator - prompt": (
                    "STRING",
                    {
                        "default": DEFAULT_VALIDATOR_PROMPT,
                        "multiline": True,
                        "tooltip": "Image-aware validator prompt. Existing file path is also accepted as shorthand.",
                    },
                ),
                "Validator - num predict": (
                    "INT",
                    {"default": 2200, "min": 64, "max": 12000, "step": 64},
                ),
                "Validator - temperature": (
                    "FLOAT",
                    {"default": 0.0, "min": 0.0, "max": 2.0, "step": 0.01},
                ),
                "Validator - top p": (
                    "FLOAT",
                    {"default": 0.92, "min": 0.0, "max": 1.0, "step": 0.01},
                ),
                "Validator - top k": (
                    "INT",
                    {"default": 80, "min": 0, "max": 500, "step": 1},
                ),
                "Validator - write prompt JSONL": (
                    "BOOLEAN",
                    {"default": False},
                ),
                "Validator - preserve raw VLM response": (
                    "BOOLEAN",
                    {"default": False},
                ),

                # Final export controls.
                "Final - caption style": (
                    FINAL_CAPTION_STYLES,
                    {"default": "narrative"},
                ),
                "Final - write TXT sidecars": (
                    "BOOLEAN",
                    {"default": True},
                ),
                "Final - write JSONL": (
                    "BOOLEAN",
                    {"default": True},
                ),
            },
            "optional": {
                "Input - single image": (
                    "IMAGE",
                    {
                        "tooltip": (
                            "Optional quick-workflow IMAGE passthrough/reference. In planned single-image "
                            "runs, wire the Planner single_image output here so the file-based VLM validator "
                            "can resolve direct IMAGE caption records."
                        ),
                    },
                ),
                "pipeline_plan": (
                    "CAPTIONFORGE_PIPELINE_PLAN",
                    {
                        "tooltip": (
                            "Connect the CaptionForge Pipeline Planner pipeline_plan output here. "
                            "When connected, planner-owned settings override matching widgets."
                        ),
                    },
                ),
            },
        }

    RETURN_TYPES = ("STRING", "STRING", "STRING", "STRING")
    RETURN_NAMES = ("final_captions", "final_jsonl_records", "output_paths_json", "status")
    FUNCTION = "forge"
    CATEGORY = "JLC/Captioning"

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        return float("NaN")

    def forge(self, **kwargs):
        plan = normalize_captionforge_pipeline_plan(kwargs.get("pipeline_plan") or kwargs.get("captionforge_run_config"))

        output_dir = _resolve_output_dir(str(kwargs.get("Output - folder", "") or ""), plan)
        output_dir.mkdir(parents=True, exist_ok=True)
        run_name = _resolve_run_name(str(kwargs.get("Output - run name", "captionforge_run") or "captionforge_run"), plan)
        paths = _derive_paths(output_dir, run_name, plan)

        distiller_enabled = _safe_bool(
            _resolve_setting(
                plan,
                kwargs.get("Distiller - enabled"),
                "distiller.enabled",
                "pass_b.enabled",
                "pass_b_distiller.enabled",
                default=True,
            ),
            True,
        )
        validator_enabled = _safe_bool(
            _resolve_setting(
                plan,
                kwargs.get("Validator - enabled"),
                "validator.enabled",
                "pass_c.enabled",
                "pass_c_vlm_validator.enabled",
                default=True,
            ),
            True,
        )
        if not distiller_enabled and not validator_enabled:
            raise RuntimeError(
                "JLC CaptionForge Node has nothing to do: both Distiller and Validator are disabled."
            )

        caption_jsonl = _resolve_caption_jsonl(str(kwargs.get("Input - captions JSONL", "") or ""), plan, paths)
        if distiller_enabled:
            if not caption_jsonl:
                raise RuntimeError("JLC CaptionForge Node requires Input - captions JSONL or a planner path when Distiller is enabled.")

            caption_path = Path(caption_jsonl)
            if not caption_path.exists() or caption_path.is_dir():
                raise FileNotFoundError(f"Caption JSONL not found: {caption_path}")
            if not caption_path.read_text(encoding="utf-8").strip():
                raise FileNotFoundError(f"Caption JSONL is empty: {caption_path}")
        else:
            # In validator-only reuse mode, Pass A input is not required. The
            # existing B_DISTILL file is keyed entirely by Output folder + run name.
            caption_jsonl = caption_jsonl or paths["caption_jsonl"] or paths["pass_a_jsonl"]
            distiller_path = Path(paths["distiller_jsonl"])
            if not distiller_path.exists() or distiller_path.is_dir():
                raise FileNotFoundError(
                    "Distiller is disabled, but the expected B_DISTILL JSONL was not found: "
                    f"{distiller_path}"
                )
            if not distiller_path.read_text(encoding="utf-8").strip():
                raise FileNotFoundError(
                    "Distiller is disabled, but the expected B_DISTILL JSONL is empty: "
                    f"{distiller_path}"
                )

        image_root = _resolve_image_root(str(kwargs.get("Input - image path", "") or ""), plan, caption_jsonl)
        single_image_root = _save_single_image_inputs_for_validator(
            kwargs.get("Input - single image"),
            output_dir,
            run_name,
        )
        if single_image_root:
            image_root = single_image_root

        overwrite = _safe_bool(_resolve_setting(plan, kwargs.get("Output - overwrite outputs"), "final.overwrite_outputs", "output.overwrite_outputs", "shared.overwrite_outputs", default=True), True)
        trigger_word = _normalize_text(_resolve_setting(plan, kwargs.get("LoRA - trigger word"), "shared.trigger_word", "lora.trigger_word", "trigger_word", default=""))
        user_caption_anchor = _normalize_text(_resolve_setting(plan, kwargs.get("LoRA - user caption anchor"), "shared.user_caption_anchor", "lora.user_caption_anchor", "user_caption_anchor", default=""))

        # Distiller settings. Planner-owned values override widgets.
        distiller_choice = _resolve_setting(
            plan,
            kwargs.get("Distiller - model"),
            "distiller.model",
            "pass_b.model",
            "distiller.ollama_model",
            "pass_b.ollama_model",
            "distiller.model_family",
            "pass_b.model_family",
            default=DEFAULT_DISTILLER_MODEL,
        )
        distiller_model = _resolve_ollama_model_name(
            distiller_choice,
            kwargs.get("Distiller - custom Ollama model"),
            DEFAULT_DISTILLER_MODEL,
        )
        distiller_seed = _seed_for_stage(
            _resolve_setting(plan, kwargs.get("Distiller - base seed"), "distiller.base_seed", "pass_b.base_seed", default=-1),
            _resolve_setting(plan, kwargs.get("Distiller - seed mode"), "distiller.seed_mode", "pass_b.seed_mode", default="fixed"),
            0,
        )
        distiller_prompt = _resolve_prompt(
            str(kwargs.get("Distiller - prompt", "") or ""),
            plan,
            ("distiller.prompt", "distiller.instructions", "pass_b.prompt", "pass_b.instructions"),
            DEFAULT_DISTILLER_INSTRUCTIONS,
        )
        distiller_prompt = _upgrade_legacy_distiller_prompt(distiller_prompt)
        distiller_strategy = str(_resolve_setting(plan, kwargs.get("Distiller - strategy"), "distiller.strategy", "pass_b.strategy", default="pollster_then_copywriter") or "pollster_then_copywriter")
        distiller_max_chars = _coerce_int(_resolve_setting(plan, kwargs.get("Distiller - max caption chars for LLM"), "distiller.max_caption_chars_for_llm", "pass_b.max_caption_chars_for_llm", default=1536), 1536, 0, 12000)
        distiller_num_predict = _coerce_int(_resolve_setting(plan, kwargs.get("Distiller - num predict"), "distiller.num_predict", "distiller.ollama_num_predict", "pass_b.num_predict", default=3096), 3096, 64, 12000)
        distiller_temperature = _coerce_float(_resolve_setting(plan, kwargs.get("Distiller - temperature"), "distiller.temperature", "distiller.ollama_temperature", "pass_b.temperature", default=0.24), 0.24, 0.0, 2.0)
        distiller_top_p = _coerce_float(_resolve_setting(plan, kwargs.get("Distiller - top p"), "distiller.top_p", "distiller.ollama_top_p", "pass_b.top_p", default=0.90), 0.90, 0.0, 1.0)
        distiller_top_k = _coerce_int(_resolve_setting(plan, kwargs.get("Distiller - top k"), "distiller.top_k", "distiller.ollama_top_k", "pass_b.top_k", default=60), 60, 0, 500)
        distiller_write_prompt = _safe_bool(_resolve_setting(plan, kwargs.get("Distiller - write prompt JSONL"), "distiller.write_prompt_jsonl", "pass_b.write_prompt_jsonl", default=False), False)
        distiller_preserve_raw = _safe_bool(_resolve_setting(plan, kwargs.get("Distiller - preserve raw response"), "distiller.preserve_raw_response", "pass_b.preserve_raw_response", default=False), False)

        if distiller_enabled:
            distiller_config = DistillerConfig(
                llm_backend="ollama",
                llm_model=distiller_model,
                instructions=distiller_prompt,
                strategy=distiller_strategy,
                max_caption_chars_for_llm=distiller_max_chars,
                ollama_num_predict=distiller_num_predict,
                ollama_temperature=distiller_temperature,
                ollama_top_p=distiller_top_p,
                ollama_top_k=distiller_top_k,
                preserve_raw_response=distiller_preserve_raw,
                trigger_word=trigger_word,
                user_caption_anchor=user_caption_anchor,
            )
            _set_if_present(distiller_config, ("ollama_seed", "seed"), distiller_seed)

            distiller_batch = DistillerBatchConfig(
                input_jsonl=caption_jsonl,
                output_jsonl=paths["distiller_jsonl"],
                readable_jsonl=paths["distiller_readable_jsonl"],
                readable_json=paths["distiller_readable_json"],
                prompt_jsonl=paths["distiller_prompt_jsonl"],
                dry_run=False,
                append_output=not overwrite,
                skip_existing=False,
                no_readable_sidecars=False,
                write_prompt_jsonl=distiller_write_prompt,
            )
            distiller_rc: int | str = run_distiller_batch(distiller_batch, distiller_config)
        else:
            distiller_rc = "skipped_existing_b_distill"
            print(
                f"[JLC CaptionForge Node] Distiller disabled; using existing B_DISTILL JSONL: {paths['distiller_jsonl']}",
                flush=True,
            )

        # Validator settings. Planner-owned values override widgets.
        validator_choice = _resolve_setting(
            plan,
            kwargs.get("Validator - model"),
            "validator.model",
            "pass_c.model",
            "validator.ollama_model",
            "pass_c.ollama_model",
            "validator.model_family",
            "pass_c.model_family",
            default=DEFAULT_VALIDATOR_MODEL,
        )
        validator_model = _resolve_ollama_model_name(
            validator_choice,
            kwargs.get("Validator - custom Ollama model"),
            DEFAULT_VALIDATOR_MODEL,
        )
        validator_seed = _seed_for_stage(
            _resolve_setting(plan, kwargs.get("Validator - base seed"), "validator.base_seed", "pass_c.base_seed", default=-1),
            _resolve_setting(plan, kwargs.get("Validator - seed mode"), "validator.seed_mode", "pass_c.seed_mode", default="fixed"),
            0,
        )
        validator_prompt = _resolve_prompt(
            str(kwargs.get("Validator - prompt", "") or ""),
            plan,
            ("validator.prompt", "validator.vlm_validator_prompt", "pass_c.prompt", "pass_c.vlm_validator_prompt"),
            DEFAULT_VALIDATOR_PROMPT,
        )
        validator_prompt = _upgrade_legacy_validator_prompt(validator_prompt)
        validator_num_predict = _coerce_int(_resolve_setting(plan, kwargs.get("Validator - num predict"), "validator.num_predict", "validator.ollama_num_predict", "pass_c.num_predict", default=2200), 2200, 64, 12000)
        validator_temperature = _coerce_float(_resolve_setting(plan, kwargs.get("Validator - temperature"), "validator.temperature", "validator.ollama_temperature", "pass_c.temperature", default=0.0), 0.0, 0.0, 2.0)
        validator_top_p = _coerce_float(_resolve_setting(plan, kwargs.get("Validator - top p"), "validator.top_p", "validator.ollama_top_p", "pass_c.top_p", default=0.92), 0.92, 0.0, 1.0)
        validator_top_k = _coerce_int(_resolve_setting(plan, kwargs.get("Validator - top k"), "validator.top_k", "validator.ollama_top_k", "pass_c.top_k", default=80), 80, 0, 500)
        validator_write_prompt = _safe_bool(_resolve_setting(plan, kwargs.get("Validator - write prompt JSONL"), "validator.write_prompt_jsonl", "pass_c.write_prompt_jsonl", default=False), False)
        validator_preserve_raw = _safe_bool(_resolve_setting(plan, kwargs.get("Validator - preserve raw VLM response"), "validator.preserve_raw_vlm_response", "pass_c.preserve_raw_vlm_response", default=False), False)

        if validator_enabled:
            validator_config = VLMValidatorConfig(
                vlm_backend="ollama",
                vlm_model=validator_model,
                vlm_validator_prompt=validator_prompt,
                trigger_word=trigger_word,
                user_caption_anchor=user_caption_anchor,
                preserve_raw_vlm_response=validator_preserve_raw,
                ollama_num_predict=validator_num_predict,
                ollama_temperature=validator_temperature,
                ollama_top_p=validator_top_p,
                ollama_top_k=validator_top_k,
                ollama_format_mode="schema",
            )
            _set_if_present(validator_config, ("ollama_seed", "seed"), validator_seed)

            validator_batch = BatchVLMValidatorConfig(
                input_jsonl=paths["distiller_jsonl"],
                image_root=image_root,
                output_jsonl=paths["validator_jsonl"],
                readable_sidecar_dir=paths["validator_readable_dir"],
                write_jsonl=True,
                write_readable_sidecars=True,
                dry_run=False,
                overwrite=overwrite,
                resume=False,
                write_prompt_jsonl=validator_write_prompt,
                prompt_jsonl=paths["validator_prompt_jsonl"],
            )
            validator_result = extract_validate_batch(validator_batch, validator_config)

            caption_style = str(_resolve_setting(plan, kwargs.get("Final - caption style"), "final.caption_style", "final_caption_style", default="narrative") or "narrative")
            write_txt = _safe_bool(_resolve_setting(plan, kwargs.get("Final - write TXT sidecars"), "final.write_txt_sidecars", "final.final_write_txt_sidecars", default=True), True)
            write_jsonl = _safe_bool(_resolve_setting(plan, kwargs.get("Final - write JSONL"), "final.write_jsonl", "final.final_write_jsonl", default=True), True)

            final_records, final_captions, final_ok, final_failed = _write_final_outputs(
                validator_jsonl=Path(paths["validator_jsonl"]),
                final_jsonl=Path(paths["final_jsonl"]),
                final_txt_dir=Path(paths["final_txt_dir"]),
                caption_style=caption_style,
                write_txt=write_txt,
                write_final_jsonl=write_jsonl,
            )
            validator_processed = validator_result.processed
            validator_failed = validator_result.failed
            validator_skipped = validator_result.skipped
        else:
            print(
                "[JLC CaptionForge Node] Validator disabled; stopping after Distiller stage. "
                "No final validated TXT/JSONL export was produced.",
                flush=True,
            )
            final_records = []
            final_captions = ""
            final_ok = 0
            final_failed = 0
            validator_processed = 0
            validator_failed = 0
            validator_skipped = "skipped_by_user"

        output_paths = dict(paths)
        output_paths.update(
            {
                "caption_jsonl": caption_jsonl,
                "pass_a_jsonl": caption_jsonl,
                "image_root": image_root,
                "planner_connected": _planner_overrides(plan),
                "distiller_enabled": distiller_enabled,
                "validator_enabled": validator_enabled,
                "distiller_model_resolved": distiller_model,
                "validator_model_resolved": validator_model,
                "distiller_seed_resolved": distiller_seed,
                "validator_seed_resolved": validator_seed,
            }
        )
        output_paths_json = json.dumps(output_paths, ensure_ascii=False, indent=2)
        try:
            Path(paths["output_paths_json"]).parent.mkdir(parents=True, exist_ok=True)
            Path(paths["output_paths_json"]).write_text(output_paths_json + "\n", encoding="utf-8")
        except Exception:
            pass

        final_jsonl_records = "\n".join(json.dumps(_json_safe(r), ensure_ascii=False) for r in final_records)
        status = (
            f"[JLC CaptionForge Node v{CAPTIONFORGE_NODE_VERSION}] complete | "
            f"planner_connected={_planner_overrides(plan)} | "
            f"distiller_enabled={distiller_enabled} "
            f"validator_enabled={validator_enabled} | "
            f"distiller_rc={distiller_rc} | "
            f"validator_processed={validator_processed} "
            f"validator_failed={validator_failed} "
            f"validator_skipped={validator_skipped} | "
            f"final_ok={final_ok} final_failed={final_failed} | "
            f"run={run_name} output={output_dir}"
        )
        print(status, flush=True)

        return (final_captions, final_jsonl_records, output_paths_json, status)



NODE_CLASS_MAPPINGS = {
    "JLC_CaptionForge": JLC_CaptionForge,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "JLC_CaptionForge": "\u2003JLC CaptionForge Node",
}
