"""
JLC CaptionForge Orchestrator

- CaptionForge
  - This node is part of **CaptionForge**, a model-agnostic captioning
    framework for ComfyUI developed by **J. L. Córdova**.

  - Repository
    https://github.com/Damkohler/CaptionForge

- CaptionForge 1.0 uses independent Pass-A witnesses, text-LLM synthesis,
  image-aware validation, SHORT/TAGGY formatting, and JSONL audit trails to
  produce grounded LoRA dataset captions.

- Node Purpose
    - The **JLC CaptionForge Orchestrator** is the production capstone for the current
      CaptionForge mainline.

    - This file is the **ComfyUI-facing wrapper**, not a caption model. It is
      responsible for:
            • ComfyUI INPUT_TYPES / widget definitions
            • standalone captions JSONL + image-root execution
            • CAPTIONFORGE_PIPELINE_PLAN consumption through `pipeline_plan`
            • planner-owned override resolution
            • optional direct IMAGE tensor handoff for validator resolution
            • Pass A raw caption selection and grouping by image
            • Pass B fat-draft construction with a text-only Ollama LLM
            • Pass C natural-caption validation with an image-aware Ollama VLM
            • Pass D short/taggy derivative construction with a text-only Ollama LLM
            • final long/short/taggy TXT and JSONL export
            • output path derivation and run audit status strings

- CaptionForge Pipeline Role
    - In planned mode, this node consumes the Pipeline Planner object and lets
      planner-owned values override matching visible widgets.

    - In standalone mode, it can run directly from a Pass A captions JSONL and
      an image file/folder root.

    - The mainline flow is:

            A_RAW_CAPTIONS
              -> B_FAT_DRAFT                  text-only LLM
              -> C_VLM_VALIDATED_FINAL        image-aware VLM natural caption
              -> D_FORMAT_TAGGY               text-only derivative formatter
              -> final TXT/JSONL export

    - The VLM-validated natural paragraph is the natural final caption. The
      formatter pass must not rewrite that natural paragraph; it only derives
      shorter natural-language and comma-separated variants from it.

- Ollama Model Dropdowns
    - Fat Draft, Validator, and Formatter dropdown values are explicit Ollama
      model tags.

    - Family aliases and shorthand substitutions are intentionally not used.

    - Dropdown choices are loaded at node-import time from:
            config/captionforge_ollama_models.json

    - Supported config keys:
            distiller_models / defaults.distiller_model
            validator_models / defaults.validator_model
            format_models    / defaults.format_model

    - The optional Custom choice lets users enter any installed Ollama model tag
      without editing Python.

- Prompting Model
    - The Fat Draft LLM does not see the image. It merges multiple raw captions
      into one deliberately over-complete draft.

    - The Validator VLM sees the actual image and the fat draft. It returns one
      corrected natural paragraph.

    - The Formatter LLM sees only the validated paragraph. In one call it
      returns a shorter natural caption and a taggy comma-separated caption.

- Model and Dependency Notes
    - This node talks to a local Ollama server over HTTP.

    - Large Ollama models may take significant time to load or respond. The
      `Ollama - request timeout seconds` widget controls network patience only;
      it does not change caption quality.

    - The node sends top-level `think: false` to Ollama requests so thinking
      models do not spend the entire token budget in hidden reasoning before
      producing visible caption text.

- Design Philosophy
    - CaptionForge is an original concept and implementation, not derived from
      or based on another ComfyUI workflow.

    - The Orchestrator keeps coordination and ComfyUI UI concerns separate from
      model-specific caption generation.

    - The node prioritizes auditable local caption refinement, explicit model
      selection, deterministic output paths, and high-value LoRA captions.

- Production Status
    - This is the active CaptionForge 1.0 B/C/D implementation. It remains fully
      usable standalone; when a Planner is connected, Planner-owned values take
      precedence over matching local controls.

- Attribution & License
  - Concept and implementation by **J. L. Córdova**
    with development assistance from **ChatGPT (OpenAI)**.

  - Designed for use with:
    https://github.com/comfyanonymous/ComfyUI

  - Copyright (c) 2026 J. L. Córdova

  - Released under the **MIT License**.
"""

from __future__ import annotations
from ..captionforge_version import CAPTIONFORGE_VERSION

MANIFEST = {
    "name": "JLC CaptionForge Orchestrator",
    "version": CAPTIONFORGE_VERSION,
    "author": "J. L. Córdova",
    "description": (
        "CaptionForge Orchestrator 1.0 node. Consumes Pass A raw caption "
        "JSONL directly or through a CAPTIONFORGE_PIPELINE_PLAN, builds a text-only "
        "fat draft with an Ollama LLM, validates it against the image with an Ollama "
        "VLM to produce the natural final caption, derives short and taggy variants "
        "with one format-model call, and exports TXT/JSONL artifacts. The natural "
        "caption is the VLM-validated output directly; the formatter pass does not "
        "rewrite the natural paragraph."
    ),
}

import base64
import hashlib
import io
import json
import re
import urllib.error
import urllib.request
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import Any

import numpy as np
import torch
from PIL import Image

from ..engines.captionforge_cleanup import (
    apply_cleanup_contract,
    normalize_forbidden_phrases,
    normalize_replace_pairs,
)

from ..engines.captionforge_prompt_defaults import (
    DEFAULT_FAT_DRAFT_INSTRUCTIONS,
    DEFAULT_TAGGY_FORMATTER_INSTRUCTIONS,
    DEFAULT_VALIDATOR_INSTRUCTIONS,
    DEFAULT_VALIDATOR_SYSTEM_PROMPT,
)
from ..engines.captionforge_source_identity import optional_image_filename
from ..engines.captionforge_dataset_export import (
    dataset_export_inputs,
    dataset_root,
    export_dimensions,
    export_pair,
    export_settings_from_widgets,
    is_dataset_export,
    normalize_export_settings,
    prepare_dataset_root,
)

try:
    import folder_paths
except Exception:  # pragma: no cover - useful outside ComfyUI smoke tests
    folder_paths = None

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

try:
    from ..engines.captionforge_model_cache import (
        cache_size as _captionforge_cache_size,
        unload_all as _captionforge_unload_all,
    )
except Exception:  # pragma: no cover - keeps direct/local smoke tests importable
    _captionforge_cache_size = None
    _captionforge_unload_all = None


CAPTIONFORGE_NODE_VERSION = CAPTIONFORGE_VERSION
TXT_EXPORT_FORMATS = ["natural", "taggy", "both_separate"]

DEFAULT_OLLAMA_URL = "http://127.0.0.1:11434"
CONFIG_RELATIVE_PATH = Path("config") / "captionforge_ollama_models.json"

DEFAULT_DISTILLER_MODELS = [
    "mistral-small:24b",
    "VladimirGav/gemma4-26b-16GB-VRAM-Uncensored",
    "deepseek-r1:32b",
    "tarruda/neuraldaredevil-8b-abliterated:fp16",
    "gpt-oss:20b",
]
DEFAULT_VALIDATOR_MODELS = ["gemma4:26b", "qwen3.6:35B-A3B"]
DEFAULT_FORMAT_MODELS = ["mistral-small:24b", "VladimirGav/gemma4-26b-16GB-VRAM-Uncensored", "gpt-oss:20b", "deepseek-r1:32b"]
DEFAULT_DISTILLER_MODEL = "mistral-small:24b"
DEFAULT_VALIDATOR_MODEL = "gemma4:26b"
DEFAULT_FORMAT_MODEL = "mistral-small:24b"

_SUPPORTED_IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff"}


def _evict_python_models_before_ollama_if_needed(caller: str) -> None:
    """Unload resident Python/HF CaptionForge models before Orchestrator Ollama calls.

    The Orchestrator's B/C/D stages are served by the local Ollama daemon. Joy/Qwen
    and other Python-side caption models may still be resident after Pass A, so
    clear the CaptionForge process-local cache once before the first Ollama
    request. Because Ollama models are not registered in this cache, this does
    not unload or churn between Ollama text/VLM/formatter models.
    """
    if _captionforge_cache_size is None or _captionforge_unload_all is None:
        return

    try:
        resident = int(_captionforge_cache_size(include_keep=True))
    except Exception as exc:
        print(f"[{caller}] WARNING: Could not inspect CaptionForge Python model cache before Ollama handoff: {exc}", flush=True)
        return

    if resident <= 0:
        return

    try:
        evicted = int(
            _captionforge_unload_all(
                include_keep=True,
                reason="handoff_to_ollama",
                safe=True,
            )
        )
        if evicted:
            print(f"[{caller}] Evicted {evicted} CaptionForge Python model(s) before Ollama handoff.", flush=True)
    except Exception as exc:
        print(f"[{caller}] WARNING: Python model cache eviction before Ollama handoff failed: {exc}", flush=True)


@dataclass
class StageRecord:
    captionforge_pass: str
    engine: str
    engine_version: str
    image_key: str
    image: str
    status: str
    text: str
    model: str
    prompt: str
    params: dict[str, Any]
    source: dict[str, Any]
    timestamp: str


# -----------------------------------------------------------------------------
# Path, JSON, and config helpers
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


def _find_repo_root() -> Path:
    here = Path(__file__).resolve()
    for candidate in [here.parent, *here.parents]:
        if (candidate / CONFIG_RELATIVE_PATH).exists():
            return candidate
    # Expected when this file lives in CaptionForge/nodes/.
    try:
        return here.parents[1]
    except Exception:
        return Path.cwd()


def _config_path() -> Path:
    return _find_repo_root() / CONFIG_RELATIVE_PATH


def _load_ollama_model_dropdowns() -> dict[str, Any]:
    data: dict[str, Any] = {}
    path = _config_path()
    try:
        if path.exists():
            loaded = json.loads(path.read_text(encoding="utf-8"))
            if isinstance(loaded, dict):
                data = loaded
    except Exception as exc:
        print(f"[JLC CaptionForge Orchestrator] Could not read Ollama model config {path}: {exc}", flush=True)

    defaults = data.get("defaults") if isinstance(data.get("defaults"), dict) else {}
    include_custom = bool(data.get("include_custom", True))

    def choices(key: str, fallback: list[str], default_key: str, default_value: str) -> tuple[list[str], str]:
        raw = data.get(key)
        models = [str(x).strip() for x in raw if str(x).strip()] if isinstance(raw, list) else list(fallback)
        default = str(defaults.get(default_key) or default_value).strip() or default_value
        if default not in models:
            models.insert(0, default)
        out = list(dict.fromkeys(models))
        if include_custom and "Custom" not in out:
            out.append("Custom")
        return out, default

    distiller_models, distiller_default = choices(
        "distiller_models", DEFAULT_DISTILLER_MODELS, "distiller_model", DEFAULT_DISTILLER_MODEL
    )
    validator_models, validator_default = choices(
        "validator_models", DEFAULT_VALIDATOR_MODELS, "validator_model", DEFAULT_VALIDATOR_MODEL
    )
    format_models, format_default = choices(
        "format_models", DEFAULT_FORMAT_MODELS, "format_model", DEFAULT_FORMAT_MODEL
    )
    return {
        "distiller_models": distiller_models,
        "validator_models": validator_models,
        "format_models": format_models,
        "distiller_default": distiller_default,
        "validator_default": validator_default,
        "format_default": format_default,
    }


_MODEL_DROPDOWNS = _load_ollama_model_dropdowns()
DISTILLER_MODEL_CHOICES = _MODEL_DROPDOWNS["distiller_models"]
VALIDATOR_MODEL_CHOICES = _MODEL_DROPDOWNS["validator_models"]
FORMAT_MODEL_CHOICES = _MODEL_DROPDOWNS["format_models"]
DEFAULT_DISTILLER_MODEL = _MODEL_DROPDOWNS["distiller_default"]
DEFAULT_VALIDATOR_MODEL = _MODEL_DROPDOWNS["validator_default"]
DEFAULT_FORMAT_MODEL = _MODEL_DROPDOWNS["format_default"]


def _clean_run_name(value: Any) -> str:
    text = str(value or "captionforge_run").strip()
    text = re.sub(r"[^A-Za-z0-9_.-]+", "_", text).strip("._")
    return text or "captionforge_run"


def _normalize_text(value: Any) -> str:
    return re.sub(r"\s+", " ", str(value or "").replace("\n", " ")).strip()


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


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            text = line.strip()
            if not text:
                continue
            try:
                obj = json.loads(text)
            except Exception as exc:
                raise RuntimeError(f"Invalid JSONL at {path}:{line_no}: {exc}") from exc
            if isinstance(obj, dict):
                records.append(obj)
    return records


def _write_jsonl(path: Path, records: list[dict[str, Any]] | list[StageRecord], *, append: bool = True) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    mode = "a" if append else "w"
    with path.open(mode, encoding="utf-8", newline="\n") as f:
        for record in records:
            obj = asdict(record) if hasattr(record, "__dataclass_fields__") else record
            f.write(json.dumps(_json_safe(obj), ensure_ascii=False) + "\n")


def _write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(str(text or "").rstrip() + "\n", encoding="utf-8")


def _write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_json_safe(data), indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def _truncate_for_prompt(text: str, max_chars: int) -> str:
    text = str(text or "").strip()
    if max_chars <= 0 or len(text) <= max_chars:
        return text
    return text[: max(0, max_chars)].rstrip() + " …"


# -----------------------------------------------------------------------------
# Planner/setting resolution
# -----------------------------------------------------------------------------


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
        value = _deep_get(plan, key, None) if "." in key else plan.get(key)
        if value not in (None, ""):
            return value
    return default


def _planner_overrides(plan: dict[str, Any]) -> bool:
    return bool(plan and str(plan.get("captionforge_config_type", "")).lower() in {"captionforge_pipeline_plan", "captionforge_run_config"})


def _resolve_setting(plan: dict[str, Any], widget_value: Any, *plan_keys: str, default: Any = None) -> Any:
    if _planner_overrides(plan):
        value = _plan_get(plan, *plan_keys, default=None)
        if value not in (None, ""):
            return value
    return widget_value if widget_value not in (None, "") else default


def _resolve_fat_draft_max_caption_chars(plan: dict[str, Any], widget_value: Any) -> int:
    """Resolve the per-source Pass A caption cap without changing prompt semantics."""
    return _coerce_int(
        _resolve_setting(
            plan,
            widget_value,
            "distiller.max_caption_chars_for_llm",
            "pass_b.max_caption_chars_for_llm",
            "pass_b_distiller.max_caption_chars_for_llm",
            default=1536,
        ),
        1536,
        0,
        12000,
    )


def _resolve_stage_audit_settings(
    plan: dict[str, Any],
    widget_write_prompts: Any,
    widget_preserve_raw: Any,
) -> dict[str, bool]:
    """Resolve independent Planner-owned B/C/D audit controls when planned."""
    standalone_write = _safe_bool(widget_write_prompts, False)
    standalone_raw = _safe_bool(widget_preserve_raw, False)
    return {
        "fat_write_prompts": _safe_bool(
            _resolve_setting(
                plan,
                standalone_write,
                "distiller.write_prompt_jsonl",
                "pass_b.write_prompt_jsonl",
                "pass_b_distiller.write_prompt_jsonl",
                default=False,
            ),
            False,
        ),
        "fat_preserve_raw": _safe_bool(
            _resolve_setting(
                plan,
                standalone_raw,
                "distiller.preserve_raw_response",
                "pass_b.preserve_raw_response",
                "pass_b_distiller.preserve_raw_response",
                default=False,
            ),
            False,
        ),
        "val_write_prompts": _safe_bool(
            _resolve_setting(
                plan,
                standalone_write,
                "validator.write_prompt_jsonl",
                "pass_c.write_prompt_jsonl",
                "pass_c_vlm_validator.write_prompt_jsonl",
                default=False,
            ),
            False,
        ),
        "val_preserve_raw": _safe_bool(
            _resolve_setting(
                plan,
                standalone_raw,
                "validator.preserve_raw_vlm_response",
                "pass_c.preserve_raw_vlm_response",
                "pass_c_vlm_validator.preserve_raw_vlm_response",
                default=False,
            ),
            False,
        ),
        "fmt_write_prompts": _safe_bool(
            _resolve_setting(
                plan,
                standalone_write,
                "formatter.write_prompt_jsonl",
                "format.write_prompt_jsonl",
                "pass_d.write_prompt_jsonl",
                "pass_d_formatter.write_prompt_jsonl",
                default=False,
            ),
            False,
        ),
        "fmt_preserve_raw": _safe_bool(
            _resolve_setting(
                plan,
                standalone_raw,
                "formatter.preserve_raw_response",
                "format.preserve_raw_response",
                "pass_d.preserve_raw_response",
                "pass_d_formatter.preserve_raw_response",
                default=False,
            ),
            False,
        ),
    }


def _resolve_ollama_model_name(value: Any, custom_value: Any, fallback: str) -> str:
    text = str(value or "").strip()
    custom = str(custom_value or "").strip()
    if text.lower() == "custom":
        return custom or fallback
    return text or custom or fallback


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


def _normalize_optional_seed(value: Any) -> int | None:
    if value in (None, ""):
        return None
    seed = _coerce_int(value, -1, -1, MAX_SEED_32)
    return seed if seed >= 0 else None


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
    """Resolve Orchestrator artifact paths.

    Planned runs receive paths from the Pipeline Planner. Standalone runs treat
    the visible Output folder as an output root and create a run-specific working
    directory inside it. Final LoRA TXT sidecars are written beside each resolved
    source image by _write_final_txt_sidecars(), not into final_txt_dir.
    """
    planned_paths = _plan_get(plan, "paths", default={}) if _planner_overrides(plan) else {}
    planned_paths = planned_paths if isinstance(planned_paths, dict) else {}

    output_root = Path(str(planned_paths.get("output_root") or "").strip() or output_dir)
    working_dir = Path(str(planned_paths.get("working_dir") or "").strip() or (output_root / f"{run_name}__working"))

    def pick(names: tuple[str, ...], fallback: Path) -> str:
        for name in names:
            value = planned_paths.get(name)
            if str(value or "").strip():
                return str(value).strip()
        return str(fallback)

    return {
        "output_root": str(output_root),
        "working_dir": str(working_dir),
        # Compatibility: internal audit/artifact writers use output_dir.
        "output_dir": pick(("output_dir",), working_dir),
        "run_name": run_name,
        "caption_jsonl": pick(("caption_jsonl", "pass_a_jsonl"), working_dir / f"{run_name}__A_RAW_CAPTIONS.jsonl"),
        "pass_a_jsonl": pick(("pass_a_jsonl", "caption_jsonl"), working_dir / f"{run_name}__A_RAW_CAPTIONS.jsonl"),
        "fat_draft_jsonl": pick(("fat_draft_jsonl", "distiller_jsonl"), working_dir / f"{run_name}__B_FAT_DRAFT.jsonl"),
        "fat_draft_prompt_jsonl": pick(("fat_draft_prompt_jsonl", "distiller_prompt_jsonl"), working_dir / f"{run_name}__B_FAT_DRAFT_prompts.jsonl"),
        "validator_jsonl": pick(("validator_jsonl",), working_dir / f"{run_name}__C_VLM_VALIDATED_FINAL.jsonl"),
        "validator_prompt_jsonl": pick(("validator_prompt_jsonl",), working_dir / f"{run_name}__C_VLM_VALIDATOR_prompts.jsonl"),
        "taggy_jsonl": pick(("taggy_jsonl", "formatter_jsonl"), working_dir / f"{run_name}__D_FORMAT_TAGGY.jsonl"),
        "taggy_prompt_jsonl": pick(("taggy_prompt_jsonl", "formatter_prompt_jsonl"), working_dir / f"{run_name}__D_FORMAT_TAGGY_prompts.jsonl"),
        "final_jsonl": pick(("final_jsonl",), working_dir / f"{run_name}__D_FINAL_EXPORT.jsonl"),
        # Legacy/debug-only. Primary final TXT sidecars are written beside images.
        "final_txt_dir": pick(("final_txt_dir",), working_dir / f"{run_name}__TXT"),
        "final_sidecar_policy": str(planned_paths.get("final_sidecar_policy") or "beside_resolved_source_image"),
        "working_images_dir": pick(("working_images_dir",), working_dir / "images"),
        "opt_images_dir": pick(("opt_images_dir",), working_dir / "images" / "opt_images"),
        "output_paths_json": pick(("output_paths_json",), working_dir / f"{run_name}__output_paths.json"),
        "run_config_json": pick(("run_config_json",), working_dir / f"{run_name}__run_config.json"),
        "raw_response_dir": pick(("raw_response_dir",), working_dir / f"{run_name}__raw_responses"),
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
        p = Path(widget_value)
        if p.is_absolute():
            return str(p)
        return str(Path(str(paths.get("output_dir") or ".")) / p)
    return paths["caption_jsonl"] or paths["pass_a_jsonl"]


def _resolve_image_root(widget_value: str, plan: dict[str, Any], caption_jsonl: str) -> str:
    if _planner_overrides(plan):
        planned = _plan_get(plan, "paths.image_root", "shared.image_root", "shared.input_path", "input_path", default="")
        if str(planned or "").strip():
            return str(planned).strip()
    widget_value = str(widget_value or "").strip()
    if widget_value:
        return widget_value
    try:
        return str(Path(caption_jsonl).resolve().parent)
    except Exception:
        return ""


# -----------------------------------------------------------------------------
# Image helpers
# -----------------------------------------------------------------------------


def _tensor_to_pil_images(image_tensor: Any) -> list[Image.Image]:
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


def _save_single_image_inputs_for_validator(image_tensor: Any, image_dir: Path) -> str:
    images = _tensor_to_pil_images(image_tensor)
    if not images:
        return ""
    image_dir.mkdir(parents=True, exist_ok=True)
    for index, pil in enumerate(images):
        stem = f"comfy_image_{index:04d}"
        audit_target = image_dir / f"{stem}.png"
        pil.save(audit_target, format="PNG")
    print(f"[JLC CaptionForge Orchestrator] Saved optional IMAGE input(s) for validator: {image_dir}", flush=True)
    return str(image_dir)

def _basename_cross_platform(value: Any) -> str:
    text = str(value or "").strip()
    if not text:
        return ""
    return PureWindowsPath(text).name if "\\" in text else Path(text).name


def _safe_txt_stem(value: Any) -> str:
    text = str(value or "").strip()
    if not text:
        return "image"
    base = _basename_cross_platform(text)
    stem = Path(base).stem or base
    stem = re.sub(r'[<>:"/\\|?*\x00-\x1f]+', "_", stem).rstrip(" .")
    stem = stem or "image"
    if "/" in text or "\\" in text:
        digest = hashlib.sha256(text.encode("utf-8")).hexdigest()[:10]
        return f"{stem}__{digest}"
    return stem


def _safe_source_name(value: str) -> str:
    cleaned = []
    for ch in str(value or "").replace("\\", "/"):
        if ch.isalnum() or ch in {"-", "_", "."}:
            cleaned.append(ch)
        elif ch == "/":
            cleaned.append("__")
        else:
            cleaned.append("_")
    out = "".join(cleaned).strip("._")
    return out or "image"


def _dedupe_paths(values: list[Any]) -> list[Path]:
    out: list[Path] = []
    seen: set[str] = set()
    for value in values:
        text = str(value or "").strip()
        if not text:
            continue
        p = Path(text)
        marker = str(p)
        if marker in seen:
            continue
        out.append(p)
        seen.add(marker)
    return out


def _record_image_key(record: dict[str, Any], fallback_index: int = 0) -> str:
    """Return the stable grouping key for a Pass A image record.

    ``image_key`` is the canonical relative-path or optional-image identity emitted
    by caption witnesses. It must remain opaque here so distinct directories and
    extensions cannot be merged into one downstream group.
    """
    explicit_key = record.get("image_key")
    if explicit_key is not None and str(explicit_key) != "":
        return str(explicit_key)

    for key in ("image", "source_image", "filename"):
        value = str(record.get(key) or "").strip()
        if value:
            base = _basename_cross_platform(value)
            if base:
                candidate = Path(base)
                if candidate.suffix.lower() in _SUPPORTED_IMAGE_SUFFIXES:
                    return candidate.stem or base
                return base
            return value
    return f"image_{fallback_index:04d}"


def _candidate_image_paths(image_root: str, value: Any) -> list[Path]:
    text = str(value or "").strip()
    if not text:
        return []
    root = Path(str(image_root or "").strip()) if str(image_root or "").strip() else None
    candidates: list[Path] = []

    raw_path = Path(text)
    if raw_path.is_absolute():
        candidates.append(raw_path)
    elif root is not None:
        candidates.append(root / text)

    base = _basename_cross_platform(text)
    stem = Path(base).stem if base else text
    names = [base, stem]
    for name in list(names):
        if name:
            for suffix in _SUPPORTED_IMAGE_SUFFIXES:
                names.append(str(Path(name).with_suffix(suffix)))
    seen: set[str] = set()
    for name in names:
        if not name:
            continue
        if root is not None:
            p = root / name
            marker = str(p)
            if marker not in seen:
                candidates.append(p)
                seen.add(marker)
    return candidates


def _candidate_keys_for_image_file(path: Path, root: Path | None = None) -> set[str]:
    keys = {
        path.name,
        path.stem,
        _safe_source_name(path.name),
        _safe_source_name(path.stem),
    }
    if root is not None:
        try:
            rel = path.relative_to(root).with_suffix("")
            keys.add(str(rel))
            keys.add(_safe_source_name(str(rel)))
        except Exception:
            pass
    return {k for k in keys if k}


def _values_for_group_resolution(records: list[dict[str, Any]]) -> set[str]:
    values: set[str] = set()
    for record in records:
        for key in ("image_resolved_path", "image_path", "source_path", "image", "image_key"):
            value = str(record.get(key) or "").strip()
            if not value:
                continue
            base = _basename_cross_platform(value)
            stem = Path(base).stem if base else value
            values.update({value, base, stem, _safe_source_name(value), _safe_source_name(base), _safe_source_name(stem)})
    return {v for v in values if v}


def _resolve_image_path_for_group(
    records: list[dict[str, Any]],
    image_roots: str | Path | list[str | Path],
    *,
    optional_image_root: str | Path | None = None,
) -> Path | None:
    if isinstance(image_roots, (str, Path)):
        roots = _dedupe_paths([image_roots])
    else:
        roots = _dedupe_paths(list(image_roots or []))

    optional_root = Path(optional_image_root) if optional_image_root else None
    explicit_keys = [str(record.get("image_key")) for record in records if record.get("image_key")]
    optional_filenames = {optional_image_filename(key) for key in explicit_keys}
    optional_filenames.discard("")

    # Optional IMAGE tensors live in a reserved key namespace. Resolve them only
    # against their saved opt_images directory so an ordinary dataset file with
    # the same synthetic basename cannot be selected accidentally.
    if optional_filenames:
        if optional_root is None or len(optional_filenames) != 1:
            return None
        candidate = optional_root / next(iter(optional_filenames))
        return candidate if candidate.exists() and candidate.is_file() else None

    # New Pass-A keys are relative POSIX-style paths with their extension intact.
    # Prefer an exact root/key match before any historical loose-name fallback.
    for key in explicit_keys:
        key_path = PurePosixPath(key)
        if key_path.is_absolute() or ".." in key_path.parts:
            continue
        relative_key = Path(*key_path.parts)
        for root in roots:
            if root.is_file():
                if len(key_path.parts) == 1 and root.name == key:
                    return root
                continue
            candidate = root / relative_key
            if candidate.exists() and candidate.is_file():
                return candidate

    values = _values_for_group_resolution(records)

    # First try explicit/direct candidate paths from the raw record fields.
    existing: list[Path] = []
    has_new_relative_key = any(
        PurePosixPath(key).suffix.lower() in _SUPPORTED_IMAGE_SUFFIXES
        for key in explicit_keys
    )
    compatibility_roots = [
        root
        for root in roots
        if not (has_new_relative_key and optional_root is not None and root == optional_root)
    ]
    for value in values:
        for root in compatibility_roots:
            for candidate in _candidate_image_paths(str(root), value):
                if candidate.exists() and candidate.is_file():
                    existing.append(candidate)

    if existing:
        existing = _dedupe_paths(existing)
        # Prefer real image extensions over the extensionless compatibility copy.
        existing.sort(key=lambda p: 0 if p.suffix.lower() in _SUPPORTED_IMAGE_SUFFIXES else 1)
        return existing[0]

    # Then index actual image files by exact names/stems and legacy sanitized keys.
    # The latter keeps older RAW JSONL artifacts resolvable after canonical witness
    # keys switched to verbatim source-filename stems.
    for root in compatibility_roots:
        try:
            if root.is_file() and root.suffix.lower() in _SUPPORTED_IMAGE_SUFFIXES:
                if values.intersection(_candidate_keys_for_image_file(root, None)):
                    return root
            if not root.exists() or not root.is_dir():
                continue
            for path in sorted(p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in _SUPPORTED_IMAGE_SUFFIXES and not is_dataset_export(p)):
                if values.intersection(_candidate_keys_for_image_file(path, root)):
                    return path
        except Exception:
            continue

    return None

def _pil_to_base64_png(path: Path, max_size: int = 0, *, prepared_images: list | None = None) -> str:
    with Image.open(path) as img:
        source_size = img.size
        rgb = img.convert("RGB")
        max_size = int(max_size or 0)
        if max_size > 0:
            width, height = rgb.size
            longest = max(width, height)
            if longest > max_size:
                scale = max_size / float(longest)
                new_size = (
                    max(1, int(round(width * scale))),
                    max(1, int(round(height * scale))),
                )
                rgb = rgb.resize(new_size, Image.Resampling.LANCZOS)
        validator_size = rgb.size
        if prepared_images is not None:
            prepared_images.append((rgb, source_size))
        buf = io.BytesIO()
        rgb.save(buf, format="PNG")
    encoded = base64.b64encode(buf.getvalue()).decode("ascii")
    print(
        "[JLC CaptionForge Orchestrator] Validator image preparation: "
        f"source={source_size[0]}x{source_size[1]} "
        f"validator={validator_size[0]}x{validator_size[1]} "
        f"encoded_payload={len(encoded) / (1024 * 1024):.2f} MiB",
        flush=True,
    )
    return encoded


# -----------------------------------------------------------------------------
# Prompt and caption helpers
# -----------------------------------------------------------------------------


def _field(record: dict[str, Any], names: tuple[str, ...]) -> str:
    for name in names:
        value = record.get(name)
        if value not in (None, ""):
            return str(value).strip()
    return ""


def _group_records_by_image(records: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for index, record in enumerate(records):
        key = _record_image_key(record, index)
        grouped.setdefault(key, []).append(record)
    return grouped


def _family_of(record: dict[str, Any]) -> str:
    return _field(record, ("model_family", "family", "backend", "model_name")).lower()


def _caption_of(record: dict[str, Any]) -> str:
    return _field(record, ("caption", "raw_caption", "text", "final_caption"))


def _select_caption_records(
    records: list[dict[str, Any]],
    *,
    include_families: str,
    max_per_family: int,
    max_total: int,
) -> list[dict[str, Any]]:
    include = [x.strip().lower() for x in str(include_families or "").split(",") if x.strip()]
    include_all = not include or "all" in include or "*" in include
    per_family_counts: dict[str, int] = {}
    selected: list[dict[str, Any]] = []

    for record in records:
        status = str(record.get("status") or "ok").strip().lower()
        if status not in {"", "ok", "prompt_only"}:
            continue
        caption = _caption_of(record)
        if not caption:
            continue
        family = _family_of(record) or "unknown"
        if not include_all and family not in include:
            continue
        if max_per_family > 0 and per_family_counts.get(family, 0) >= max_per_family:
            continue
        per_family_counts[family] = per_family_counts.get(family, 0) + 1
        selected.append(record)
        if max_total > 0 and len(selected) >= max_total:
            break
    return selected


def _build_caption_blocks(records: list[dict[str, Any]], max_caption_chars: int) -> str:
    blocks: list[str] = []
    for i, record in enumerate(records, start=1):
        family = _family_of(record) or "unknown"
        run = _field(record, ("ensemble_run_index", "run_index"))
        caption = _truncate_for_prompt(_caption_of(record), max_caption_chars)
        if not caption:
            continue
        blocks.append(f"[{i}] family={family} run={run}\n{caption}")
    return "\n\n".join(blocks).strip()


def _append_optional_lora_guidance(prompt: str, trigger_word: str, user_caption_anchor: str) -> str:
    additions: list[str] = []
    if str(trigger_word or "").strip():
        additions.append(f"Configured LoRA trigger word: {str(trigger_word).strip()}")
    if str(user_caption_anchor or "").strip():
        additions.append(f"Configured user caption anchor: {str(user_caption_anchor).strip()}")
    if not additions:
        return prompt
    return prompt.rstrip() + "\n\nPipeline metadata:\n" + "\n".join(f"- {x}" for x in additions)


def _build_fat_draft_prompt(instructions: str, caption_blocks: str, trigger_word: str, user_caption_anchor: str) -> str:
    instr = _append_optional_lora_guidance(str(instructions or DEFAULT_FAT_DRAFT_INSTRUCTIONS), trigger_word, user_caption_anchor)
    return f"{instr}\n\nCaptions:\n{caption_blocks}\n\nOver-complete merged draft:"


def _build_validator_prompt(instructions: str, draft_caption: str, trigger_word: str, user_caption_anchor: str) -> str:
    instr = _append_optional_lora_guidance(str(instructions or DEFAULT_VALIDATOR_INSTRUCTIONS), trigger_word, user_caption_anchor)
    return f"{instr}\n\nDraft caption to validate:\n{draft_caption}\n\nCorrected caption only:"


def _build_taggy_prompt(instructions: str, validated_caption: str, trigger_word: str, user_caption_anchor: str) -> str:
    """Build the single Pass-D prompt for short and taggy derivatives.

    The historical name is retained for internal compatibility with tests and
    custom integrations that import this helper.
    """
    instr = _append_optional_lora_guidance(str(instructions or DEFAULT_TAGGY_FORMATTER_INSTRUCTIONS), trigger_word, user_caption_anchor)
    return f"{instr}\n\nValidated paragraph:\n{validated_caption}\n\nCaption derivatives:"


def _cleanup_single_paragraph(text: str) -> str:
    text = str(text or "").strip().strip('"').strip()
    text = re.sub(r"(?is)^\s*(corrected caption only:|final caption:|caption:)\s*", "", text).strip()
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def _cleanup_taggy(text: str) -> str:
    text = str(text or "").strip().strip('"').strip()
    match = re.search(r"(?is)TAGGY:\s*(.*?)(?:\n\s*(?:SHORT:|NATURAL:|$)|\Z)", text)
    if match:
        text = match.group(1).strip()
    text = re.sub(r"(?is)^\s*(taggy comma-list only:|taggy:|caption:)\s*", "", text).strip()
    text = text.replace("\n", ", ")
    text = re.sub(r"\s*,\s*", ", ", text)
    text = re.sub(r",\s*,+", ", ", text)
    return text.strip(" ,")


def _parse_formatter_derivatives(text: str) -> tuple[str, str]:
    """Parse the dual Pass-D response, with legacy taggy-only compatibility."""
    raw = str(text or "").strip()
    short_match = re.search(
        r"(?is)(?:^|\n)\s*SHORT:\s*(.*?)(?=\s*(?:\n\s*)?TAGGY:|\Z)",
        raw,
    )
    taggy_match = re.search(r"(?is)(?:^|\n)\s*TAGGY:\s*(.*?)\s*\Z", raw)

    short = _cleanup_single_paragraph(short_match.group(1)) if short_match else ""
    if taggy_match:
        taggy = _cleanup_taggy(taggy_match.group(1))
    elif not short_match:
        # Custom and legacy formatter prompts may still return one unlabeled
        # comma list. Keep that behavior and let the deterministic short
        # generator provide the fallback.
        taggy = _cleanup_taggy(raw)
    else:
        taggy = ""
    return short, taggy


def _prepend_metadata(text: str, trigger_word: str, user_caption_anchor: str) -> str:
    caption = str(text or "").strip()
    pieces: list[str] = []
    for value in (trigger_word, user_caption_anchor):
        v = str(value or "").strip().strip(",")
        if v and v.lower() not in caption[: max(80, len(v) + 5)].lower():
            pieces.append(v)
    if pieces and caption:
        return ", ".join(pieces + [caption])
    return caption or ", ".join(pieces)


# -----------------------------------------------------------------------------
# Ollama HTTP helpers
# -----------------------------------------------------------------------------


def _normalize_ollama_url(value: str) -> str:
    return (str(value or DEFAULT_OLLAMA_URL).strip() or DEFAULT_OLLAMA_URL).rstrip("/")


def _http_json(method: str, url: str, payload: dict[str, Any] | None = None, timeout: float = 900.0) -> dict[str, Any]:
    data = None
    headers = {"Accept": "application/json"}
    if payload is not None:
        data = json.dumps(payload).encode("utf-8")
        headers["Content-Type"] = "application/json"
    req = urllib.request.Request(url, data=data, headers=headers, method=method.upper())
    try:
        with urllib.request.urlopen(req, timeout=float(timeout)) as resp:
            raw = resp.read().decode("utf-8", errors="replace")
            return json.loads(raw) if raw.strip() else {}
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace") if exc.fp else ""
        raise RuntimeError(f"HTTP {exc.code} from {url}: {body or exc.reason}") from exc
    except urllib.error.URLError as exc:
        reason = getattr(exc, "reason", exc)
        raise RuntimeError(f"Could not reach Ollama at {url}: {reason}") from exc


def _http_error_status(exc: BaseException) -> int | None:
    """Return an HTTP status carried by an exception or its causal chain."""
    current: BaseException | None = exc
    seen: set[int] = set()
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        for attribute in ("code", "status", "status_code"):
            value = getattr(current, attribute, None)
            try:
                if value is not None:
                    return int(value)
            except (TypeError, ValueError):
                pass
        current = current.__cause__ or current.__context__
    return None


def _validator_image_retry_caps(configured_max_size: int) -> tuple[int, ...]:
    configured = max(0, int(configured_max_size or 0))
    caps = [configured]
    if configured == 0:
        caps.extend((1024, 768, 512))
    else:
        caps.extend(cap for cap in (1024, 768, 512) if cap < configured)
    return tuple(caps)


def _is_explicit_abort(exc: BaseException) -> bool:
    return isinstance(exc, (KeyboardInterrupt, SystemExit, GeneratorExit)) or exc.__class__.__name__ in {
        "CancelledError",
        "InterruptProcessingException",
    }


def _ollama_options(num_predict: int, temperature: float, top_p: float, top_k: int, seed: int | None) -> dict[str, Any]:
    options: dict[str, Any] = {
        "num_predict": int(num_predict),
        "temperature": float(temperature),
        "top_p": float(top_p),
        "top_k": int(top_k),
    }
    if seed is not None and int(seed) >= 0:
        options["seed"] = int(seed)
    return options


def _extract_generate_text(data: dict[str, Any]) -> str:
    if not isinstance(data, dict):
        return ""
    return str(data.get("response") or data.get("thinking") or "").strip()


def _extract_chat_text(data: dict[str, Any]) -> str:
    if not isinstance(data, dict):
        return ""
    message = data.get("message") if isinstance(data.get("message"), dict) else {}
    return str(message.get("content") or data.get("response") or data.get("thinking") or "").strip()


def _summarize_ollama_response(data: Any) -> str:
    if not isinstance(data, dict):
        return str(type(data))
    parts = [f"keys={sorted(data.keys())}"]
    if "done_reason" in data:
        parts.append(f"done_reason={data.get('done_reason')}")
    if isinstance(data.get("message"), dict):
        msg = data["message"]
        parts.append(f"message_keys={sorted(msg.keys())}")
        parts.append(f"message_content_len={len(str(msg.get('content') or ''))}")
        if "thinking" in msg:
            parts.append(f"message_thinking_len={len(str(msg.get('thinking') or ''))}")
    if "response" in data:
        parts.append(f"response_len={len(str(data.get('response') or ''))}")
    if "eval_count" in data:
        parts.append(f"eval_count={data.get('eval_count')}")
    if "prompt_eval_count" in data:
        parts.append(f"prompt_eval_count={data.get('prompt_eval_count')}")
    return "; ".join(parts)


def _save_raw_response(
    raw_dir: Path | None,
    image_key: str,
    stage: str,
    data: dict[str, Any],
    *,
    overwrite: bool = True,
) -> str:
    if raw_dir is None:
        return ""
    raw_dir.mkdir(parents=True, exist_ok=True)
    path = raw_dir / f"{_safe_txt_stem(image_key)}__{stage}.json"
    if path.exists() and not overwrite:
        return ""
    path.write_text(json.dumps(_json_safe(data), indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return str(path)


def _ollama_generate_text(
    *,
    ollama_url: str,
    model: str,
    prompt: str,
    num_predict: int,
    temperature: float,
    top_p: float,
    top_k: int,
    seed: int | None,
    keep_loaded: bool,
    timeout: float,
) -> tuple[str, dict[str, Any]]:
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "think": False,
        "options": _ollama_options(num_predict, temperature, top_p, top_k, seed),
        "keep_alive": "5m" if bool(keep_loaded) else "0s",
    }
    data = _http_json("POST", f"{ollama_url}/api/generate", payload=payload, timeout=timeout)
    text = _extract_generate_text(data)
    if not text:
        raise RuntimeError(
            f"Ollama returned empty text for '{model}' via /api/generate. "
            f"max_new_tokens maps to num_predict; current value: {num_predict}. "
            f"{_summarize_ollama_response(data)}"
        )
    return text, data


def _ollama_chat_image(
    *,
    ollama_url: str,
    model: str,
    system_prompt: str,
    user_prompt: str,
    image_b64: str,
    num_predict: int,
    temperature: float,
    top_p: float,
    top_k: int,
    seed: int | None,
    keep_loaded: bool,
    timeout: float,
) -> tuple[str, dict[str, Any]]:
    payload = {
        "model": model,
        "stream": False,
        "messages": [
            {"role": "system", "content": str(system_prompt or "")},
            {"role": "user", "content": str(user_prompt or ""), "images": [image_b64]},
        ],
        "think": False,
        "options": _ollama_options(num_predict, temperature, top_p, top_k, seed),
        "keep_alive": "5m" if bool(keep_loaded) else "0s",
    }
    chat_data = _http_json("POST", f"{ollama_url}/api/chat", payload=payload, timeout=timeout)
    text = _extract_chat_text(chat_data)
    if text:
        return text, chat_data

    print(
        f"[JLC CaptionForge Orchestrator] /api/chat returned empty text for '{model}'. Trying /api/generate fallback. "
        f"{_summarize_ollama_response(chat_data)}",
        flush=True,
    )
    full_prompt = f"SYSTEM:\n{system_prompt}\n\nUSER:\n{user_prompt}" if str(system_prompt or "").strip() else user_prompt
    generate_payload = {
        "model": model,
        "prompt": full_prompt,
        "stream": False,
        "images": [image_b64],
        "think": False,
        "options": _ollama_options(num_predict, temperature, top_p, top_k, seed),
        "keep_alive": "5m" if bool(keep_loaded) else "0s",
    }
    gen_data = _http_json("POST", f"{ollama_url}/api/generate", payload=generate_payload, timeout=timeout)
    text = _extract_generate_text(gen_data)
    if text:
        combined = {"chat_response": chat_data, "generate_response": gen_data}
        return text, combined
    raise RuntimeError(
        f"Ollama returned empty text for '{model}' via both /api/chat and /api/generate. "
        f"max_new_tokens maps to num_predict; current value: {num_predict}. "
        f"chat_response: {_summarize_ollama_response(chat_data)} generate_response: {_summarize_ollama_response(gen_data)}"
    )


# -----------------------------------------------------------------------------
# Final export helpers
# -----------------------------------------------------------------------------


def _normalize_txt_export_format(export_format: str) -> str:
    fmt = str(export_format or "natural").strip().lower()
    if fmt in {"narrative", "natural"}:
        return "natural"
    if fmt in {"comma", "taggy"}:
        return "taggy"
    if fmt in {"both", "both_separate", "both separate"}:
        return "both_separate"
    return fmt or "natural"


def _split_tag_items(text: str) -> list[str]:
    raw = str(text or "").replace("\n", ",")
    items = [re.sub(r"\s+", " ", part).strip(" ,;:\t") for part in raw.split(",")]
    cleaned = [re.sub(r"[.!?]+$", "", item).strip() for item in items if item]
    return [item for item in cleaned if item]


def _dedupe_keep_order(items: list[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for item in items:
        key = re.sub(r"[^a-z0-9]+", " ", str(item).lower()).strip()
        if not key or key in seen:
            continue
        seen.add(key)
        out.append(item)
    return out


def _compact_taggy_caption(taggy: str, *, max_items: int = 64, max_chars: int = 1200) -> str:
    """Make a compact LoRA-style comma caption without adding new visual claims."""
    items = _dedupe_keep_order(_split_tag_items(taggy))

    # Drop model-output scaffolding that occasionally leaks into comma lists.
    banned_prefixes = (
        "taggy comma-list only",
        "caption",
        "the image shows",
        "this image shows",
        "there is",
        "there are",
    )
    cleaned: list[str] = []
    for item in items:
        low = item.lower().strip()
        if any(low.startswith(prefix) for prefix in banned_prefixes):
            item = re.sub(
                r"(?is)^(taggy comma-list only|caption|the image shows|this image shows|there is|there are)\s*[:\-]?\s*",
                "",
                item,
            ).strip(" ,;:")
        if item:
            cleaned.append(item)

    cleaned = _dedupe_keep_order(cleaned)[:max_items]
    out = ", ".join(cleaned)
    if max_chars > 0 and len(out) > max_chars:
        out = out[:max_chars].rsplit(",", 1)[0].strip(" ,")
    return out


def _compact_lora_short_caption(long_caption: str, taggy_caption: str, *, target_words: int = 100) -> str:
    """Create a shorter LoRA-length caption from the validated long caption.

    This is deterministic and conservative: it compresses existing validated text
    instead of asking another model to invent or rewrite details. The word target
    is soft so fallback output always ends at a natural sentence boundary.
    """
    text = _cleanup_single_paragraph(long_caption or "")
    if not text:
        return _compact_taggy_caption(taggy_caption, max_items=42, max_chars=0)

    sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", text) if s.strip()]
    kept: list[str] = []
    words = 0
    for sentence in sentences:
        sentence_words = sentence.split()
        if not sentence_words:
            continue
        if kept and words >= target_words:
            break
        kept.append(sentence)
        words += len(sentence_words)

    return (" ".join(kept) or text).strip()


def _normalize_ai_short_caption(short_caption: str) -> str:
    """Normalize generated SHORT text without imposing a hard length boundary."""
    return _cleanup_single_paragraph(short_caption)


def _write_final_txt_sidecars(
    image_path: Path,
    long_caption: str,
    short_caption: str,
    taggy_caption: str,
    export_format: str = "",
    *,
    overwrite: bool = True,
) -> list[str]:
    """Write CaptionForge 1.0 final sidecars beside the resolved source image.

    CaptionForge keeps the rich validated caption while also exporting
    a shorter LoRA-length caption and a compact taggy caption.
    """
    written: list[str] = []
    stem = image_path.stem
    parent = image_path.parent

    long_text = _cleanup_single_paragraph(long_caption)
    taggy_text = _compact_taggy_caption(taggy_caption)
    short_text = _cleanup_single_paragraph(short_caption) or _compact_lora_short_caption(long_text, taggy_text)

    targets = [
        (parent / f"{stem}_long.txt", long_text),
        (parent / f"{stem}_short.txt", short_text),
        (parent / f"{stem}_taggy.txt", taggy_text),
    ]

    for path, text in targets:
        if text:
            if path.exists() and not overwrite:
                continue
            _write_text(path, text)
            written.append(str(path))

    return written


def _final_sidecar_output_paths(
    image_path: Path,
    long_caption: str,
    short_caption: str,
    taggy_caption: str,
    *,
    enabled: bool,
) -> dict[str, str]:
    """Return the named final sidecar paths exposed to automation consumers."""
    if not enabled:
        return {"long": "", "short": "", "taggy": ""}

    parent = image_path.parent
    stem = image_path.stem
    return {
        "long": str(parent / f"{stem}_long.txt") if long_caption else "",
        "short": str(parent / f"{stem}_short.txt") if short_caption else "",
        "taggy": str(parent / f"{stem}_taggy.txt") if taggy_caption else "",
    }


def _make_final_failure_record(
    *,
    image_key: str,
    status: str,
    error: str,
    selected_caption_count: int,
    source_caption_families: list[str],
    trigger_word: str,
    user_caption_anchor: str,
    models: dict[str, str],
    image: str = "",
    error_stage: str = "",
    error_type: str = "",
    error_message: str = "",
) -> dict[str, Any]:
    return {
        "captionforge_pass": "D_FINAL_EXPORT",
        "engine": "jlc_captionforge_node",
        "engine_version": CAPTIONFORGE_NODE_VERSION,
        "image_key": image_key,
        "image": image,
        "status": status,
        "error": error,
        "error_stage": error_stage,
        "error_type": error_type,
        "error_message": error_message,
        "export_format": "",
        "final_caption": "",
        "long": "",
        "short": "",
        "taggy": "",
        "final_caption_long": "",
        "final_caption_short": "",
        "final_caption_natural": "",
        "final_caption_taggy": "",
        "fat_draft": "",
        "trigger_word": trigger_word,
        "user_caption_anchor": user_caption_anchor,
        "models": models,
        "selected_caption_count": int(selected_caption_count),
        "source_caption_families": source_caption_families,
        "outputs": {"long": "", "short": "", "taggy": ""},
        "sidecar_paths": [],
        "timestamp": datetime.now().isoformat(timespec="seconds"),
    }


def _concise_error_message(exc: BaseException, max_chars: int = 500) -> str:
    message = _normalize_text(str(exc)) or exc.__class__.__name__
    if len(message) > max_chars:
        return message[: max(1, max_chars - 1)].rstrip() + "…"
    return message


def _completed_final_records(path: Path, *, include_export_failures: bool = False) -> dict[str, dict[str, Any]]:
    """Return latest completed D_FINAL_EXPORT records keyed by image_key."""
    if not path.exists() or not path.is_file():
        return {}

    latest: dict[str, dict[str, Any]] = {}
    with path.open("r", encoding="utf-8") as ledger:
        for line_number, line in enumerate(ledger, start=1):
            text = line.strip()
            if not text:
                continue
            try:
                record = json.loads(text)
            except Exception as exc:
                print(
                    f"[JLC CaptionForge Orchestrator] WARNING: Ignoring unusable resume ledger line "
                    f"{path}:{line_number}: {_concise_error_message(exc)}",
                    flush=True,
                )
                continue
            if not isinstance(record, dict):
                continue
            if str(record.get("captionforge_pass") or "").strip() != "D_FINAL_EXPORT":
                continue
            image_key = str(record.get("image_key") or "").strip()
            if image_key:
                latest[image_key] = record

    completed: dict[str, dict[str, Any]] = {}
    for image_key, record in latest.items():
        if (str(record.get("status") or "").strip().lower() != "ok"
                and not (include_export_failures and record.get("error_stage") == "dataset_export")):
            continue
        long_caption = _normalize_text(
            record.get("long") or record.get("final_caption_long") or record.get("final_caption_natural")
        )
        short_caption = _normalize_text(record.get("short") or record.get("final_caption_short"))
        taggy_caption = _normalize_text(record.get("taggy") or record.get("final_caption_taggy"))
        if long_caption and short_caption and taggy_caption:
            completed[image_key] = record
    return completed


def _selected_export_caption(natural: str, taggy: str, export_format: str) -> str:
    fmt = _normalize_txt_export_format(export_format)
    if fmt == "taggy":
        return taggy or natural
    return natural or taggy


def _reset_outputs(paths: dict[str, str], overwrite: bool) -> None:
    if not overwrite:
        return
    # These are append-oriented semantic/audit logs. Current-run descriptors
    # such as output_paths.json are intentionally regenerated at completion and
    # are not part of this reset set.
    for key in ("fat_draft_jsonl", "fat_draft_prompt_jsonl", "validator_jsonl", "validator_prompt_jsonl", "taggy_jsonl", "taggy_prompt_jsonl", "final_jsonl"):
        p = Path(paths[key])
        if p.exists() and p.is_file():
            p.unlink()


# -----------------------------------------------------------------------------
# ComfyUI node
# -----------------------------------------------------------------------------


class JLC_CaptionForge:
    """Production B/C/D Orchestrator: draft -> VLM LONG -> SHORT/TAGGY."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "Input - captions JSONL": (
                    "STRING",
                    {"default": "", "multiline": False, "tooltip": "Pass A raw caption JSONL produced by CaptionForge Caption nodes."},
                ),
                "Input - image path": (
                    "STRING",
                    {"default": "", "multiline": False, "tooltip": "Image file/folder root used by the VLM validator to resolve source images."},
                ),
                "Input - include caption families": (
                    "STRING",
                    {"default": "joy,qwen,ollama", "multiline": False, "tooltip": "Comma-separated model_family values to use from Pass A. Use all or * to include everything."},
                ),
                "Input - max captions per family": (
                    "INT",
                    {"default": 5, "min": 0, "max": 50, "step": 1, "tooltip": "Maximum selected Pass A captions per model family. 0 means no per-family cap."},
                ),
                "Input - max total captions": (
                    "INT",
                    {"default": 20, "min": 0, "max": 100, "step": 1, "tooltip": "Maximum selected Pass A captions per image. 0 means no total cap."},
                ),
                "Output - folder": (
                    "STRING",
                    {"default": "", "multiline": False, "tooltip": "Output folder. Planner value overrides this when pipeline_plan is connected."},
                ),
                "Output - run name": (
                    "STRING",
                    {"default": "captionforge_run", "multiline": False, "tooltip": "Base name used for the run working folder and B/C/D JSONL/audit files."},
                ),
                "Output - overwrite outputs": ("BOOLEAN", {"default": True, "tooltip": "Replace run-level files that already use this folder and run name. Final image sidecars are also replaced when enabled."}),
                "Ollama - URL": (
                    "STRING",
                    {"default": DEFAULT_OLLAMA_URL, "multiline": False, "tooltip": "Local Ollama server URL."},
                ),
                "Ollama - keep loaded": (
                    "BOOLEAN",
                    {"default": True, "tooltip": "Ask Ollama to keep the most recently used model in memory between calls. Ollama makes the final residency decision."},
                ),
                "Ollama - request timeout seconds": (
                    "INT",
                    {"default": 1800, "min": 10, "max": 7200, "step": 10, "tooltip": "HTTP patience for Ollama calls. This does not affect caption quality."},
                ),
                "LoRA - trigger word": ("STRING", {"default": "", "multiline": False, "tooltip": "Optional LoRA trigger token or phrase preserved in final captions as training metadata."}),
                "LoRA - user caption anchor": ("STRING", {"default": "", "multiline": False, "tooltip": "Optional persistent caption/training anchor. In CaptionForge 1.x, a non-empty anchor is preserved in the final caption variants rather than treated as image evidence that the Validator may remove."}),
                "Cleanup - forbidden phrases": ("STRING", {"default": "", "multiline": True, "tooltip": "Standalone forbidden words/phrases, one per line. Planner values override this when connected."}),
                "Cleanup - replace pairs": ("STRING", {"default": "", "multiline": True, "tooltip": "Standalone boundary-safe old=>new replacements, one per line. Planner values override this when connected."}),
                "Fat Draft - model": (DISTILLER_MODEL_CHOICES, {"default": DEFAULT_DISTILLER_MODEL, "tooltip": "Concrete Ollama text-model tag for Pass B. Choose custom to enter another installed tag below."}),
                "Fat Draft - custom Ollama model": (
                    "STRING",
                    {"default": "", "multiline": False, "tooltip": "Used only when Fat Draft - model is Custom."},
                ),
                "Fat Draft - prompt": (
                    "STRING",
                    {"default": DEFAULT_FAT_DRAFT_INSTRUCTIONS, "multiline": True, "tooltip": "Instructions for the text-only fat draft LLM. Captions are appended automatically."},
                ),
                "Fat Draft - max caption chars": ("INT", {"default": 1536, "min": 0, "max": 12000, "step": 64, "tooltip": "Maximum characters kept from each source witness caption before Pass B. 0 keeps the complete caption."}),
                "Fat Draft - max new tokens": (
                    "INT",
                    {"default": 3096, "min": 64, "max": 12000, "step": 64, "tooltip": "Maximum Pass-B output-token budget sent to Ollama (num_predict)."},
                ),
                "Fat Draft - temperature": ("FLOAT", {"default": 0.24, "min": 0.0, "max": 2.0, "step": 0.01, "tooltip": "Pass-B variation level. Lower values are steadier; higher values permit more varied wording."}),
                "Fat Draft - top p": ("FLOAT", {"default": 0.90, "min": 0.0, "max": 1.0, "step": 0.01, "tooltip": "Pass-B nucleus-sampling limit. Lower values restrict the model to more likely tokens."}),
                "Fat Draft - top k": ("INT", {"default": 60, "min": 0, "max": 500, "step": 1, "tooltip": "Pass-B token-choice limit. Lower values are more restrictive; 0 lets the backend disable top-k filtering."}),
                "Validator - model": (VALIDATOR_MODEL_CHOICES, {"default": DEFAULT_VALIDATOR_MODEL, "tooltip": "Concrete Ollama vision-model tag for image-aware Pass C. Choose custom to enter another installed VLM tag below."}),
                "Validator - custom Ollama model": (
                    "STRING",
                    {"default": "", "multiline": False, "tooltip": "Used only when Validator - model is Custom."},
                ),
                "Validator - system prompt": (
                    "STRING",
                    {"default": DEFAULT_VALIDATOR_SYSTEM_PROMPT, "multiline": True, "tooltip": "System prompt for the image-aware VLM validator."},
                ),
                "Validator - prompt": (
                    "STRING",
                    {"default": DEFAULT_VALIDATOR_INSTRUCTIONS, "multiline": True, "tooltip": "Instructions for the image-aware VLM validator. The fat draft is appended automatically."},
                ),
                "Validator - max new tokens": (
                    "INT",
                    {"default": 2112, "min": 64, "max": 12000, "step": 64, "tooltip": "Maximum Pass-C output-token budget sent to Ollama (num_predict)."},
                ),
                "Validator - max image size": (
                    "INT",
                    {
                        "default": 1024,
                        "min": 0,
                        "max": 8192,
                        "step": 64,
                        "tooltip": (
                            "Longest image edge sent to the Validator VLM; aspect ratio is preserved. "
                            "0 disables resizing. The Pipeline Planner overrides this value when connected."
                        ),
                    },
                ),
                "Validator - temperature": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 2.0, "step": 0.01, "tooltip": "Pass-C variation level. Zero requests the most deterministic image-validation result."}),
                "Validator - top p": ("FLOAT", {"default": 0.92, "min": 0.0, "max": 1.0, "step": 0.01, "tooltip": "Pass-C nucleus-sampling limit. Lower values restrict the validator to more likely tokens."}),
                "Validator - top k": ("INT", {"default": 80, "min": 0, "max": 500, "step": 1, "tooltip": "Pass-C token-choice limit. Lower values are more restrictive; 0 lets the backend disable top-k filtering."}),
                "Formatter - model": (FORMAT_MODEL_CHOICES, {"default": DEFAULT_FORMAT_MODEL, "tooltip": "Concrete Ollama text-model tag for the Pass-D SHORT/TAGGY formatter. Choose custom to enter another installed tag below."}),
                "Formatter - custom Ollama model": (
                    "STRING",
                    {"default": "", "multiline": False, "tooltip": "Used only when Formatter - model is Custom."},
                ),
                "Formatter - prompt": (
                    "STRING",
                    {"default": DEFAULT_TAGGY_FORMATTER_INSTRUCTIONS, "multiline": True, "tooltip": "Instructions for the text-only short/taggy formatter. The validated paragraph is appended automatically."},
                ),
                "Formatter - max new tokens": (
                    "INT",
                    {"default": 3200, "min": 64, "max": 12000, "step": 64, "tooltip": "Maximum Pass-D output-token budget sent to Ollama (num_predict)."},
                ),
                "Formatter - temperature": ("FLOAT", {"default": 0.12, "min": 0.0, "max": 2.0, "step": 0.01, "tooltip": "Pass-D variation level. Lower values make SHORT/TAGGY formatting more consistent."}),
                "Formatter - top p": ("FLOAT", {"default": 0.88, "min": 0.0, "max": 1.0, "step": 0.01, "tooltip": "Pass-D nucleus-sampling limit. Lower values restrict the formatter to more likely tokens."}),
                "Formatter - top k": ("INT", {"default": 50, "min": 0, "max": 500, "step": 1, "tooltip": "Pass-D token-choice limit. Lower values are more restrictive; 0 lets the backend disable top-k filtering."}),
                "Audit - write prompt JSONL": ("BOOLEAN", {"default": False, "tooltip": "Write full B/C/D prompts to separate JSONL audit files. This can substantially increase output size."}),
                "Audit - preserve raw responses": ("BOOLEAN", {"default": False, "tooltip": "Keep unparsed B/C/D model responses in audit records for troubleshooting."}),
                "Final - TXT export format": (TXT_EXPORT_FORMATS, {"default": "natural", "tooltip": "Select the primary final_caption field in JSONL: natural uses LONG, taggy uses TAGGY, and both_separate keeps LONG primary while retaining separate named fields. TXT sidecars always include LONG, SHORT, and TAGGY files."}),
                "Final - write TXT sidecars": ("BOOLEAN", {"default": True, "tooltip": "Write LONG, SHORT, and TAGGY text variants beside each resolved source image, plus the selected plain .txt training sidecar."}),
                "Final - write JSONL": ("BOOLEAN", {"default": True, "tooltip": "Write the final run-level JSONL containing LONG, SHORT, and TAGGY captions for every processed image."}),
            },
            "optional": {
                **dataset_export_inputs(),
                "Input - single image": (
                    "IMAGE",
                    {"tooltip": "Optional IMAGE passthrough/reference for planned single-image workflows."},
                ),
                "pipeline_plan": (
                    "CAPTIONFORGE_PIPELINE_PLAN",
                    {"tooltip": "Connect the CaptionForge Pipeline Planner pipeline_plan output here."},
                ),
                "Distiller seed": (
                    "INT",
                    {"forceInput": True, "tooltip": "Optional standalone Pass-B seed. Planner seed overrides it when connected."},
                ),
                "Validator seed": (
                    "INT",
                    {"forceInput": True, "tooltip": "Optional standalone Pass-C seed. Planner seed overrides it when connected."},
                ),
                "Formatter seed": (
                    "INT",
                    {"forceInput": True, "tooltip": "Optional standalone Pass-D seed. Planner seed overrides it when connected."},
                ),
            },
        }

    RETURN_TYPES = ("STRING", "STRING", "STRING", "STRING", "STRING")
    RETURN_NAMES = ("long_captions", "short_captions", "taggy_captions", "final_records", "status")
    FUNCTION = "forge"
    CATEGORY = "Caption/CaptionForge"

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        return float("NaN")

    def forge(self, **kwargs):
        plan = normalize_captionforge_pipeline_plan(kwargs.get("pipeline_plan") or kwargs.get("captionforge_run_config"))

        output_dir = _resolve_output_dir(str(kwargs.get("Output - folder", "") or ""), plan)
        output_dir.mkdir(parents=True, exist_ok=True)
        run_name = _resolve_run_name(str(kwargs.get("Output - run name", "captionforge_run") or "captionforge_run"), plan)
        paths = _derive_paths(output_dir, run_name, plan)
        Path(paths["output_dir"]).mkdir(parents=True, exist_ok=True)
        overwrite = _safe_bool(
            _resolve_setting(plan, kwargs.get("Output - overwrite outputs"), "final.overwrite_outputs", "output.overwrite_outputs", "shared.overwrite_outputs", default=True),
            True,
        )
        _reset_outputs(paths, overwrite)

        caption_jsonl = _resolve_caption_jsonl(str(kwargs.get("Input - captions JSONL", "") or ""), plan, paths)
        caption_path = Path(caption_jsonl)
        if not caption_path.exists() or caption_path.is_dir():
            raise FileNotFoundError(f"Caption JSONL not found: {caption_path}")
        if not caption_path.read_text(encoding="utf-8").strip():
            raise FileNotFoundError(f"Caption JSONL is empty: {caption_path}")

        image_root = _resolve_image_root(str(kwargs.get("Input - image path", "") or ""), plan, caption_jsonl)
        opt_images_dir = Path(paths.get("opt_images_dir") or (Path(paths["working_dir"]) / "images" / "opt_images"))
        single_image_root = _save_single_image_inputs_for_validator(kwargs.get("Input - single image"), opt_images_dir)
        image_roots = _dedupe_paths([
            image_root,
            _plan_get(plan, "paths.image_root", "shared.image_root", "shared.input_path", "input_path", default=""),
            single_image_root,
            opt_images_dir,
            Path(caption_jsonl).resolve().parent,
        ])

        ollama_url = _normalize_ollama_url(
            str(
                _resolve_setting(
                    plan,
                    kwargs.get("Ollama - URL"),
                    "ollama.url",
                    default=DEFAULT_OLLAMA_URL,
                )
            )
        )
        keep_loaded = _safe_bool(
            _resolve_setting(
                plan,
                kwargs.get("Ollama - keep loaded"),
                "ollama.keep_loaded",
                default=True,
            ),
            True,
        )
        timeout = float(
            _coerce_int(
                _resolve_setting(
                    plan,
                    kwargs.get("Ollama - request timeout seconds"),
                    "ollama.request_timeout_seconds",
                    default=1800,
                ),
                1800,
                10,
                7200,
            )
        )
        # Standalone mode keeps the Orchestrator's global audit widgets. In planned
        # mode, Pass B/C/D resolve audit settings independently from their
        # Planner namespaces.
        audit = _resolve_stage_audit_settings(
            plan,
            kwargs.get("Audit - write prompt JSONL", False),
            kwargs.get("Audit - preserve raw responses", False),
        )
        fat_write_prompts = audit["fat_write_prompts"]
        fat_preserve_raw = audit["fat_preserve_raw"]
        val_write_prompts = audit["val_write_prompts"]
        val_preserve_raw = audit["val_preserve_raw"]
        fmt_write_prompts = audit["fmt_write_prompts"]
        fmt_preserve_raw = audit["fmt_preserve_raw"]
        raw_dir = Path(paths["raw_response_dir"])

        trigger_word = _normalize_text(_resolve_setting(plan, kwargs.get("LoRA - trigger word"), "shared.trigger_word", "lora.trigger_word", "trigger_word", default=""))
        user_caption_anchor = _normalize_text(_resolve_setting(plan, kwargs.get("LoRA - user caption anchor"), "shared.user_caption_anchor", "lora.user_caption_anchor", "user_caption_anchor", default=""))
        forbidden_phrases = normalize_forbidden_phrases(
            _resolve_setting(
                plan,
                kwargs.get("Cleanup - forbidden phrases", ""),
                "cleanup.forbidden_phrases",
                default=[],
            )
        )
        replace_pairs = normalize_replace_pairs(
            _resolve_setting(
                plan,
                kwargs.get("Cleanup - replace pairs", ""),
                "cleanup.replace_pairs",
                default=[],
            )
        )
        cleanup_contract = {
            "forbidden_phrases": list(forbidden_phrases),
            "replace_pairs": [{"old": old, "new": new} for old, new in replace_pairs],
            "matching": "boundary_safe_case_insensitive",
            "order": ["replace_pairs", "forbidden_phrases", "normalize_whitespace_punctuation"],
        }
        validator_max_size = _coerce_int(
            _resolve_setting(
                plan,
                kwargs.get("Validator - max image size"),
                "caption_settings.max_size",
                "caption_generation.max_size",
                "pass_a_settings.max_size",
                "shared.max_size",
                default=1024,
            ),
            1024,
            0,
            8192,
        )

        fat_model_choice = _resolve_setting(
            plan,
            kwargs.get("Fat Draft - model"),
            "distiller.model",
            "pass_b.model",
            "distiller.ollama_model",
            "pass_b.ollama_model",
            "pass_b_distiller.model",
            "pass_b_distiller.ollama_model",
            default=DEFAULT_DISTILLER_MODEL,
        )
        fat_model = _resolve_ollama_model_name(fat_model_choice, kwargs.get("Fat Draft - custom Ollama model"), DEFAULT_DISTILLER_MODEL)
        fat_seed = _normalize_optional_seed(
            _resolve_setting(
                plan,
                kwargs.get("Distiller seed", kwargs.get("Fat Draft - base seed")),
                "distiller.seed",
                "pass_b.seed",
                "pass_b_distiller.seed",
                "distiller.base_seed",
                "pass_b.base_seed",
                "pass_b_distiller.base_seed",
                default=None,
            )
        )
        fat_num = _coerce_int(_resolve_setting(plan, kwargs.get("Fat Draft - max new tokens"), "distiller.num_predict", "pass_b.num_predict", "pass_b_distiller.num_predict", default=3096), 3096, 64, 12000)
        fat_temp = _coerce_float(_resolve_setting(plan, kwargs.get("Fat Draft - temperature"), "distiller.temperature", "pass_b.temperature", "pass_b_distiller.temperature", default=0.24), 0.24, 0.0, 2.0)
        fat_top_p = _coerce_float(_resolve_setting(plan, kwargs.get("Fat Draft - top p"), "distiller.top_p", "pass_b.top_p", "pass_b_distiller.top_p", default=0.90), 0.90, 0.0, 1.0)
        fat_top_k = _coerce_int(_resolve_setting(plan, kwargs.get("Fat Draft - top k"), "distiller.top_k", "pass_b.top_k", "pass_b_distiller.top_k", default=60), 60, 0, 500)
        fat_prompt_instructions = str(
            _resolve_setting(
                plan,
                kwargs.get("Fat Draft - prompt"),
                "distiller.prompt",
                "pass_b.prompt",
                "pass_b_distiller.prompt",
                default=DEFAULT_FAT_DRAFT_INSTRUCTIONS,
            )
        )
        max_caption_chars = _resolve_fat_draft_max_caption_chars(
            plan,
            kwargs.get("Fat Draft - max caption chars"),
        )

        val_model_choice = _resolve_setting(
            plan,
            kwargs.get("Validator - model"),
            "validator.model",
            "pass_c.model",
            "validator.ollama_model",
            "pass_c.ollama_model",
            "pass_c_vlm_validator.model",
            "pass_c_vlm_validator.ollama_model",
            default=DEFAULT_VALIDATOR_MODEL,
        )
        val_model = _resolve_ollama_model_name(val_model_choice, kwargs.get("Validator - custom Ollama model"), DEFAULT_VALIDATOR_MODEL)
        val_seed = _normalize_optional_seed(
            _resolve_setting(
                plan,
                kwargs.get("Validator seed", kwargs.get("Validator - base seed")),
                "validator.seed",
                "pass_c.seed",
                "pass_c_vlm_validator.seed",
                "validator.base_seed",
                "pass_c.base_seed",
                "pass_c_vlm_validator.base_seed",
                default=None,
            )
        )
        val_num = _coerce_int(_resolve_setting(plan, kwargs.get("Validator - max new tokens"), "validator.num_predict", "pass_c.num_predict", "pass_c_vlm_validator.num_predict", default=2112), 2112, 64, 12000)
        val_temp = _coerce_float(_resolve_setting(plan, kwargs.get("Validator - temperature"), "validator.temperature", "pass_c.temperature", "pass_c_vlm_validator.temperature", default=0.0), 0.0, 0.0, 2.0)
        val_top_p = _coerce_float(_resolve_setting(plan, kwargs.get("Validator - top p"), "validator.top_p", "pass_c.top_p", "pass_c_vlm_validator.top_p", default=0.92), 0.92, 0.0, 1.0)
        val_top_k = _coerce_int(_resolve_setting(plan, kwargs.get("Validator - top k"), "validator.top_k", "pass_c.top_k", "pass_c_vlm_validator.top_k", default=80), 80, 0, 500)
        val_system = str(
            _resolve_setting(
                plan,
                kwargs.get("Validator - system prompt"),
                "validator.system_prompt",
                "pass_c.system_prompt",
                "pass_c_vlm_validator.system_prompt",
                default=DEFAULT_VALIDATOR_SYSTEM_PROMPT,
            )
        )
        val_prompt_instructions = str(
            _resolve_setting(
                plan,
                kwargs.get("Validator - prompt"),
                "validator.prompt",
                "pass_c.prompt",
                "pass_c_vlm_validator.prompt",
                default=DEFAULT_VALIDATOR_INSTRUCTIONS,
            )
        )

        fmt_model_choice = _resolve_setting(
            plan,
            kwargs.get("Formatter - model"),
            "formatter.model",
            "format.model",
            "pass_d.model",
            "formatter.ollama_model",
            "format.ollama_model",
            "pass_d.ollama_model",
            "pass_d_formatter.model",
            "pass_d_formatter.ollama_model",
            default=DEFAULT_FORMAT_MODEL,
        )
        fmt_model = _resolve_ollama_model_name(fmt_model_choice, kwargs.get("Formatter - custom Ollama model"), DEFAULT_FORMAT_MODEL)
        fmt_seed = _normalize_optional_seed(
            _resolve_setting(
                plan,
                kwargs.get("Formatter seed", kwargs.get("Formatter - base seed")),
                "formatter.seed",
                "format.seed",
                "pass_d.seed",
                "pass_d_formatter.seed",
                "formatter.base_seed",
                "format.base_seed",
                "pass_d.base_seed",
                default=None,
            )
        )
        fmt_num = _coerce_int(_resolve_setting(plan, kwargs.get("Formatter - max new tokens"), "formatter.num_predict", "format.num_predict", "pass_d.num_predict", "pass_d_formatter.num_predict", default=3200), 3200, 64, 12000)
        fmt_temp = _coerce_float(_resolve_setting(plan, kwargs.get("Formatter - temperature"), "formatter.temperature", "format.temperature", "pass_d.temperature", "pass_d_formatter.temperature", default=0.12), 0.12, 0.0, 2.0)
        fmt_top_p = _coerce_float(_resolve_setting(plan, kwargs.get("Formatter - top p"), "formatter.top_p", "format.top_p", "pass_d.top_p", "pass_d_formatter.top_p", default=0.88), 0.88, 0.0, 1.0)
        fmt_top_k = _coerce_int(_resolve_setting(plan, kwargs.get("Formatter - top k"), "formatter.top_k", "format.top_k", "pass_d.top_k", "pass_d_formatter.top_k", default=50), 50, 0, 500)
        fmt_prompt_instructions = str(
            _resolve_setting(
                plan,
                kwargs.get("Formatter - prompt"),
                "formatter.prompt",
                "format.prompt",
                "pass_d.prompt",
                "pass_d_formatter.prompt",
                default=DEFAULT_TAGGY_FORMATTER_INSTRUCTIONS,
            )
        )

        txt_export_format = str(_resolve_setting(plan, kwargs.get("Final - TXT export format"), "final.txt_export_format", default="natural") or "natural")
        txt_export_format = _normalize_txt_export_format(txt_export_format)
        write_txt = _safe_bool(_resolve_setting(plan, kwargs.get("Final - write TXT sidecars"), "final.write_txt_sidecars", default=True), True)
        write_jsonl = _safe_bool(_resolve_setting(plan, kwargs.get("Final - write JSONL"), "final.write_jsonl", default=True), True)

        # Even blank/disabled Planner values own the decision; older plans default
        # to disabled instead of falling through to Orchestrator-local widgets.
        export = normalize_export_settings(
            plan.get("dataset_export") if _planner_overrides(plan)
            else export_settings_from_widgets(kwargs)
        )
        training_root = dataset_root(export, paths["output_root"])
        if export["enabled"]:
            prepare_dataset_root(training_root, image_root)

        records = _read_jsonl(caption_path)
        grouped = _group_records_by_image(records)
        include_families = str(kwargs.get("Input - include caption families", "joy,qwen,ollama") or "joy,qwen,ollama")
        max_per_family = _coerce_int(kwargs.get("Input - max captions per family", 5), 5, 0, 50)
        max_total = _coerce_int(kwargs.get("Input - max total captions", 20), 20, 0, 100)

        final_records: list[dict[str, Any]] = []
        ok = 0
        failed = 0
        resume_skipped = 0

        _evict_python_models_before_ollama_if_needed("JLC CaptionForge Orchestrator")

        final_jsonl_path = Path(paths["final_jsonl"])
        if write_jsonl:
            final_jsonl_path.parent.mkdir(parents=True, exist_ok=True)
            if overwrite:
                final_jsonl_path.write_text("", encoding="utf-8")

        completed_records = _completed_final_records(final_jsonl_path, include_export_failures=export["enabled"]) if not overwrite else {}
        if not overwrite:
            print(
                f"[JLC CaptionForge Orchestrator] Resume mode: completed={len(completed_records)} "
                f"ledger={final_jsonl_path}",
                flush=True,
            )

        models_for_record = {"fat_draft": fat_model, "validator": val_model, "formatter": fmt_model}

        run_config = {
            "planner_connected": _planner_overrides(plan),
            "cleanup": cleanup_contract,
            "models": models_for_record,
        }
        try:
            _write_json(Path(paths["run_config_json"]), run_config)
        except Exception:
            pass

        def finish_dataset_export(record: dict, image_path: Path, prepared_image=None) -> dict:
            try:
                record["long"] = record.get("long") or record.get("final_caption_long") or record.get("final_caption_natural") or ""
                record["short"] = record.get("short") or record.get("final_caption_short") or ""
                record["taggy"] = record.get("taggy") or record.get("final_caption_taggy") or ""
                record["dataset_export"] = export_pair(
                    source=image_path, image_key=record["image_key"], input_root=image_root,
                    root=training_root, settings=export, validator_max_size=validator_max_size,
                    captions=record, overwrite=overwrite, prepared_image=prepared_image,
                    write_variants=write_txt,
                )
                exported_image = Path(record["dataset_export"]["image"])
                record["outputs"] = _final_sidecar_output_paths(
                    exported_image, record["long"], record["short"], record["taggy"], enabled=write_txt,
                )
                record["sidecar_paths"] = list(record["dataset_export"]["variants"].values())
                record["status"] = "ok"
                for key in ("error", "error_stage", "error_type", "error_message"):
                    record.pop(key, None)
            except Exception as exc:
                if _is_explicit_abort(exc):
                    raise
                record.update(status="error", error_stage="dataset_export", error_type=type(exc).__name__,
                              error_message=_concise_error_message(exc))
                record["dataset_export"] = {"status": "error", "message": _concise_error_message(exc)}
            return record

        def process_image_group(image_key: str, image_records: list[dict[str, Any]]) -> dict[str, Any]:
            selected: list[dict[str, Any]] = []
            source_families: list[str] = []
            image_path: Path | None = None
            error_stage = "caption_selection"
            try:
                selected = _select_caption_records(
                    image_records,
                    include_families=include_families,
                    max_per_family=max_per_family,
                    max_total=max_total,
                )
                source_families = sorted({(_family_of(r) or "unknown") for r in selected})

                if not selected:
                    return _make_final_failure_record(
                        image_key=image_key,
                        status="error",
                        error="no_usable_captions_selected",
                        error_stage="caption_selection",
                        error_type="NoUsableCaptions",
                        error_message="No usable captions were selected for this image.",
                        selected_caption_count=0,
                        source_caption_families=[],
                        trigger_word=trigger_word,
                        user_caption_anchor=user_caption_anchor,
                        models=models_for_record,
                    )

                error_stage = "image_path_resolution"
                image_path = _resolve_image_path_for_group(
                    selected,
                    image_roots,
                    optional_image_root=opt_images_dir,
                )
                if image_path is None:
                    searched = "; ".join(str(p) for p in image_roots)
                    return _make_final_failure_record(
                        image_key=image_key,
                        status="error",
                        error=f"could_not_resolve_image_path; searched={searched}",
                        error_stage="image_path_resolution",
                        error_type="ImageResolutionError",
                        error_message="Could not resolve the source image path.",
                        selected_caption_count=len(selected),
                        source_caption_families=source_families,
                        trigger_word=trigger_word,
                        user_caption_anchor=user_caption_anchor,
                        models=models_for_record,
                    )

                error_stage = "distiller"
                caption_blocks = _build_caption_blocks(selected, max_caption_chars)
                fat_prompt = _build_fat_draft_prompt(
                    fat_prompt_instructions,
                    caption_blocks,
                    trigger_word,
                    user_caption_anchor,
                )
                if fat_write_prompts:
                    _write_jsonl(
                        Path(paths["fat_draft_prompt_jsonl"]),
                        [{"image_key": image_key, "prompt": fat_prompt, "model": fat_model, "stage": "B_FAT_DRAFT"}],
                        append=True,
                    )

                fat_text, fat_raw = _ollama_generate_text(
                    ollama_url=ollama_url,
                    model=fat_model,
                    prompt=fat_prompt,
                    num_predict=fat_num,
                    temperature=fat_temp,
                    top_p=fat_top_p,
                    top_k=fat_top_k,
                    seed=fat_seed,
                    keep_loaded=keep_loaded,
                    timeout=timeout,
                )
                fat_text = apply_cleanup_contract(
                    _cleanup_single_paragraph(fat_text), forbidden_phrases, replace_pairs
                )
                fat_raw_path = _save_raw_response(
                    raw_dir if fat_preserve_raw else None,
                    image_key,
                    "01_fat_draft_raw",
                    fat_raw,
                    overwrite=overwrite,
                )
                _write_jsonl(
                    Path(paths["fat_draft_jsonl"]),
                    [
                        asdict(
                            StageRecord(
                                captionforge_pass="B_FAT_DRAFT",
                                engine="jlc_captionforge_node",
                                engine_version=CAPTIONFORGE_NODE_VERSION,
                                image_key=image_key,
                                image=str(image_path),
                                status="ok" if fat_text else "empty",
                                text=fat_text,
                                model=fat_model,
                                prompt=fat_prompt if fat_write_prompts else "",
                                params={"max_new_tokens": fat_num, "temperature": fat_temp, "top_p": fat_top_p, "top_k": fat_top_k, "seed": fat_seed},
                                source={"selected_caption_count": len(selected), "raw_response_path": fat_raw_path},
                                timestamp=datetime.now().isoformat(timespec="seconds"),
                            )
                        )
                    ],
                    append=True,
                )

                error_stage = "validator"
                val_prompt = _build_validator_prompt(
                    val_prompt_instructions,
                    fat_text,
                    trigger_word,
                    user_caption_anchor,
                )
                if val_write_prompts:
                    _write_jsonl(
                        Path(paths["validator_prompt_jsonl"]),
                        [{"image_key": image_key, "prompt": val_prompt, "system_prompt": val_system, "model": val_model, "stage": "C_VLM_VALIDATED_FINAL"}],
                        append=True,
                    )

                retry_caps = _validator_image_retry_caps(validator_max_size)
                prepared_images: list[tuple[Image.Image, tuple[int, int]]] = []
                natural = ""
                val_raw: dict[str, Any] = {}
                for attempt_index, attempt_cap in enumerate(retry_caps):
                    error_stage = "validator_image_preparation"
                    if export["enabled"] and attempt_index == 0:
                        image_b64 = _pil_to_base64_png(image_path, max_size=attempt_cap, prepared_images=prepared_images)
                    else:
                        image_b64 = _pil_to_base64_png(image_path, max_size=attempt_cap)
                    error_stage = "validator"
                    try:
                        natural, val_raw = _ollama_chat_image(
                            ollama_url=ollama_url,
                            model=val_model,
                            system_prompt=val_system,
                            user_prompt=val_prompt,
                            image_b64=image_b64,
                            num_predict=val_num,
                            temperature=val_temp,
                            top_p=val_top_p,
                            top_k=val_top_k,
                            seed=val_seed,
                            keep_loaded=keep_loaded,
                            timeout=timeout,
                        )
                        break
                    except Exception as exc:
                        if _is_explicit_abort(exc):
                            raise
                        next_index = attempt_index + 1
                        if _http_error_status(exc) != 413 or next_index >= len(retry_caps):
                            raise
                        retry_cap = retry_caps[next_index]
                        print(
                            f"[JLC CaptionForge Orchestrator] Validator HTTP 413 for image_key={image_key}; "
                            f"retrying with max_size={retry_cap}",
                            flush=True,
                        )

                natural = apply_cleanup_contract(
                    _prepend_metadata(_cleanup_single_paragraph(natural), trigger_word, user_caption_anchor),
                    forbidden_phrases,
                    replace_pairs,
                )
                val_raw_path = _save_raw_response(
                    raw_dir if val_preserve_raw else None,
                    image_key,
                    "02_validator_raw",
                    val_raw,
                    overwrite=overwrite,
                )
                _write_jsonl(
                    Path(paths["validator_jsonl"]),
                    [
                        asdict(
                            StageRecord(
                                captionforge_pass="C_VLM_VALIDATED_FINAL",
                                engine="jlc_captionforge_node",
                                engine_version=CAPTIONFORGE_NODE_VERSION,
                                image_key=image_key,
                                image=str(image_path),
                                status="ok" if natural else "empty",
                                text=natural,
                                model=val_model,
                                prompt=val_prompt if val_write_prompts else "",
                                params={"max_new_tokens": val_num, "temperature": val_temp, "top_p": val_top_p, "top_k": val_top_k, "seed": val_seed},
                                source={"fat_draft": fat_text, "raw_response_path": val_raw_path},
                                timestamp=datetime.now().isoformat(timespec="seconds"),
                            )
                        )
                    ],
                    append=True,
                )

                error_stage = "formatter"
                fmt_prompt = _build_taggy_prompt(fmt_prompt_instructions, natural, trigger_word, user_caption_anchor)
                if fmt_write_prompts:
                    _write_jsonl(
                        Path(paths["taggy_prompt_jsonl"]),
                        [{"image_key": image_key, "prompt": fmt_prompt, "model": fmt_model, "stage": "D_FORMAT_TAGGY"}],
                        append=True,
                    )

                formatter_text, fmt_raw = _ollama_generate_text(
                    ollama_url=ollama_url,
                    model=fmt_model,
                    prompt=fmt_prompt,
                    num_predict=fmt_num,
                    temperature=fmt_temp,
                    top_p=fmt_top_p,
                    top_k=fmt_top_k,
                    seed=fmt_seed,
                    keep_loaded=keep_loaded,
                    timeout=timeout,
                )
                short_candidate, taggy_candidate = _parse_formatter_derivatives(formatter_text)
                taggy = _compact_taggy_caption(
                    _prepend_metadata(taggy_candidate, trigger_word, user_caption_anchor)
                )
                short = _normalize_ai_short_caption(
                    _prepend_metadata(short_candidate, trigger_word, user_caption_anchor)
                )
                if not short:
                    short = _compact_lora_short_caption(natural, taggy)
                natural = apply_cleanup_contract(natural, forbidden_phrases, replace_pairs)
                short = apply_cleanup_contract(short, forbidden_phrases, replace_pairs)
                taggy = apply_cleanup_contract(taggy, forbidden_phrases, replace_pairs)
                fmt_raw_path = _save_raw_response(
                    raw_dir if fmt_preserve_raw else None,
                    image_key,
                    "03_taggy_raw",
                    fmt_raw,
                    overwrite=overwrite,
                )
                _write_jsonl(
                    Path(paths["taggy_jsonl"]),
                    [
                        asdict(
                            StageRecord(
                                captionforge_pass="D_FORMAT_TAGGY",
                                engine="jlc_captionforge_node",
                                engine_version=CAPTIONFORGE_NODE_VERSION,
                                image_key=image_key,
                                image=str(image_path),
                                status="ok" if taggy else "empty",
                                text=taggy,
                                model=fmt_model,
                                prompt=fmt_prompt if fmt_write_prompts else "",
                                params={"max_new_tokens": fmt_num, "temperature": fmt_temp, "top_p": fmt_top_p, "top_k": fmt_top_k, "seed": fmt_seed},
                                source={
                                    "validated_natural": natural,
                                    "short_caption": short,
                                    "raw_response_path": fmt_raw_path,
                                },
                                timestamp=datetime.now().isoformat(timespec="seconds"),
                            )
                        )
                    ],
                    append=True,
                )

                error_stage = "final_export"
                export_caption = _selected_export_caption(natural, taggy, txt_export_format)
                is_ok = bool(natural and taggy)
                final_record = {
                    "captionforge_pass": "D_FINAL_EXPORT",
                    "engine": "jlc_captionforge_node",
                    "engine_version": CAPTIONFORGE_NODE_VERSION,
                    "image_key": image_key,
                    "image": str(image_path),
                    "status": "ok" if is_ok else "error",
                    "export_format": txt_export_format,
                    "final_caption": export_caption,
                    "long": natural,
                    "short": short,
                    "taggy": taggy,
                    "final_caption_long": natural,
                    "final_caption_short": short,
                    "final_caption_natural": natural,
                    "final_caption_taggy": taggy,
                    "fat_draft": fat_text,
                    "trigger_word": trigger_word,
                    "user_caption_anchor": user_caption_anchor,
                    "cleanup": cleanup_contract,
                    "models": models_for_record,
                    "selected_caption_count": len(selected),
                    "source_caption_families": source_families,
                    "outputs": _final_sidecar_output_paths(
                        image_path,
                        natural,
                        short,
                        taggy,
                        enabled=write_txt and not export["enabled"],
                    ),
                    "sidecar_paths": [],
                    "timestamp": datetime.now().isoformat(timespec="seconds"),
                }

                if export["enabled"]:
                    if not is_ok:
                        return final_record
                    # Reuse the validator pixels only when no further resampling
                    # is needed; otherwise resize directly from the source.
                    prepared = None
                    if prepared_images:
                        candidate, source_size = prepared_images[0]
                        # A too-small image must be reported as an export failure,
                        # after preserving the already computed captions.
                        try:
                            if candidate.size == export_dimensions(source_size, export["max_size"] or validator_max_size, export["divisor"]):
                                prepared = candidate
                        except ValueError:
                            pass
                    return finish_dataset_export(final_record, image_path, prepared)
                if write_txt and (natural or short or taggy):
                    final_record["sidecar_paths"] = _write_final_txt_sidecars(
                        image_path,
                        natural,
                        short,
                        taggy,
                        txt_export_format,
                        overwrite=overwrite,
                    )
                return final_record
            except Exception as exc:
                if _is_explicit_abort(exc):
                    raise
                return _make_final_failure_record(
                    image_key=image_key,
                    image=str(image_path or ""),
                    status="error",
                    error="processing_exception",
                    error_stage=error_stage,
                    error_type=exc.__class__.__name__,
                    error_message=_concise_error_message(exc),
                    selected_caption_count=len(selected),
                    source_caption_families=source_families,
                    trigger_word=trigger_word,
                    user_caption_anchor=user_caption_anchor,
                    models=models_for_record,
                )

        for image_index, (image_key, image_records) in enumerate(grouped.items(), start=1):
            if image_key in completed_records and not export["enabled"]:
                resume_skipped += 1
                ok += 1
                final_records.append(completed_records[image_key])
                print(
                    f"[JLC CaptionForge Orchestrator] Resume skip {image_index}/{len(grouped)} image_key={image_key}",
                    flush=True,
                )
                continue

            if image_key in completed_records and export["enabled"]:
                # Enabling export on an already captioned run needs no model calls.
                final_record = dict(completed_records[image_key])
                resolved = _resolve_image_path_for_group(image_records, image_roots, optional_image_root=opt_images_dir)
                if resolved is None:
                    final_record.update(status="error", error_stage="dataset_export", error_message="Source image is missing.")
                else:
                    final_record = finish_dataset_export(final_record, resolved)
                resume_skipped += 1
            else:
                final_record = process_image_group(image_key, image_records)
            final_records.append(final_record)
            is_ok = str(final_record.get("status") or "").strip().lower() == "ok"
            ok += int(is_ok)
            failed += int(not is_ok)

            if write_jsonl:
                try:
                    _write_jsonl(final_jsonl_path, [final_record], append=True)
                except Exception as exc:
                    if _is_explicit_abort(exc):
                        raise
                    if is_ok:
                        ok -= 1
                        failed += 1
                        final_record = _make_final_failure_record(
                            image_key=image_key,
                            image=str(final_record.get("image") or ""),
                            status="error",
                            error="processing_exception",
                            error_stage="final_ledger_write",
                            error_type=exc.__class__.__name__,
                            error_message=_concise_error_message(exc),
                            selected_caption_count=int(final_record.get("selected_caption_count") or 0),
                            source_caption_families=list(final_record.get("source_caption_families") or []),
                            trigger_word=trigger_word,
                            user_caption_anchor=user_caption_anchor,
                            models=models_for_record,
                        )
                        final_records[-1] = final_record
                        is_ok = False
                    else:
                        print(
                            f"[JLC CaptionForge Orchestrator] WARNING: Could not append failure record "
                            f"for image_key={image_key}: {_concise_error_message(exc)}",
                            flush=True,
                        )

            if is_ok:
                print(
                    f"[JLC CaptionForge Orchestrator] processed {image_index}/{len(grouped)} image_key={image_key} "
                    f"captions={final_record.get('selected_caption_count', 0)} "
                    f"natural_len={len(str(final_record.get('long') or ''))} "
                    f"taggy_len={len(str(final_record.get('taggy') or ''))}",
                    flush=True,
                )
            else:
                print(
                    f"[JLC CaptionForge Orchestrator] failed {image_index}/{len(grouped)} image_key={image_key} "
                    f"stage={final_record.get('error_stage') or 'final_output'} "
                    f"error_type={final_record.get('error_type') or final_record.get('error') or 'Error'} "
                    f"message={final_record.get('error_message') or final_record.get('error') or ''}",
                    flush=True,
                )

        output_paths = dict(paths)
        output_paths.update(
            {
                "caption_jsonl": caption_jsonl,
                "pass_a_jsonl": caption_jsonl,
                "image_root": image_root,
                "image_roots": [str(p) for p in image_roots],
                "working_images_dir": str(paths.get("working_images_dir") or (Path(paths["working_dir"]) / "images")),
                "opt_images_dir": str(opt_images_dir),
                "planner_connected": _planner_overrides(plan),
                "fat_draft_model_resolved": fat_model,
                "validator_model_resolved": val_model,
                "formatter_model_resolved": fmt_model,
                "txt_export_format": txt_export_format,
                "final_ok": ok,
                "final_failed": failed,
                "resume_skipped": resume_skipped,
                "cleanup": cleanup_contract,
            }
        )
        output_paths_json = json.dumps(_json_safe(output_paths), ensure_ascii=False, indent=2)
        try:
            Path(paths["output_paths_json"]).parent.mkdir(parents=True, exist_ok=True)
            Path(paths["output_paths_json"]).write_text(output_paths_json + "\n", encoding="utf-8")
        except Exception:
            pass

        long_captions = "\n\n".join(str(record.get("long") or "") for record in final_records)
        short_captions = "\n\n".join(str(record.get("short") or "") for record in final_records)
        taggy_captions = "\n\n".join(str(record.get("taggy") or "") for record in final_records)
        final_records_json = json.dumps(
            _json_safe({"records": final_records, "run_outputs": output_paths}),
            ensure_ascii=False,
            indent=2,
        )
        status = (
            f"[JLC CaptionForge Orchestrator v{CAPTIONFORGE_NODE_VERSION}] complete | "
            f"planner_connected={_planner_overrides(plan)} | images={len(grouped)} | "
            f"final_ok={ok} final_failed={failed} resume_skipped={resume_skipped} | "
            f"models fat={fat_model} validator={val_model} formatter={fmt_model} | "
            f"run={run_name} output={output_dir}"
        )
        print(status, flush=True)
        return (long_captions, short_captions, taggy_captions, final_records_json, status)


NODE_CLASS_MAPPINGS = {
    "JLC_CaptionForge": JLC_CaptionForge,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "JLC_CaptionForge": "\u2003JLC CaptionForge Orchestrator",
}
