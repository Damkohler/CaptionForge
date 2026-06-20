"""
JLC CaptionForge Node — ComfyUI Capstone Node Wrapper

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
    - The **JLC CaptionForge Node** is the production capstone for the current
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
            • Pass D taggy-format construction with a text-only Ollama LLM
            • final natural/taggy TXT and JSONL export
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
              -> D_FORMAT_TAGGY               text-only formatter
              -> final TXT/JSONL export

    - The VLM-validated natural paragraph is the natural final caption. The
      formatter pass must not rewrite that natural paragraph; it only derives a
      comma-separated taggy caption from it.

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

    - The Formatter LLM sees only the validated paragraph. It returns one taggy
      comma-separated caption.

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

    - The capstone keeps orchestration and ComfyUI UI concerns separate from
      model-specific caption generation.

    - The node prioritizes auditable local caption refinement, explicit model
      selection, deterministic output paths, and high-value LoRA captions.

- ⚠️ Development Status
    - This is release-candidate CaptionForge capstone infrastructure.
    - Output schema details may evolve as the release candidate is tested across
      local Ollama/VLM installations.

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
    "name": "JLC CaptionForge Node",
    "version": CAPTIONFORGE_VERSION,
    "author": "J. L. Córdova",
    "description": (
        "Release-candidate CaptionForge capstone node. Consumes Pass A raw caption "
        "JSONL directly or through a CAPTIONFORGE_PIPELINE_PLAN, builds a text-only "
        "fat draft with an Ollama LLM, validates it against the image with an Ollama "
        "VLM to produce the natural final caption, derives a taggy comma-list with a "
        "format model, and exports deterministic TXT/JSONL artifacts. The natural "
        "caption is the VLM-validated output directly; the formatter pass does not "
        "rewrite the natural paragraph."
    ),
}

import base64
import io
import json
import random
import re
import urllib.error
import urllib.request
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path, PureWindowsPath
from typing import Any

import numpy as np
import torch
from PIL import Image

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


CAPTIONFORGE_NODE_VERSION = "0.2.0"
SEED_MODES = ["fixed", "increment", "decrement", "random"]
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

DEFAULT_FAT_DRAFT_INSTRUCTIONS = """/no_think

You are a detail-preserving caption merger for LoRA dataset preparation.

You receive multiple captions of the same image. You do NOT see the image.

Task:
Merge all non-contradictory caption details into one deliberately over-complete draft caption.

Rules:
- Do not validate against the image.
- Do not decide that details are false just because they appear once.
- Do not summarize aggressively.
- Preserve concrete details from all captions.
- Split contradictions by choosing cautious wording or listing the alternative only when needed.
- Prefer specific visual language over generic language.
- Keep visible body, clothing, material, accessory, color, pose, lighting, style, and framing details.
- Preserve doll-like, glossy/plastic-like, material, garment-construction, body-shape, and facial-feature details when present.
- Use neutral dataset-caption language, including visible sensual styling or revealing clothing when present.
- Do not add details absent from the captions.
- Treat subject names or trigger-like identity tokens as optional identity labels. Preserve them only when they appear consistently in the captions; do not let them replace visible description.
- Output only one paragraph, no notes, no JSON."""

DEFAULT_VALIDATOR_SYSTEM_PROMPT = (
    "/no_think\n"
    "You are a direct image validation engine. Inspect the image and answer only with the requested caption."
)

DEFAULT_VALIDATOR_INSTRUCTIONS = """/no_think

Look at the image and validate this draft caption.

Task:
Return a corrected caption paragraph that keeps only image-supported details.

Rules:
- Output only the corrected caption.
- One paragraph.
- No reasoning, no notes, no JSON.
- Keep all true visible details from the draft.
- Delete unsupported details.
- Correct small visible errors.
- Do not add new details unless needed to correct an error already present.
- Preserve useful LoRA details: subject, face, hair, eyes, makeup, lips, skin texture, pose, body shape, outfit, accessories, materials, colors, lighting, background, framing, and visual style.
- Visible sensual styling, revealing clothing, cleavage, thighs, bare skin, swimwear, lingerie, or body-shape details may be described neutrally when present.
- Do not invent hidden anatomy, unseen clothing, explicit acts, or details contradicted by the image."""

DEFAULT_TAGGY_FORMATTER_INSTRUCTIONS = """/no_think

You are a LoRA caption format converter.

The validated paragraph is already the natural-language final caption. Do not rewrite it.

Task:
Create one TAGGY caption from the validated paragraph.

Rules:
- Output only the taggy comma-separated caption.
- Use only details already present in the validated paragraph.
- Preserve concrete LoRA-useful details.
- Do not add new details.
- Do not mention this process.
- Do not output markdown.
- Do not include a TAGGY: label.
- Keep the result as a comma-separated list, not full prose."""

_SUPPORTED_IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff"}


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
        print(f"[JLC CaptionForge Node] Could not read Ollama model config {path}: {exc}", flush=True)

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

    def pick(names: tuple[str, ...], fallback: Path) -> str:
        for name in names:
            value = planned_paths.get(name)
            if str(value or "").strip():
                return str(value).strip()
        return str(fallback)

    return {
        "output_dir": pick(("output_dir",), output_dir),
        "run_name": run_name,
        "caption_jsonl": pick(("caption_jsonl", "pass_a_jsonl"), output_dir / f"{run_name}__A_RAW_CAPTIONS.jsonl"),
        "pass_a_jsonl": pick(("pass_a_jsonl", "caption_jsonl"), output_dir / f"{run_name}__A_RAW_CAPTIONS.jsonl"),
        "fat_draft_jsonl": pick(("fat_draft_jsonl", "distiller_jsonl"), output_dir / f"{run_name}__B_FAT_DRAFT.jsonl"),
        "fat_draft_prompt_jsonl": pick(("fat_draft_prompt_jsonl", "distiller_prompt_jsonl"), output_dir / f"{run_name}__B_FAT_DRAFT_prompts.jsonl"),
        "validator_jsonl": pick(("validator_jsonl",), output_dir / f"{run_name}__C_VLM_VALIDATED_FINAL.jsonl"),
        "validator_prompt_jsonl": pick(("validator_prompt_jsonl",), output_dir / f"{run_name}__C_VLM_VALIDATOR_prompts.jsonl"),
        "taggy_jsonl": pick(("taggy_jsonl", "formatter_jsonl"), output_dir / f"{run_name}__D_FORMAT_TAGGY.jsonl"),
        "taggy_prompt_jsonl": pick(("taggy_prompt_jsonl", "formatter_prompt_jsonl"), output_dir / f"{run_name}__D_FORMAT_TAGGY_prompts.jsonl"),
        "final_jsonl": pick(("final_jsonl",), output_dir / f"{run_name}__E_FINAL_EXPORT.jsonl"),
        "final_txt_dir": pick(("final_txt_dir",), output_dir / f"{run_name}__TXT"),
        "output_paths_json": pick(("output_paths_json",), output_dir / f"{run_name}__output_paths.json"),
        "raw_response_dir": pick(("raw_response_dir",), output_dir / f"{run_name}__raw_responses"),
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


def _save_single_image_inputs_for_validator(image_tensor: Any, output_dir: Path) -> str:
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
    print(f"[JLC CaptionForge Node] Saved optional IMAGE input(s) for validator: {image_dir}", flush=True)
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
    return stem or "image"


def _record_image_key(record: dict[str, Any], fallback_index: int = 0) -> str:
    for key in ("image_key", "image", "source_image", "filename"):
        value = str(record.get(key) or "").strip()
        if value:
            base = _basename_cross_platform(value)
            if base:
                return Path(base).stem or base
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


def _resolve_image_path_for_group(records: list[dict[str, Any]], image_root: str) -> Path | None:
    values: list[Any] = []
    for record in records:
        values.extend([
            record.get("image_resolved_path"),
            record.get("image_path"),
            record.get("source_path"),
            record.get("image"),
            record.get("image_key"),
        ])
    for value in values:
        for candidate in _candidate_image_paths(image_root, value):
            if candidate.exists() and candidate.is_file():
                return candidate
    return None


def _pil_to_base64_png(path: Path) -> str:
    with Image.open(path) as img:
        rgb = img.convert("RGB")
        buf = io.BytesIO()
        rgb.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("ascii")


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
    instr = _append_optional_lora_guidance(str(instructions or DEFAULT_TAGGY_FORMATTER_INSTRUCTIONS), trigger_word, user_caption_anchor)
    return f"{instr}\n\nValidated paragraph:\n{validated_caption}\n\nTaggy comma-list only:"


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


def _save_raw_response(raw_dir: Path | None, image_key: str, stage: str, data: dict[str, Any]) -> str:
    if raw_dir is None:
        return ""
    raw_dir.mkdir(parents=True, exist_ok=True)
    path = raw_dir / f"{_safe_txt_stem(image_key)}__{stage}.json"
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
        f"[JLC CaptionForge Node] /api/chat returned empty text for '{model}'. Trying /api/generate fallback. "
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


def _selected_export_caption(natural: str, taggy: str, export_format: str) -> str:
    fmt = str(export_format or "natural").strip().lower()
    if fmt == "taggy":
        return taggy or natural
    return natural or taggy


def _reset_outputs(paths: dict[str, str], overwrite: bool) -> None:
    if not overwrite:
        return
    for key in ("fat_draft_jsonl", "fat_draft_prompt_jsonl", "validator_jsonl", "validator_prompt_jsonl", "taggy_jsonl", "taggy_prompt_jsonl", "final_jsonl"):
        p = Path(paths[key])
        if p.exists() and p.is_file():
            p.unlink()


# -----------------------------------------------------------------------------
# ComfyUI node
# -----------------------------------------------------------------------------


class JLC_CaptionForge:
    """CaptionForge capstone node: fat draft -> VLM natural final -> taggy formatter."""

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
                    {"default": "captionforge_run", "multiline": False, "tooltip": "Run-root used for B/C/D/E JSONL and TXT artifacts."},
                ),
                "Output - overwrite outputs": ("BOOLEAN", {"default": True}),
                "Ollama - URL": (
                    "STRING",
                    {"default": DEFAULT_OLLAMA_URL, "multiline": False, "tooltip": "Local Ollama server URL."},
                ),
                "Ollama - keep loaded": (
                    "BOOLEAN",
                    {"default": True, "tooltip": "Pass keep_alive to Ollama. Ollama ultimately owns model residency."},
                ),
                "Ollama - request timeout seconds": (
                    "INT",
                    {"default": 1800, "min": 10, "max": 7200, "step": 10, "tooltip": "HTTP patience for Ollama calls. This does not affect caption quality."},
                ),
                "LoRA - trigger word": ("STRING", {"default": "", "multiline": False}),
                "LoRA - user caption anchor": ("STRING", {"default": "", "multiline": False}),
                "Fat Draft - model": (DISTILLER_MODEL_CHOICES, {"default": DEFAULT_DISTILLER_MODEL}),
                "Fat Draft - custom Ollama model": (
                    "STRING",
                    {"default": "", "multiline": False, "tooltip": "Used only when Fat Draft - model is Custom."},
                ),
                "Fat Draft - prompt": (
                    "STRING",
                    {"default": DEFAULT_FAT_DRAFT_INSTRUCTIONS, "multiline": True, "tooltip": "Instructions for the text-only fat draft LLM. Captions are appended automatically."},
                ),
                "Fat Draft - base seed": ("INT", {"default": 1, "min": -1, "max": MAX_SEED_32, "step": 1}),
                "Fat Draft - seed mode": (SEED_MODES, {"default": "fixed"}),
                "Fat Draft - max caption chars": ("INT", {"default": 1536, "min": 0, "max": 12000, "step": 64}),
                "Fat Draft - max new tokens": (
                    "INT",
                    {"default": 5000, "min": 64, "max": 12000, "step": 64, "tooltip": "Maps to Ollama num_predict."},
                ),
                "Fat Draft - temperature": ("FLOAT", {"default": 0.12, "min": 0.0, "max": 2.0, "step": 0.01}),
                "Fat Draft - top p": ("FLOAT", {"default": 0.88, "min": 0.0, "max": 1.0, "step": 0.01}),
                "Fat Draft - top k": ("INT", {"default": 50, "min": 0, "max": 500, "step": 1}),
                "Validator - model": (VALIDATOR_MODEL_CHOICES, {"default": DEFAULT_VALIDATOR_MODEL}),
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
                "Validator - base seed": ("INT", {"default": 1, "min": -1, "max": MAX_SEED_32, "step": 1}),
                "Validator - seed mode": (SEED_MODES, {"default": "fixed"}),
                "Validator - max new tokens": (
                    "INT",
                    {"default": 5000, "min": 64, "max": 12000, "step": 64, "tooltip": "Maps to Ollama num_predict."},
                ),
                "Validator - temperature": ("FLOAT", {"default": 0.05, "min": 0.0, "max": 2.0, "step": 0.01}),
                "Validator - top p": ("FLOAT", {"default": 0.88, "min": 0.0, "max": 1.0, "step": 0.01}),
                "Validator - top k": ("INT", {"default": 50, "min": 0, "max": 500, "step": 1}),
                "Formatter - model": (FORMAT_MODEL_CHOICES, {"default": DEFAULT_FORMAT_MODEL}),
                "Formatter - custom Ollama model": (
                    "STRING",
                    {"default": "", "multiline": False, "tooltip": "Used only when Formatter - model is Custom."},
                ),
                "Formatter - prompt": (
                    "STRING",
                    {"default": DEFAULT_TAGGY_FORMATTER_INSTRUCTIONS, "multiline": True, "tooltip": "Instructions for the text-only taggy formatter. The validated paragraph is appended automatically."},
                ),
                "Formatter - base seed": ("INT", {"default": 1, "min": -1, "max": MAX_SEED_32, "step": 1}),
                "Formatter - seed mode": (SEED_MODES, {"default": "fixed"}),
                "Formatter - max new tokens": (
                    "INT",
                    {"default": 3200, "min": 64, "max": 12000, "step": 64, "tooltip": "Maps to Ollama num_predict."},
                ),
                "Formatter - temperature": ("FLOAT", {"default": 0.12, "min": 0.0, "max": 2.0, "step": 0.01}),
                "Formatter - top p": ("FLOAT", {"default": 0.88, "min": 0.0, "max": 1.0, "step": 0.01}),
                "Formatter - top k": ("INT", {"default": 50, "min": 0, "max": 500, "step": 1}),
                "Audit - write prompt JSONL": ("BOOLEAN", {"default": False}),
                "Audit - preserve raw responses": ("BOOLEAN", {"default": False}),
                "Final - TXT export format": (TXT_EXPORT_FORMATS, {"default": "natural"}),
                "Final - write TXT sidecars": ("BOOLEAN", {"default": True}),
                "Final - write JSONL": ("BOOLEAN", {"default": True}),
            },
            "optional": {
                "Input - single image": (
                    "IMAGE",
                    {"tooltip": "Optional IMAGE passthrough/reference for planned single-image workflows."},
                ),
                "pipeline_plan": (
                    "CAPTIONFORGE_PIPELINE_PLAN",
                    {"tooltip": "Connect the CaptionForge Pipeline Planner pipeline_plan output here."},
                ),
            },
        }

    RETURN_TYPES = ("STRING", "STRING", "STRING", "STRING", "STRING")
    RETURN_NAMES = ("natural_captions", "taggy_captions", "final_jsonl_records", "output_paths_json", "status")
    FUNCTION = "forge"
    CATEGORY = "Captioning/CaptionForge"

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        return float("NaN")

    def forge(self, **kwargs):
        plan = normalize_captionforge_pipeline_plan(kwargs.get("pipeline_plan") or kwargs.get("captionforge_run_config"))

        output_dir = _resolve_output_dir(str(kwargs.get("Output - folder", "") or ""), plan)
        output_dir.mkdir(parents=True, exist_ok=True)
        run_name = _resolve_run_name(str(kwargs.get("Output - run name", "captionforge_run") or "captionforge_run"), plan)
        paths = _derive_paths(output_dir, run_name, plan)
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
        single_image_root = _save_single_image_inputs_for_validator(kwargs.get("Input - single image"), output_dir)
        if single_image_root:
            image_root = single_image_root

        ollama_url = _normalize_ollama_url(str(kwargs.get("Ollama - URL", DEFAULT_OLLAMA_URL)))
        keep_loaded = _safe_bool(kwargs.get("Ollama - keep loaded", True), True)
        timeout = float(_coerce_int(kwargs.get("Ollama - request timeout seconds", 1800), 1800, 10, 7200))
        write_prompts = _safe_bool(kwargs.get("Audit - write prompt JSONL", False), False)
        preserve_raw = _safe_bool(kwargs.get("Audit - preserve raw responses", False), False)
        raw_dir = Path(paths["raw_response_dir"]) if preserve_raw else None

        trigger_word = _normalize_text(_resolve_setting(plan, kwargs.get("LoRA - trigger word"), "shared.trigger_word", "lora.trigger_word", "trigger_word", default=""))
        user_caption_anchor = _normalize_text(_resolve_setting(plan, kwargs.get("LoRA - user caption anchor"), "shared.user_caption_anchor", "lora.user_caption_anchor", "user_caption_anchor", default=""))

        fat_model_choice = _resolve_setting(
            plan,
            kwargs.get("Fat Draft - model"),
            "distiller.model",
            "pass_b.model",
            "distiller.ollama_model",
            "pass_b.ollama_model",
            default=DEFAULT_DISTILLER_MODEL,
        )
        fat_model = _resolve_ollama_model_name(fat_model_choice, kwargs.get("Fat Draft - custom Ollama model"), DEFAULT_DISTILLER_MODEL)
        fat_seed = _seed_for_stage(
            _resolve_setting(plan, kwargs.get("Fat Draft - base seed"), "distiller.base_seed", "pass_b.base_seed", default=1),
            _resolve_setting(plan, kwargs.get("Fat Draft - seed mode"), "distiller.seed_mode", "pass_b.seed_mode", default="fixed"),
            0,
        )
        fat_num = _coerce_int(_resolve_setting(plan, kwargs.get("Fat Draft - max new tokens"), "distiller.num_predict", "pass_b.num_predict", default=5000), 5000, 64, 12000)
        fat_temp = _coerce_float(_resolve_setting(plan, kwargs.get("Fat Draft - temperature"), "distiller.temperature", "pass_b.temperature", default=0.12), 0.12, 0.0, 2.0)
        fat_top_p = _coerce_float(_resolve_setting(plan, kwargs.get("Fat Draft - top p"), "distiller.top_p", "pass_b.top_p", default=0.88), 0.88, 0.0, 1.0)
        fat_top_k = _coerce_int(_resolve_setting(plan, kwargs.get("Fat Draft - top k"), "distiller.top_k", "pass_b.top_k", default=50), 50, 0, 500)
        fat_prompt_instructions = str(kwargs.get("Fat Draft - prompt") or DEFAULT_FAT_DRAFT_INSTRUCTIONS)
        max_caption_chars = _coerce_int(kwargs.get("Fat Draft - max caption chars", 1536), 1536, 0, 12000)

        val_model_choice = _resolve_setting(
            plan,
            kwargs.get("Validator - model"),
            "validator.model",
            "pass_c.model",
            "validator.ollama_model",
            "pass_c.ollama_model",
            default=DEFAULT_VALIDATOR_MODEL,
        )
        val_model = _resolve_ollama_model_name(val_model_choice, kwargs.get("Validator - custom Ollama model"), DEFAULT_VALIDATOR_MODEL)
        val_seed = _seed_for_stage(
            _resolve_setting(plan, kwargs.get("Validator - base seed"), "validator.base_seed", "pass_c.base_seed", default=1),
            _resolve_setting(plan, kwargs.get("Validator - seed mode"), "validator.seed_mode", "pass_c.seed_mode", default="fixed"),
            1,
        )
        val_num = _coerce_int(_resolve_setting(plan, kwargs.get("Validator - max new tokens"), "validator.num_predict", "pass_c.num_predict", default=5000), 5000, 64, 12000)
        val_temp = _coerce_float(_resolve_setting(plan, kwargs.get("Validator - temperature"), "validator.temperature", "pass_c.temperature", default=0.05), 0.05, 0.0, 2.0)
        val_top_p = _coerce_float(_resolve_setting(plan, kwargs.get("Validator - top p"), "validator.top_p", "pass_c.top_p", default=0.88), 0.88, 0.0, 1.0)
        val_top_k = _coerce_int(_resolve_setting(plan, kwargs.get("Validator - top k"), "validator.top_k", "pass_c.top_k", default=50), 50, 0, 500)
        val_system = str(kwargs.get("Validator - system prompt") or DEFAULT_VALIDATOR_SYSTEM_PROMPT)
        val_prompt_instructions = str(kwargs.get("Validator - prompt") or DEFAULT_VALIDATOR_INSTRUCTIONS)

        fmt_model_choice = _resolve_setting(
            plan,
            kwargs.get("Formatter - model"),
            "formatter.model",
            "format.model",
            "pass_d.model",
            "formatter.ollama_model",
            "format.ollama_model",
            default=DEFAULT_FORMAT_MODEL,
        )
        fmt_model = _resolve_ollama_model_name(fmt_model_choice, kwargs.get("Formatter - custom Ollama model"), DEFAULT_FORMAT_MODEL)
        fmt_seed = _seed_for_stage(
            _resolve_setting(plan, kwargs.get("Formatter - base seed"), "formatter.base_seed", "format.base_seed", "pass_d.base_seed", default=1),
            _resolve_setting(plan, kwargs.get("Formatter - seed mode"), "formatter.seed_mode", "format.seed_mode", "pass_d.seed_mode", default="fixed"),
            2,
        )
        fmt_num = _coerce_int(_resolve_setting(plan, kwargs.get("Formatter - max new tokens"), "formatter.num_predict", "format.num_predict", "pass_d.num_predict", default=3200), 3200, 64, 12000)
        fmt_temp = _coerce_float(_resolve_setting(plan, kwargs.get("Formatter - temperature"), "formatter.temperature", "format.temperature", "pass_d.temperature", default=0.12), 0.12, 0.0, 2.0)
        fmt_top_p = _coerce_float(_resolve_setting(plan, kwargs.get("Formatter - top p"), "formatter.top_p", "format.top_p", "pass_d.top_p", default=0.88), 0.88, 0.0, 1.0)
        fmt_top_k = _coerce_int(_resolve_setting(plan, kwargs.get("Formatter - top k"), "formatter.top_k", "format.top_k", "pass_d.top_k", default=50), 50, 0, 500)
        fmt_prompt_instructions = str(kwargs.get("Formatter - prompt") or DEFAULT_TAGGY_FORMATTER_INSTRUCTIONS)

        txt_export_format = str(_resolve_setting(plan, kwargs.get("Final - TXT export format"), "final.txt_export_format", "final.caption_style", default="natural") or "natural")
        if txt_export_format == "comma":
            txt_export_format = "taggy"
        write_txt = _safe_bool(_resolve_setting(plan, kwargs.get("Final - write TXT sidecars"), "final.write_txt_sidecars", default=True), True)
        write_jsonl = _safe_bool(_resolve_setting(plan, kwargs.get("Final - write JSONL"), "final.write_jsonl", default=True), True)

        records = _read_jsonl(caption_path)
        grouped = _group_records_by_image(records)
        include_families = str(kwargs.get("Input - include caption families", "joy,qwen,ollama") or "joy,qwen,ollama")
        max_per_family = _coerce_int(kwargs.get("Input - max captions per family", 5), 5, 0, 50)
        max_total = _coerce_int(kwargs.get("Input - max total captions", 20), 20, 0, 100)

        final_records: list[dict[str, Any]] = []
        natural_blocks: list[str] = []
        taggy_blocks: list[str] = []
        ok = 0
        failed = 0

        if write_jsonl:
            Path(paths["final_jsonl"]).parent.mkdir(parents=True, exist_ok=True)
            if overwrite:
                Path(paths["final_jsonl"]).write_text("", encoding="utf-8")

        for image_index, (image_key, image_records) in enumerate(grouped.items(), start=1):
            selected = _select_caption_records(
                image_records,
                include_families=include_families,
                max_per_family=max_per_family,
                max_total=max_total,
            )
            if not selected:
                failed += 1
                print(f"[JLC CaptionForge Node] No usable captions selected for image_key={image_key}", flush=True)
                continue
            image_path = _resolve_image_path_for_group(selected, image_root)
            if image_path is None:
                failed += 1
                print(f"[JLC CaptionForge Node] Could not resolve image path for image_key={image_key} under {image_root}", flush=True)
                continue

            caption_blocks = _build_caption_blocks(selected, max_caption_chars)
            fat_prompt = _build_fat_draft_prompt(fat_prompt_instructions, caption_blocks, trigger_word, user_caption_anchor)
            if write_prompts:
                _write_jsonl(Path(paths["fat_draft_prompt_jsonl"]), [{"image_key": image_key, "prompt": fat_prompt, "model": fat_model, "stage": "B_FAT_DRAFT"}], append=True)

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
            fat_text = _cleanup_single_paragraph(fat_text)
            fat_raw_path = _save_raw_response(raw_dir, image_key, "01_fat_draft_raw", fat_raw)
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
                            prompt=fat_prompt if write_prompts else "",
                            params={"max_new_tokens": fat_num, "temperature": fat_temp, "top_p": fat_top_p, "top_k": fat_top_k, "seed": fat_seed},
                            source={"selected_caption_count": len(selected), "raw_response_path": fat_raw_path},
                            timestamp=datetime.now().isoformat(timespec="seconds"),
                        )
                    )
                ],
                append=True,
            )

            image_b64 = _pil_to_base64_png(image_path)
            val_prompt = _build_validator_prompt(val_prompt_instructions, fat_text, trigger_word, user_caption_anchor)
            if write_prompts:
                _write_jsonl(Path(paths["validator_prompt_jsonl"]), [{"image_key": image_key, "prompt": val_prompt, "system_prompt": val_system, "model": val_model, "stage": "C_VLM_VALIDATED_FINAL"}], append=True)

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
            natural = _prepend_metadata(_cleanup_single_paragraph(natural), trigger_word, user_caption_anchor)
            val_raw_path = _save_raw_response(raw_dir, image_key, "02_validator_raw", val_raw)
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
                            prompt=val_prompt if write_prompts else "",
                            params={"max_new_tokens": val_num, "temperature": val_temp, "top_p": val_top_p, "top_k": val_top_k, "seed": val_seed},
                            source={"fat_draft": fat_text, "raw_response_path": val_raw_path},
                            timestamp=datetime.now().isoformat(timespec="seconds"),
                        )
                    )
                ],
                append=True,
            )

            fmt_prompt = _build_taggy_prompt(fmt_prompt_instructions, natural, trigger_word, user_caption_anchor)
            if write_prompts:
                _write_jsonl(Path(paths["taggy_prompt_jsonl"]), [{"image_key": image_key, "prompt": fmt_prompt, "model": fmt_model, "stage": "D_FORMAT_TAGGY"}], append=True)

            taggy, fmt_raw = _ollama_generate_text(
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
            taggy = _prepend_metadata(_cleanup_taggy(taggy), trigger_word, user_caption_anchor)
            fmt_raw_path = _save_raw_response(raw_dir, image_key, "03_taggy_raw", fmt_raw)
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
                            prompt=fmt_prompt if write_prompts else "",
                            params={"max_new_tokens": fmt_num, "temperature": fmt_temp, "top_p": fmt_top_p, "top_k": fmt_top_k, "seed": fmt_seed},
                            source={"validated_natural": natural, "raw_response_path": fmt_raw_path},
                            timestamp=datetime.now().isoformat(timespec="seconds"),
                        )
                    )
                ],
                append=True,
            )

            export_caption = _selected_export_caption(natural, taggy, txt_export_format)
            is_ok = bool(natural and taggy)
            ok += int(is_ok)
            failed += int(not is_ok)
            if natural:
                natural_blocks.append(natural)
            if taggy:
                taggy_blocks.append(taggy)

            final_record = {
                "captionforge_pass": "E_FINAL_EXPORT",
                "engine": "jlc_captionforge_node",
                "engine_version": CAPTIONFORGE_NODE_VERSION,
                "image_key": image_key,
                "image": str(image_path),
                "status": "ok" if is_ok else "error",
                "export_format": txt_export_format,
                "final_caption": export_caption,
                "final_caption_natural": natural,
                "final_caption_taggy": taggy,
                "fat_draft": fat_text,
                "trigger_word": trigger_word,
                "user_caption_anchor": user_caption_anchor,
                "models": {"fat_draft": fat_model, "validator": val_model, "formatter": fmt_model},
                "selected_caption_count": len(selected),
                "source_caption_families": sorted({(_family_of(r) or "unknown") for r in selected}),
                "timestamp": datetime.now().isoformat(timespec="seconds"),
            }
            final_records.append(final_record)

            if write_txt and export_caption:
                stem = _safe_txt_stem(image_path)
                txt_dir = Path(paths["final_txt_dir"])
                if txt_export_format == "both_separate":
                    _write_text(txt_dir / f"{stem}.txt", natural)
                    _write_text(txt_dir / f"{stem}.taggy.txt", taggy)
                else:
                    _write_text(txt_dir / f"{stem}.txt", export_caption)

            if write_jsonl:
                _write_jsonl(Path(paths["final_jsonl"]), [final_record], append=True)

            print(
                f"[JLC CaptionForge Node] processed {image_index}/{len(grouped)} image_key={image_key} "
                f"captions={len(selected)} natural_len={len(natural)} taggy_len={len(taggy)}",
                flush=True,
            )

        output_paths = dict(paths)
        output_paths.update(
            {
                "caption_jsonl": caption_jsonl,
                "pass_a_jsonl": caption_jsonl,
                "image_root": image_root,
                "planner_connected": _planner_overrides(plan),
                "fat_draft_model_resolved": fat_model,
                "validator_model_resolved": val_model,
                "formatter_model_resolved": fmt_model,
                "txt_export_format": txt_export_format,
                "final_ok": ok,
                "final_failed": failed,
            }
        )
        output_paths_json = json.dumps(_json_safe(output_paths), ensure_ascii=False, indent=2)
        try:
            Path(paths["output_paths_json"]).parent.mkdir(parents=True, exist_ok=True)
            Path(paths["output_paths_json"]).write_text(output_paths_json + "\n", encoding="utf-8")
        except Exception:
            pass

        final_jsonl_records = "\n".join(json.dumps(_json_safe(r), ensure_ascii=False) for r in final_records)
        status = (
            f"[JLC CaptionForge Node v{CAPTIONFORGE_NODE_VERSION}] complete | "
            f"planner_connected={_planner_overrides(plan)} | images={len(grouped)} | "
            f"final_ok={ok} final_failed={failed} | "
            f"models fat={fat_model} validator={val_model} formatter={fmt_model} | "
            f"run={run_name} output={output_dir}"
        )
        print(status, flush=True)
        return ("\n\n".join(natural_blocks), "\n\n".join(taggy_blocks), final_jsonl_records, output_paths_json, status)


NODE_CLASS_MAPPINGS = {
    "JLC_CaptionForge": JLC_CaptionForge,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "JLC_CaptionForge": "\u2003JLC CaptionForge Node",
}
