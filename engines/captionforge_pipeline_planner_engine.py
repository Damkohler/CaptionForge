"""
CaptionForge Pipeline Planner Engine

- CaptionForge
  - This engine is part of **CaptionForge**, a model-agnostic captioning
    framework for ComfyUI developed by **J. L. Córdova**.

  - Repository:
    https://github.com/Damkohler/CaptionForge

- CaptionForge focuses on practical dataset-captioning infrastructure for
  LoRA dataset preparation, using multi-engine caption generation, JSONL
  audit trails, claim extraction and refinement, text-LLM distillation,
  image-aware VLM validation, and consensus-oriented caption improvement
  to produce grounded, auditable training captions.

- Engine Purpose
    - The **CaptionForge Pipeline Planner Engine** builds reusable
      `CAPTIONFORGE_PIPELINE_PLAN` dictionaries for CaptionForge nodes.

    - It centralizes shared run configuration so individual node wrappers do not
      each need to solve routing, output-path derivation, Pass A run counts,
      seed scheduling, sampling schedules, or downstream engine defaults.

    - The planner coordinates:
            • input image path and optional single-image passthrough state
            • recursive folder traversal
            • filename glob filtering
            • output directory and run name
            • overwrite policy
            • Pass A witness run counts
            • per-run seed values
            • temperature, top-p, and top-k schedules
            • shared max image size and token budget
            • LoRA trigger word
            • user caption anchor
            • Pass B distiller defaults
            • Pass C VLM validator defaults
            • Pass D final export options
            • derived JSONL, TXT, readable-sidecar, prompt-log, and config paths

- CaptionForge Pipeline Role
    - This engine supports the full active CaptionForge workflow:

            Pipeline Planner
              -> Pass A caption witness nodes
              -> Pass B_DISTILL text-LLM distillation
              -> Pass C_VLM_VALIDATED image-aware validation
              -> Pass D final TXT/JSONL export

    - The planner does not caption images by itself. It creates the shared plan
      consumed by the node wrappers and capstone orchestration.

- Execution Model
    - The planner emits plain Python dictionaries that can also be serialized as
      JSON for auditability and debugging.

    - Existing helper functions normalize plan payloads, extract per-model Pass A
      plans, expand repeated witness runs, and preserve compatibility aliases
      used by older internal callers.

    - `PIPELINE_PLAN_VERSION` is a schema/version marker for the plan payload,
      separate from the project release version.

- Design Philosophy
    - CaptionForge should have one owner for run shape and path derivation.

    - Caption witness nodes should focus on caption generation, not on inventing
      their own folder layout, sampling schedules, or downstream pass settings.

    - The planner favors explicit dictionaries, predictable filenames, and
      auditable JSONL-oriented handoff between passes.

- Development Status
    - CaptionForge v0.1.0 experimental developer-preview infrastructure.
    - Plan schema, supported witness families, and downstream defaults may evolve
      before a stable CaptionForge release.

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
    "name": "CaptionForge Pipeline Planner Engine",
    "version": CAPTIONFORGE_VERSION,
    "author": "J. L. Córdova",
    "description": (
        "Reusable CaptionForge planner engine that builds "
        "CAPTIONFORGE_PIPELINE_PLAN dictionaries for the active multi-pass "
        "workflow. Coordinates image routing, output paths, Pass A witness run "
        "counts, seed and sampling schedules, LoRA trigger and anchor text, "
        "distiller defaults, VLM validator defaults, and final export options."
    ),
}


import json
import random
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

MAX_SEED_32 = 0xFFFFFFFF
PIPELINE_PLAN_TYPE = "captionforge_pipeline_plan"
PIPELINE_PLAN_VERSION = 5
MAX_PASS_A_RUNS_PER_MODEL = 5


@dataclass(frozen=True)
class CaptionForgeRun:
    ensemble_run_index: int
    seed: int | None
    temperature: float
    top_p: float
    top_k: int
    max_new_tokens: int
    max_size: int
    trigger_word: str
    output_dir: str
    input_path: str
    recursive: bool
    filename_glob: str


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


def _coerce_bool(value: Any, default: bool = False) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return bool(default)
    text = str(value).strip().lower()
    if text in {"1", "true", "yes", "y", "on"}:
        return True
    if text in {"0", "false", "no", "n", "off"}:
        return False
    return bool(default)


def _clean_name(value: Any, default: str = "captionforge_run") -> str:
    text = str(value or "").strip() or default
    text = re.sub(r"[^A-Za-z0-9_.-]+", "_", text).strip("._")
    return text or default


def _parse_schedule(value: Any, cast, default_value):
    if value is None:
        return [default_value]
    if isinstance(value, (list, tuple)):
        raw_items = list(value)
    else:
        text = str(value).strip()
        if not text:
            return [default_value]
        text = text.replace("\n", ",").replace(";", ",")
        raw_items = [item.strip() for item in text.split(",") if item.strip()]
    values = []
    for item in raw_items:
        try:
            values.append(cast(item))
        except Exception:
            continue
    return values or [default_value]


def _schedule_value(values: list[Any], index: int):
    if not values:
        return None
    return values[index] if index < len(values) else values[-1]


def _normalize_seed_mode(value: Any) -> str:
    value = str(value or "fixed").strip().lower()
    if value not in {"fixed", "increment", "decrement", "random"}:
        return "fixed"
    return value


def _seed_for_run(base_seed: int, seed_mode: str, index: int) -> int | None:
    if base_seed < 0:
        if seed_mode == "random":
            return random.SystemRandom().randint(0, MAX_SEED_32)
        return None
    base_seed = max(0, min(int(base_seed), MAX_SEED_32))
    if seed_mode == "fixed":
        return base_seed
    if seed_mode == "increment":
        return min(MAX_SEED_32, base_seed + index)
    if seed_mode == "decrement":
        return max(0, base_seed - index)
    if seed_mode == "random":
        rng = random.Random(base_seed)
        out = base_seed
        for _ in range(index + 1):
            out = rng.randint(0, MAX_SEED_32)
        return out
    return base_seed


def _normalize_runs_per_image(value: Any, default: Any = "Disabled") -> int:
    """Normalize a caption witness run-count widget/value.

    CaptionForge defines Disabled as exactly 0 runs. This applies to every
    Pass A caption witness family, including Joy, Qwen, Florence, and future
    model families.
    """
    text = str(value if value is not None else default).strip()
    if not text or text.lower() == "disabled":
        return 0
    return _coerce_int(text, 0, 0, MAX_PASS_A_RUNS_PER_MODEL)


# Compatibility helpers retained for older imports/callers.
def _normalize_required_runs(value: Any, default: int) -> int:
    return _normalize_runs_per_image(value, default)


def _normalize_optional_runs(value: Any) -> tuple[bool, int]:
    runs = _normalize_runs_per_image(value, "Disabled")
    return runs > 0, runs


def normalize_captionforge_pipeline_plan(config: Any) -> dict[str, Any]:
    if config is None:
        return {}
    if isinstance(config, dict):
        return dict(config)
    if isinstance(config, str):
        text = config.strip()
        if not text:
            return {}
        try:
            parsed = json.loads(text)
            return parsed if isinstance(parsed, dict) else {}
        except Exception:
            return {}
    return {}


# Compatibility name used by earlier nodes.
def normalize_captionforge_run_config(config: Any) -> dict[str, Any]:
    return normalize_captionforge_pipeline_plan(config)


def _shared_from_plan(cfg: dict[str, Any]) -> dict[str, Any]:
    shared = cfg.get("shared")
    return dict(shared) if isinstance(shared, dict) else cfg


def _legacy_captions_per_image(cfg: dict[str, Any], fallback: int) -> int:
    shared = _shared_from_plan(cfg)
    if "captions_per_image" in shared:
        return _coerce_int(shared.get("captions_per_image"), fallback, 1, 100)
    if "captions_per_image" in cfg:
        return _coerce_int(cfg.get("captions_per_image"), fallback, 1, 100)
    return _coerce_int(fallback, 1, 1, 100)


def get_pass_a_model_plan(pipeline_plan: Any, model_key: str, *, widget_captions_per_image: int = 1) -> dict[str, Any]:
    cfg = normalize_captionforge_pipeline_plan(pipeline_plan)
    pass_a = cfg.get("pass_a") if isinstance(cfg.get("pass_a"), dict) else {}
    raw = pass_a.get(model_key) if isinstance(pass_a.get(model_key), dict) else None
    if raw is None:
        legacy_runs = _legacy_captions_per_image(cfg, widget_captions_per_image)
        return {
            "model_key": model_key,
            "enabled": bool(cfg),
            "runs_per_image": legacy_runs,
            "source": "legacy_fallback",
        }
    enabled = _coerce_bool(raw.get("enabled", True), True)
    runs = _coerce_int(raw.get("runs_per_image", widget_captions_per_image), widget_captions_per_image, 0, 100)
    return {
        "model_key": str(raw.get("model_key", model_key) or model_key),
        "enabled": enabled and runs > 0,
        "runs_per_image": runs if enabled else 0,
        "role": str(raw.get("role", "") or ""),
        "source": "pass_a",
    }


def _derive_paths(output_dir: str, run_name: str, image_root: str = "") -> dict[str, str]:
    out = Path(str(output_dir or "").strip())
    name = _clean_name(run_name)
    caption_jsonl = out / f"{name}__A_RAW_CAPTIONS.jsonl"
    return {
        "output_dir": str(out),
        "run_name": name,
        "image_root": str(image_root or "").strip(),
        "caption_jsonl": str(caption_jsonl),
        "pass_a_jsonl": str(caption_jsonl),
        "distiller_jsonl": str(out / f"{name}__B_DISTILL.jsonl"),
        "distiller_readable_jsonl": str(out / f"{name}__B_DISTILL_readable.jsonl"),
        "distiller_readable_json": str(out / f"{name}__B_DISTILL_readable.json"),
        "distiller_prompt_jsonl": str(out / f"{name}__B_DISTILL_prompts.jsonl"),
        "validator_jsonl": str(out / f"{name}__C_VLM_VALIDATED.jsonl"),
        "validator_prompt_jsonl": str(out / f"{name}__C_VLM_VALIDATOR_prompts.jsonl"),
        "validator_readable_dir": str(out / f"{name}__C_VLM_VALIDATED_readable"),
        "final_jsonl": str(out / f"{name}__D_FINAL_EXPORT.jsonl"),
        "final_txt_dir": str(out / f"{name}__TXT"),
        "output_paths_json": str(out / f"{name}__output_paths.json"),
        "run_config_json": str(out / f"{name}__run_config.json"),
    }


def build_captionforge_pipeline_plan(
    *,
    output_dir: str = "",
    input_path: str = "",
    single_image_connected: bool = False,
    recursive: bool = True,
    filename_glob: str = "*",
    run_name: str = "captionforge_run",
    overwrite_outputs: bool = True,
    joy_runs_per_image: Any = 2,
    qwen_runs_per_image: Any = 2,
    florence_runs_per_image: Any = "Disabled",
    llama_vision_runs_per_image: Any = "Disabled",
    base_seed: int = -1,
    seed_mode: str = "fixed",
    temperature_schedule: str = "",
    top_p_schedule: str = "",
    top_k_schedule: str = "",
    max_size: int = 1024,
    max_new_tokens: int = 512,
    trigger_word: str = "",
    user_caption_anchor: str = "",
    distiller_model_family: str = "Llama",
    distiller_base_seed: int | None = None,
    distiller_seed_mode: str = "fixed",
    distiller_strategy: str = "single_pass",
    distiller_max_caption_chars_for_llm: int = 1536,
    distiller_num_predict: int = 3096,
    distiller_temperature: float = 0.24,
    distiller_top_p: float = 0.90,
    distiller_top_k: int = 60,
    distiller_write_prompt_jsonl: bool = False,
    distiller_preserve_raw_response: bool = False,
    validator_model_family: str = "Llama Vision",
    validator_base_seed: int | None = None,
    validator_seed_mode: str = "fixed",
    validator_num_predict: int = 2200,
    validator_temperature: float = 0.0,
    validator_top_p: float = 0.92,
    validator_top_k: int = 80,
    validator_write_prompt_jsonl: bool = False,
    validator_preserve_raw_vlm_response: bool = False,
    final_caption_style: str = "narrative",
    final_write_txt_sidecars: bool = True,
    final_write_jsonl: bool = True,
    # Legacy compatibility aliases retained for older callers.
    distiller_seed: int | None = None,
    validator_seed: int | None = None,
    distiller_model: str = "llama3.1:8b",
    validator_model: str = "llama3.2-vision:11b",
    captions_per_image: int | None = None,
    # Deprecated compatibility-only parameters. Intentionally ignored.
    pass_c_deterministic: bool = True,
    pass_c_polish_model: str = "disabled",
    semantic_profile: str = "",
) -> dict[str, Any]:
    joy_runs = _normalize_runs_per_image(joy_runs_per_image, 2)
    qwen_runs = _normalize_runs_per_image(qwen_runs_per_image, 2)
    florence_runs = _normalize_runs_per_image(florence_runs_per_image, "Disabled")
    llama_runs = _normalize_runs_per_image(llama_vision_runs_per_image, "Disabled")
    pass_a_total_runs = joy_runs + qwen_runs + florence_runs + llama_runs

    if pass_a_total_runs <= 0:
        raise ValueError(
            "CaptionForge Pipeline Planner: all caption witness families are disabled. "
            "Enable at least one captioner, or disable the Pipeline Planner for a "
            "standalone/non-planned workflow."
        )

    base_seed_n = _coerce_int(base_seed, -1, -1, MAX_SEED_32)
    seed_mode_n = _normalize_seed_mode(seed_mode)

    if distiller_base_seed is None:
        distiller_base_seed = distiller_seed
    if validator_base_seed is None:
        validator_base_seed = validator_seed

    d_seed = base_seed_n if distiller_base_seed is None else _coerce_int(distiller_base_seed, base_seed_n, -1, MAX_SEED_32)
    v_seed_default = -1 if base_seed_n < 0 else min(MAX_SEED_32, base_seed_n + 1000003)
    v_seed = v_seed_default if validator_base_seed is None else _coerce_int(validator_base_seed, v_seed_default, -1, MAX_SEED_32)

    run_name_n = _clean_name(run_name)
    input_path_n = str(input_path or "").strip()
    shared = {
        "base_seed": base_seed_n,
        "seed_mode": seed_mode_n,
        "temperature_schedule": str(temperature_schedule or "").strip(),
        "top_p_schedule": str(top_p_schedule or "").strip(),
        "top_k_schedule": str(top_k_schedule or "").strip(),
        "max_size": _coerce_int(max_size, 1024, 0, 4096),
        "max_new_tokens": _coerce_int(max_new_tokens, 512, 16, 4096),
        "trigger_word": str(trigger_word or "").strip(),
        "user_caption_anchor": str(user_caption_anchor or "").strip(),
        "output_dir": str(output_dir or "").strip(),
        "input_path": input_path_n,
        "image_root": input_path_n,
        "single_image_connected": _coerce_bool(single_image_connected, False),
        "recursive": _coerce_bool(recursive, True),
        "filename_glob": str(filename_glob or "*").strip() or "*",
        "run_name": run_name_n,
        "overwrite_outputs": _coerce_bool(overwrite_outputs, True),
    }
    shared["captions_per_image"] = (
        _coerce_int(captions_per_image, max(joy_runs, qwen_runs, florence_runs, llama_runs, 1), 1, 100)
        if captions_per_image is not None
        else max(joy_runs, qwen_runs, florence_runs, llama_runs, 1)
    )

    paths = _derive_paths(shared["output_dir"], run_name_n, image_root=input_path_n)

    distiller = {
        "backend": "ollama",
        "model_family": str(distiller_model_family or "Llama").strip() or "Llama",
        "model": str(distiller_model or "llama3.1:8b").strip() or "llama3.1:8b",
        "base_seed": d_seed,
        "seed": d_seed,
        "seed_mode": _normalize_seed_mode(distiller_seed_mode),
        "strategy": str(distiller_strategy or "single_pass").strip() or "single_pass",
        "max_caption_chars_for_llm": _coerce_int(distiller_max_caption_chars_for_llm, 1536, 0, 12000),
        "num_predict": _coerce_int(distiller_num_predict, 3096, 64, 12000),
        "temperature": _coerce_float(distiller_temperature, 0.24, 0.0, 2.0),
        "top_p": _coerce_float(distiller_top_p, 0.90, 0.0, 1.0),
        "top_k": _coerce_int(distiller_top_k, 60, 0, 500),
        "write_prompt_jsonl": _coerce_bool(distiller_write_prompt_jsonl, False),
        "preserve_raw_response": _coerce_bool(distiller_preserve_raw_response, False),
        "role": "text_llm_recall_heavy_distillation",
    }
    validator = {
        "backend": "ollama",
        "model_family": str(validator_model_family or "Llama Vision").strip() or "Llama Vision",
        "model": str(validator_model or "llama3.2-vision:11b").strip() or "llama3.2-vision:11b",
        "base_seed": v_seed,
        "seed": v_seed,
        "seed_mode": _normalize_seed_mode(validator_seed_mode),
        "image_root": input_path_n,
        "num_predict": _coerce_int(validator_num_predict, 2200, 64, 12000),
        "temperature": _coerce_float(validator_temperature, 0.0, 0.0, 2.0),
        "top_p": _coerce_float(validator_top_p, 0.92, 0.0, 1.0),
        "top_k": _coerce_int(validator_top_k, 80, 0, 500),
        "write_prompt_jsonl": _coerce_bool(validator_write_prompt_jsonl, False),
        "preserve_raw_vlm_response": _coerce_bool(validator_preserve_raw_vlm_response, False),
        "role": "image_aware_precision_validation",
    }
    final = {
        "caption_style": str(final_caption_style or "narrative").strip() or "narrative",
        "write_txt_sidecars": _coerce_bool(final_write_txt_sidecars, True),
        "write_jsonl": _coerce_bool(final_write_jsonl, True),
        "overwrite_outputs": _coerce_bool(overwrite_outputs, True),
        "role": "deterministic_validated_caption_export",
        "large_model_passes_after_validator": False,
    }

    return {
        "captionforge_config_type": PIPELINE_PLAN_TYPE,
        "captionforge_config_version": PIPELINE_PLAN_VERSION,
        "shared": shared,
        "paths": paths,
        "pass_a": {
            "joy": {"model_key": "joy", "enabled": joy_runs > 0, "runs_per_image": joy_runs, "role": "rich_descriptive_caption_witness"},
            "qwen": {"model_key": "qwen", "enabled": qwen_runs > 0, "runs_per_image": qwen_runs, "role": "detail_miner_alternate_caption_witness"},
            "florence": {"model_key": "florence", "enabled": florence_runs > 0, "runs_per_image": florence_runs, "role": "optional_grounding_caption_witness"},
            "llama_vision": {"model_key": "llama_vision", "enabled": llama_runs > 0, "runs_per_image": llama_runs, "role": "optional_vision_language_caption_witness"},
        },
        "pass_a_summary": {
            "enabled_models": [
                key for key, runs in {
                    "joy": joy_runs,
                    "qwen": qwen_runs,
                    "florence": florence_runs,
                    "llama_vision": llama_runs,
                }.items() if runs > 0
            ],
            "total_runs_per_image": pass_a_total_runs,
            "minimum_total_runs_per_image": 1,
            "minimum_satisfied": pass_a_total_runs >= 1,
        },
        "distiller": distiller,
        "validator": validator,
        "final": final,
        "pass_b_distiller": distiller,
        "pass_c_vlm_validator": validator,
        "final_export": final,
    }


# Compatibility builder retained for existing caption nodes.
def build_captionforge_run_config(
    captions_per_image: int = 1,
    base_seed: int = -1,
    seed_mode: str = "fixed",
    temperature_schedule: str = "",
    top_p_schedule: str = "",
    top_k_schedule: str = "",
    max_size: int = 1024,
    max_new_tokens: int = 512,
    trigger_word: str = "",
    output_dir: str = "",
    input_path: str = "",
    recursive: bool = True,
    filename_glob: str = "*",
) -> dict[str, Any]:
    return build_captionforge_pipeline_plan(
        output_dir=output_dir,
        input_path=input_path,
        recursive=recursive,
        filename_glob=filename_glob,
        joy_runs_per_image=captions_per_image,
        qwen_runs_per_image=captions_per_image,
        base_seed=base_seed,
        seed_mode=seed_mode,
        temperature_schedule=temperature_schedule,
        top_p_schedule=top_p_schedule,
        top_k_schedule=top_k_schedule,
        max_size=max_size,
        max_new_tokens=max_new_tokens,
        trigger_word=trigger_word,
        captions_per_image=captions_per_image,
    )


def expand_captionforge_runs(
    captionforge_run_config: Any = None,
    *,
    model_key: str | None = None,
    widget_captions_per_image: int = 1,
    widget_seed: int = -1,
    widget_temperature: float = 0.75,
    widget_top_p: float = 0.90,
    widget_top_k: int = 50,
    widget_max_new_tokens: int = 512,
    widget_max_size: int = 1024,
    widget_trigger_word: str = "",
    widget_output_dir: str = "",
    widget_input_path: str = "",
    widget_recursive: bool = True,
    widget_filename_glob: str = "*",
) -> list[CaptionForgeRun]:
    cfg = normalize_captionforge_pipeline_plan(captionforge_run_config)
    shared = _shared_from_plan(cfg)

    if model_key and cfg:
        model_plan = get_pass_a_model_plan(cfg, model_key, widget_captions_per_image=widget_captions_per_image)
        if not model_plan.get("enabled", False):
            return []
        captions_per_image = _coerce_int(model_plan.get("runs_per_image", widget_captions_per_image), widget_captions_per_image, 0, 100)
    else:
        captions_per_image = _legacy_captions_per_image(cfg, widget_captions_per_image)

    base_seed = _coerce_int(shared.get("base_seed", widget_seed), widget_seed, -1, MAX_SEED_32)
    seed_mode = _normalize_seed_mode(shared.get("seed_mode", "fixed"))
    max_size = _coerce_int(shared.get("max_size", widget_max_size), widget_max_size, 0, 4096)
    max_new_tokens = _coerce_int(shared.get("max_new_tokens", widget_max_new_tokens), widget_max_new_tokens, 16, 4096)
    trigger_word = str(shared.get("trigger_word", widget_trigger_word) or "").strip()
    output_dir = str(shared.get("output_dir", widget_output_dir) or "").strip()
    input_path = str(shared.get("input_path", widget_input_path) or "").strip()
    recursive = _coerce_bool(shared.get("recursive", widget_recursive), bool(widget_recursive))
    filename_glob = str(shared.get("filename_glob", widget_filename_glob) or "*").strip() or "*"

    temperatures = _parse_schedule(shared.get("temperature_schedule", ""), float, _coerce_float(widget_temperature, 0.75, 0.0, 2.0))
    top_ps = _parse_schedule(shared.get("top_p_schedule", ""), float, _coerce_float(widget_top_p, 0.90, 0.0, 1.0))
    top_ks = _parse_schedule(shared.get("top_k_schedule", ""), int, _coerce_int(widget_top_k, 50, 0, 500))

    return [
        CaptionForgeRun(
            ensemble_run_index=i,
            seed=_seed_for_run(base_seed, seed_mode, i),
            temperature=_coerce_float(_schedule_value(temperatures, i), widget_temperature, 0.0, 2.0),
            top_p=_coerce_float(_schedule_value(top_ps, i), widget_top_p, 0.0, 1.0),
            top_k=_coerce_int(_schedule_value(top_ks, i), widget_top_k, 0, 500),
            max_new_tokens=max_new_tokens,
            max_size=max_size,
            trigger_word=trigger_word,
            output_dir=output_dir,
            input_path=input_path,
            recursive=recursive,
            filename_glob=filename_glob,
        )
        for i in range(captions_per_image)
    ]


# Deprecated semantic-profile helper stubs retained only so old imports do not crash.
def discover_semantic_profiles() -> list[str]:
    return []


def default_semantic_profile() -> str:
    return "disabled"


def resolve_semantic_profile_path(profile_name: str) -> str:
    return ""
