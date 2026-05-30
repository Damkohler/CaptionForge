"""
CaptionForge Pipeline Planner Engine

- CaptionForge
  - This engine is part of **CaptionForge**, a model-agnostic captioning
    framework for ComfyUI developed by **J. L. Córdova**.

  - Repository
    https://github.com/Damkohler/CaptionForge

  - CaptionForge focuses on practical dataset-captioning infrastructure for:
        • LoRA dataset preparation
        • multi-engine caption generation
        • JSONL audit trails
        • claim extraction and refinement
        • consensus-oriented caption improvement

- Engine Purpose
    - The **CaptionForge Pipeline Planner Engine** provides the reusable backend
      planning logic for coordinated CaptionForge captioning workflows.

    - This file is the **reusable planner engine**, not the ComfyUI-facing node
      wrapper. It is responsible for:
            • normalizing CaptionForge pipeline-plan dictionaries
            • building typed CAPTIONFORGE_PIPELINE_PLAN objects
            • parsing comma-separated sampling schedules
            • expanding per-caption-instance sampling settings
            • deriving per-caption-instance seeds
            • resolving shared input/output routing
            • resolving shared image-size and token-budget guards
            • resolving shared LoRA trigger-word policy
            • resolving raw-caption run counts by model family
            • normalizing optional model-family participation
            • exposing model-family plan accessors for captioning nodes
            • returning immutable per-run planning records for caption engines
            • preserving temporary compatibility aliases during development

    - The ComfyUI-facing planner node lives in:
            captionforge_pipeline_planner_node.py

      That node handles:
            • ComfyUI INPUT_TYPES / widget definitions
            • user-facing planner labels
            • output-folder and input-path widgets
            • raw-caption witness-count widgets
            • optional validator and final-polish widgets
            • seed and sampling widgets
            • LoRA trigger-word widget
            • CAPTIONFORGE_PIPELINE_PLAN socket output
            • JSON string output for inspection
            • node display name, category, and mapping registration

- CaptionForge Pipeline Role
    - This engine creates and interprets the shared pipeline plan used by
      CaptionForge captioning nodes and downstream processing.

    - Raw caption witnesses generate auditable caption evidence records from one
      or more captioning engines.

    - The planner engine defines how many raw caption witnesses each model family
      contributes per image.

    - Current raw-caption family keys include:
            • joy
            • qwen
            • florence
            • llama_vision

    - Joy and Qwen are treated as foundational raw-caption witnesses in the
      standard planner design.

    - Optional model families may be disabled or assigned one or more raw-caption
      witness runs as CaptionForge expands.

- Planning Model
    - The engine emits a CAPTIONFORGE_PIPELINE_PLAN dictionary that may include:
            • shared routing fields
            • raw-caption model-family plan fields
            • optional validation fields
            • optional final-polish fields
            • seed policy fields
            • sampling schedule fields
            • image workload guard fields
            • generation token-budget fields
            • LoRA trigger-word fields

    - Captioning nodes consume the plan through this engine rather than parsing
      plan dictionaries independently.

    - Captioning nodes identify themselves by model-family key, such as:
            • joy
            • qwen
            • florence
            • llama_vision

    - The engine then expands the shared plan into per-caption-instance records
      that captioning nodes can execute directly.

- Deterministic Final Caption Policy
    - Deterministic CaptionForge synthesis is mandatory in the downstream final
      caption stage.

    - This engine does not expose or preserve a deterministic on/off switch.

    - Optional LLM polish, such as GPT-OSS polish, is treated as a post-process
      after deterministic synthesis rather than a replacement for deterministic
      claim-grounded caption construction.

- Compatibility Policy
    - During early CaptionForge development, this engine may expose temporary
      compatibility aliases so older captioning nodes can continue loading while
      node wrappers migrate to the new pipeline-plan schema.

    - These compatibility paths are intended to be removed once the pipeline
      planner, captioning nodes, claim extraction, and final caption synthesis
      stabilize around the canonical CAPTIONFORGE_PIPELINE_PLAN contract.

    - Compatibility should not become fallback behavior for production schema
      drift. CaptionForge favors canonical schemas and explicit migration over
      accumulating permanent alternate logic paths.

- Design Philosophy
    - This engine keeps CaptionForge pipeline planning separate from individual
      caption model implementation.

    - CaptionForge is engine-democratic: no single caption model is treated as
      canonical. The planner engine coordinates multiple caption witnesses so
      downstream claim extraction, consensus, validation, and final-caption
      synthesis can work from auditable evidence rather than one model's
      unverified output.

    - The engine prioritizes reproducibility, auditability, deterministic
      planning behavior, scalable hardware usage, and clean separation between
      ComfyUI node wrappers and reusable backend logic.

- ⚠️ Development Status
    - This is early CaptionForge pipeline-planning engine infrastructure.
    - The emitted plan schema, model-family list, validator options, polish
      options, compatibility aliases, and expansion rules may evolve as the
      multi-pass CaptionForge pipeline matures.
    - The engine is intended for local dataset preparation and controlled caption
      audit workflows.

- Attribution & License
  - Concept and implementation by **J. L. Córdova**
    with development assistance from **ChatGPT (OpenAI)**.

  - Designed for use with:
    https://github.com/comfyanonymous/ComfyUI

  - Copyright (c) 2026 J. L. Córdova

  - Released under the **MIT License**.
"""

from __future__ import annotations

MANIFEST = {
    "name": "CaptionForge Pipeline Planner Engine",
    "version": (0, 3, 0),
    "author": "J. L. Córdova",
    "description": (
        "Reusable backend planning engine for coordinated CaptionForge captioning "
        "workflows. Builds and normalizes CAPTIONFORGE_PIPELINE_PLAN dictionaries, "
        "parses sampling schedules, derives per-caption-instance seeds, resolves "
        "shared routing, image-size guards, token budgets, LoRA trigger-word policy, "
        "and raw-caption run counts by model family. Exposes model-family accessors "
        "and per-run expansion records so Joy, Qwen, Florence, Llama Vision, and "
        "future CaptionForge nodes can execute one auditable shared pipeline plan. "
        "Deterministic final-caption synthesis is treated as mandatory downstream; "
        "optional LLM polish is represented only as a post-synthesis polish choice."
    ),
}
import json
import random
from dataclasses import dataclass
from typing import Any
from pathlib import Path

MAX_SEED_32 = 0xFFFFFFFF
PIPELINE_PLAN_TYPE = "captionforge_pipeline_plan"
PIPELINE_PLAN_VERSION = 3
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


def _parse_schedule(value: Any, cast, default_value):
    """
    Parse comma/newline/semicolon separated schedule strings.

    Blank schedule means: use one-element fallback [default_value].
    Short schedules are later expanded by repeating the final value.
    """
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
    if index < len(values):
        return values[index]
    return values[-1]


def _normalize_seed_mode(value: Any) -> str:
    value = str(value or "fixed").strip().lower()
    if value not in {"fixed", "increment", "decrement", "random"}:
        return "fixed"
    return value


def _normalize_validator_model(value: Any) -> str:
    value = str(value or "disabled").strip().lower()
    allowed = {"disabled", "sam3"}
    return value if value in allowed else "disabled"


def _normalize_polish_model(value: Any) -> str:
    value = str(value or "disabled").strip().lower()
    allowed = {"disabled", "gpt_oss"}
    return value if value in allowed else "disabled"


def _normalize_required_runs(value: Any, default: int) -> int:
    """
    Required foundation caption families, currently Joy and Qwen.
    UI should only expose 1..5; this normalizer enforces that invariant too.
    """
    return _coerce_int(value, default, 1, MAX_PASS_A_RUNS_PER_MODEL)


def _normalize_optional_runs(value: Any) -> tuple[bool, int]:
    """
    Optional caption families use one dropdown:
      Disabled / 1 / 2 / 3 / 4 / 5
    """
    text = str(value if value is not None else "Disabled").strip()
    if not text or text.lower() == "disabled":
        return False, 0
    runs = _coerce_int(text, 0, 1, MAX_PASS_A_RUNS_PER_MODEL)
    return runs > 0, runs


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


def normalize_captionforge_pipeline_plan(config: Any) -> dict[str, Any]:
    """
    Accept either:
    - dict from a connected CAPTIONFORGE_PIPELINE_PLAN socket
    - JSON string
    - empty/None

    Returns a plain dict.
    """
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


# TEMP COMPAT: old helper name retained until all node variables/imports are renamed.
def normalize_captionforge_run_config(config: Any) -> dict[str, Any]:
    return normalize_captionforge_pipeline_plan(config)


def _shared_from_plan(cfg: dict[str, Any]) -> dict[str, Any]:
    shared = cfg.get("shared")
    if isinstance(shared, dict):
        return dict(shared)
    return cfg


def _legacy_captions_per_image(cfg: dict[str, Any], fallback: int) -> int:
    """
    TEMP COMPAT: old nodes call expand_captionforge_runs without model_key.
    This fallback lets them continue to run until Joy/Qwen nodes are updated.
    """
    shared = _shared_from_plan(cfg)
    if "captions_per_image" in shared:
        return _coerce_int(shared.get("captions_per_image"), fallback, 1, 100)
    if "captions_per_image" in cfg:
        return _coerce_int(cfg.get("captions_per_image"), fallback, 1, 100)
    return _coerce_int(fallback, 1, 1, 100)


def get_pass_a_model_plan(
    pipeline_plan: Any,
    model_key: str,
    *,
    widget_captions_per_image: int = 1,
) -> dict[str, Any]:
    """
    Return a normalized Pass A model-family plan.

    model_key examples:
      joy, qwen, florence, llama_vision

    Joy and Qwen are foundation caption families and normalize to enabled=True
    with 1..5 runs. Optional families normalize from Disabled/1..5.

    TEMP COMPAT: if a v3 model-specific plan is absent, fall back to the legacy
    captions_per_image field so older nodes keep working during migration.
    """
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


def _captionforge_root_from_engine_file() -> Path:
    # engines/captionforge_pipeline_planner_engine.py -> CaptionForge/
    return Path(__file__).resolve().parents[1]


def get_semantic_profiles_dir() -> Path:
    return _captionforge_root_from_engine_file() / "semantic_profiles"


def discover_semantic_profiles() -> list[str]:
    """
    Discover user-selectable semantic profiles under CaptionForge/semantic_profiles/.

    Returns relative POSIX-style paths such as:
        general_v1.semantic_profile.json
        experimental/female_character_conservative_v1.semantic_profile.json
    """
    root = get_semantic_profiles_dir()
    if not root.exists():
        return []

    candidates: list[Path] = []
    candidates.extend(root.rglob("*.semantic_profile.json"))
    candidates.extend(root.rglob("*.json"))

    # Deduplicate while preserving sorted stability.
    unique = sorted(set(candidates), key=lambda p: str(p.relative_to(root)).lower())

    return [
        p.relative_to(root).as_posix()
        for p in unique
        if p.is_file()
    ]


def default_semantic_profile() -> str:
    profiles = discover_semantic_profiles()
    if not profiles:
        return "disabled"

    preferred = [
        "general_v1.semantic_profile.json",
        "image_v1_minimum.semantic_profile.json",
    ]

    lower_map = {p.lower(): p for p in profiles}
    for item in preferred:
        found = lower_map.get(item.lower())
        if found:
            return found

    return profiles[0]


def resolve_semantic_profile_path(profile_name: str) -> str:
    """
    Resolve a semantic profile relative to CaptionForge/semantic_profiles/.

    Arbitrary filesystem paths are intentionally not supported here. This keeps
    CaptionForge runs portable, auditable, and repo/package-relative.
    """
    profile_name = str(profile_name or "").strip()
    if not profile_name or profile_name.lower() == "disabled":
        return ""

    root = get_semantic_profiles_dir()
    candidate = (root / profile_name).resolve()

    try:
        candidate.relative_to(root.resolve())
    except ValueError:
        raise RuntimeError(
            "Semantic profile must live under CaptionForge/semantic_profiles/. "
            f"Rejected path: {profile_name}"
        )

    if not candidate.exists() or not candidate.is_file():
        raise RuntimeError(f"Semantic profile not found: {candidate}")

    return str(candidate)


def build_captionforge_pipeline_plan(
    *,
    output_dir: str = "",
    input_path: str = "",
    recursive: bool = True,
    filename_glob: str = "*",
    joy_runs_per_image: Any = 2,
    qwen_runs_per_image: Any = 1,
    florence_runs_per_image: Any = "Disabled",
    llama_vision_runs_per_image: Any = "Disabled",
    validator_model: str = "disabled",
    pass_c_deterministic: bool = True,
    pass_c_polish_model: str = "disabled",
    semantic_profile: str = "",
    base_seed: int = -1,
    seed_mode: str = "fixed",
    temperature_schedule: str = "",
    top_p_schedule: str = "",
    top_k_schedule: str = "",
    max_size: int = 1024,
    max_new_tokens: int = 512,
    trigger_word: str = "",
    captions_per_image: int | None = None,
) -> dict[str, Any]:
    """
    Build a v3 CaptionForge Pipeline/Run Planner config.

    New source of truth:
      pass_a.<model_family>.runs_per_image

    TEMP COMPAT:
      shared.captions_per_image is emitted only as a fallback for older caption
      nodes that have not yet been updated to pass model_key into the expander.
    """
    joy_runs = _normalize_required_runs(joy_runs_per_image, 2)
    qwen_runs = _normalize_required_runs(qwen_runs_per_image, 1)
    florence_enabled, florence_runs = _normalize_optional_runs(florence_runs_per_image)
    llama_enabled, llama_runs = _normalize_optional_runs(llama_vision_runs_per_image)
    semantic_profile = str(semantic_profile or "").strip() or default_semantic_profile()
    semantic_profile_path = resolve_semantic_profile_path(semantic_profile)
    
    shared = {
        "base_seed": _coerce_int(base_seed, -1, -1, MAX_SEED_32),
        "seed_mode": _normalize_seed_mode(seed_mode),
        "temperature_schedule": str(temperature_schedule or "").strip(),
        "top_p_schedule": str(top_p_schedule or "").strip(),
        "top_k_schedule": str(top_k_schedule or "").strip(),
        "max_size": _coerce_int(max_size, 1024, 0, 4096),
        "max_new_tokens": _coerce_int(max_new_tokens, 512, 16, 4096),
        "trigger_word": str(trigger_word or "").strip(),
        "output_dir": str(output_dir or "").strip(),
        "input_path": str(input_path or "").strip(),
        "recursive": _coerce_bool(recursive, True),
        "filename_glob": str(filename_glob or "*").strip() or "*",
    }

    # TEMP COMPAT: no single legacy number can represent per-model counts.
    # Use the maximum so old unmigrated caption nodes still produce enough
    # evidence rather than silently under-producing. Remove after nodes pass
    # model_key into expand_captionforge_runs().
    if captions_per_image is not None:
        shared["captions_per_image"] = _coerce_int(captions_per_image, max(joy_runs, qwen_runs, 1), 1, 100)
    else:
        shared["captions_per_image"] = max(joy_runs, qwen_runs, florence_runs, llama_runs, 1)

    pass_a_total_runs = joy_runs + qwen_runs + florence_runs + llama_runs

    return {
        "captionforge_config_type": PIPELINE_PLAN_TYPE,
        "captionforge_config_version": PIPELINE_PLAN_VERSION,
        "shared": shared,
        "pass_a": {
            "joy": {
                "model_key": "joy",
                "enabled": True,
                "runs_per_image": joy_runs,
                "role": "rich_descriptive_caption_witness",
            },
            "qwen": {
                "model_key": "qwen",
                "enabled": True,
                "runs_per_image": qwen_runs,
                "role": "detail_miner_alternate_caption_witness",
            },
            "florence": {
                "model_key": "florence",
                "enabled": florence_enabled,
                "runs_per_image": florence_runs,
                "role": "grounding_task_caption_witness",
            },
            "llama_vision": {
                "model_key": "llama_vision",
                "enabled": llama_enabled,
                "runs_per_image": llama_runs,
                "role": "alternate_vision_language_caption_witness",
            },
        },
        "pass_a_summary": {
            "foundation_models": ["joy", "qwen"],
            "total_runs_per_image": pass_a_total_runs,
            "minimum_total_runs_per_image": 2,
            "minimum_satisfied_by_design": True,
        },
        "pass_ab": {
            "validator": {
                "model": _normalize_validator_model(validator_model),
                "role": "region_object_visibility_claim_validator",
            }
        },
        "pass_b": {
            "claim_extractor": "llama_ollama_fixed",
            "role": "schema_guided_claim_extraction",
        },
        "pass_c": {
            "semantic_profile": semantic_profile,
            "semantic_profile_path": semantic_profile_path,
            "polish_model": _normalize_polish_model(pass_c_polish_model),
        },
    }


# TEMP COMPAT: old builder name retained until older imports are gone.
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
        florence_runs_per_image="Disabled",
        llama_vision_runs_per_image="Disabled",
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
    """
    Expand connected Pipeline Planner + node widget fallbacks into per-caption settings.

    If model_key is provided, the v3 per-model Pass A run count is used.
    If model_key is omitted, legacy captions_per_image fallback is used.
    """
    cfg = normalize_captionforge_pipeline_plan(captionforge_run_config)
    shared = _shared_from_plan(cfg)

    if model_key and cfg:
        model_plan = get_pass_a_model_plan(
            cfg,
            model_key,
            widget_captions_per_image=widget_captions_per_image,
        )
        if not model_plan.get("enabled", False):
            return []
        captions_per_image = _coerce_int(
            model_plan.get("runs_per_image", widget_captions_per_image),
            widget_captions_per_image,
            0,
            100,
        )
    else:
        captions_per_image = _legacy_captions_per_image(cfg, widget_captions_per_image)

    base_seed = _coerce_int(
        shared.get("base_seed", widget_seed),
        widget_seed,
        -1,
        MAX_SEED_32,
    )
    seed_mode = _normalize_seed_mode(shared.get("seed_mode", "fixed"))

    max_size = _coerce_int(
        shared.get("max_size", widget_max_size),
        widget_max_size,
        0,
        4096,
    )
    max_new_tokens = _coerce_int(
        shared.get("max_new_tokens", widget_max_new_tokens),
        widget_max_new_tokens,
        16,
        4096,
    )

    trigger_word = str(shared.get("trigger_word", widget_trigger_word) or "").strip()
    output_dir = str(shared.get("output_dir", widget_output_dir) or "").strip()
    input_path = str(shared.get("input_path", widget_input_path) or "").strip()
    recursive = _coerce_bool(shared.get("recursive", widget_recursive), bool(widget_recursive))
    filename_glob = str(shared.get("filename_glob", widget_filename_glob) or "*").strip() or "*"

    temperatures = _parse_schedule(
        shared.get("temperature_schedule", ""),
        float,
        _coerce_float(widget_temperature, 0.75, 0.0, 2.0),
    )
    top_ps = _parse_schedule(
        shared.get("top_p_schedule", ""),
        float,
        _coerce_float(widget_top_p, 0.90, 0.0, 1.0),
    )
    top_ks = _parse_schedule(
        shared.get("top_k_schedule", ""),
        int,
        _coerce_int(widget_top_k, 50, 0, 500),
    )

    runs: list[CaptionForgeRun] = []
    for i in range(captions_per_image):
        runs.append(
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
        )

    return runs
