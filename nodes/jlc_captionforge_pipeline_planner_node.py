"""
JLC CaptionForge Pipeline Planner — ComfyUI Node Wrapper

- CaptionForge
  - This node is part of **CaptionForge**, a model-agnostic captioning and
    caption-refinement framework for ComfyUI developed by **J. L. Córdova**.

  - Repository
    https://github.com/Damkohler/CaptionForge

- CaptionForge focuses on practical dataset-captioning infrastructure for
  LoRA dataset preparation, using multi-engine caption generation, JSONL
  audit trails, claim extraction and refinement, text-LLM distillation,
  image-aware VLM validation, and consensus-oriented caption improvement
  to produce grounded, auditable training captions.

- Node Purpose
    - The **JLC CaptionForge Pipeline Planner** is the ordinary-run control
      center for the current CaptionForge workflow.

    - This file is the **ComfyUI-facing wrapper** for building a
      CAPTIONFORGE_PIPELINE_PLAN. It is responsible for:
            • ComfyUI INPUT_TYPES / widget definitions
            • optional IMAGE passthrough for quick single-image workflows
            • shared input path, recursion, and filename-glob routing
            • output folder and run-name policy
            • LoRA trigger word and user caption anchor routing
            • raw-caption run counts for Joy, Qwen, and generic Ollama Caption nodes
            • caption seed, sampling, image-size, and token policy
            • Distiller model/settings selection
            • Validator model/settings selection
            • final export policy
            • JSON serialization of the plan for audit/debugging

    - The reusable planner implementation lives in:
            captionforge_pipeline_planner_engine.py

- Ollama Model Dropdowns
    - Distiller and Validator dropdown values are explicit Ollama model tags.
    - Caption-stage Ollama models are intentionally selected on each
      **JLC CaptionForge Ollama Caption** node, not in this Planner.
    - Dropdown choices for Distiller/Validator are loaded at node-import time
      from:
            config/captionforge_ollama_models.json
    - If the JSON file is missing or malformed, the node falls back to:
            Distiller: llama3.1:8b
            Validator: gemma4:e4b
    - The optional custom choice lets users enter any installed Ollama model tag
      without editing Python.

- CaptionForge Pipeline Role
    - The planner emits the CAPTIONFORGE_PIPELINE_PLAN consumed by caption
      nodes and the JLC CaptionForge capstone node.

    - The canonical graph flow is:
            Pipeline Planner
              -> Joy/Qwen/Ollama raw-caption nodes
              -> JLC CaptionForge capstone node
              -> Distiller Engine
              -> VLM Validator Engine
              -> final deterministic TXT/JSONL export

    - SmolVLM is not exposed in the current mainline Planner UI. It may remain
      available as a standalone/experimental node and can be revisited later.

- Design Philosophy
    - CaptionForge is an original concept and implementation, not derived from
      or based on another ComfyUI workflow.
    - The planner keeps shared workflow policy centralized while leaving model
      execution to reusable engines and model-specific caption nodes.
    - Caption-stage Ollama model choice remains local to each Ollama Caption
      node so users can chain one or more Ollama-backed VLM captioners without
      making the Planner responsible for pulling, probing, or validating model tags.
    - The node prioritizes reproducibility, auditability, explicit model tags,
      low UI ambiguity, and clean separation between ComfyUI UI and reusable
      pipeline logic.

- ⚠️ Development Status
    - This is release-candidate CaptionForge infrastructure.
    - Widget names, output schema details, and downstream validation strategy may
      evolve as CaptionForge matures.

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
    "name": "JLC CaptionForge Pipeline Planner",
    "version": CAPTIONFORGE_VERSION,
    "author": "J. L. Córdova",
    "description": (
        "ComfyUI-facing Pipeline Planner node for CaptionForge. Builds the "
        "CAPTIONFORGE_PIPELINE_PLAN consumed through the pipeline_plan pin by "
        "caption nodes and the JLC CaptionForge capstone node. Exposes Joy, Qwen, "
        "and generic Ollama Caption run counts for the current supported Pass A set. "
        "Caption-stage Ollama model tags are selected directly on each Ollama Caption "
        "node. Loads explicit Ollama Distiller/Validator dropdown tags from "
        "config/captionforge_ollama_models.json, with no family aliases or shorthand "
        "model substitutions. The selected output folder is treated as an output root; "
        "the planner derives a run-specific working directory for JSON/JSONL artifacts. "
        "When overwrite_outputs is true, the planner resets the planned Pass A caption "
        "JSONL before caption nodes append fresh records."
    ),
}

import inspect
import json
import re
from pathlib import Path
from typing import Any

try:
    from .captionforge_ollama_model_dropdowns import load_ollama_model_dropdowns
except Exception:  # pragma: no cover - useful for direct local smoke tests
    from captionforge_ollama_model_dropdowns import load_ollama_model_dropdowns

try:
    import folder_paths
except Exception:
    folder_paths = None

from ..engines.captionforge_pipeline_planner_engine import (
    MAX_SEED_32,
    build_captionforge_pipeline_plan,
)

CAPTION_RUNS = ["Disabled", "1", "2", "3", "4", "5"]
SEED_MODES = ["fixed", "increment", "decrement", "random"]

# Current production-caption defaults favor the tested Ollama VLM path. These
# values are also consumed by Joy/Qwen in planned mode through the shared
# CaptionForge run expansion.
DEFAULT_CAPTION_TEMPERATURE_SCHEDULE = "0.90"
DEFAULT_CAPTION_TOP_P_SCHEDULE = "0.60"
DEFAULT_CAPTION_TOP_K_SCHEDULE = "80"
DEFAULT_CAPTION_MAX_IMAGE_SIZE = 1024
DEFAULT_CAPTION_MAX_NEW_TOKENS = 6000

_MODEL_DROPDOWNS = load_ollama_model_dropdowns(__file__)
DISTILLER_MODEL_CHOICES = _MODEL_DROPDOWNS["distiller_models"]
VALIDATOR_MODEL_CHOICES = _MODEL_DROPDOWNS["validator_models"]
DEFAULT_DISTILLER_MODEL = _MODEL_DROPDOWNS["distiller_default"]
DEFAULT_VALIDATOR_MODEL = _MODEL_DROPDOWNS["validator_default"]
DISTILLER_STRATEGIES = ["single_pass", "by_model_then_global"]
FINAL_CAPTION_STYLES = ["narrative", "comma", "both"]


def _default_output_dir() -> str:
    if folder_paths is not None:
        try:
            return str(Path(folder_paths.get_output_directory()) / "CaptionForge")
        except Exception:
            pass
    return str(Path.cwd() / "output" / "CaptionForge")


def _clean_run_name(value: Any) -> str:
    text = str(value or "captionforge_run").strip()
    text = re.sub(r"[^A-Za-z0-9_.-]+", "_", text).strip("._")
    return text or "captionforge_run"


def _as_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


def _runs_per_image(value: Any, default: str = "Disabled") -> int:
    """Normalize caption witness count widgets. Disabled means exactly 0 runs."""
    text = str(value if value is not None else default).strip()
    if not text or text.lower() == "disabled":
        return 0
    try:
        return max(0, int(text))
    except Exception:
        return 0


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


def _call_build_captionforge_pipeline_plan_compat(**kwargs) -> dict[str, Any]:
    """Call the planner engine across minor signature revisions.

    The node is allowed to expose more UI detail than older planner-engine
    versions know about. We filter unsupported keyword arguments before calling
    the reusable engine, then patch the returned plan with the concrete model
    names and detailed capstone settings consumed by the final node.
    """
    try:
        sig = inspect.signature(build_captionforge_pipeline_plan)
        accepts_kwargs = any(p.kind == inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values())
        call_kwargs = kwargs if accepts_kwargs else {k: v for k, v in kwargs.items() if k in sig.parameters}
    except Exception:
        call_kwargs = kwargs

    plan = build_captionforge_pipeline_plan(**call_kwargs)
    if not isinstance(plan, dict):
        plan = {}

    shared = plan.setdefault("shared", {})
    if isinstance(shared, dict):
        shared.setdefault("single_image_connected", bool(kwargs.get("single_image_connected", False)))
        shared.setdefault("overwrite_outputs", bool(kwargs.get("overwrite_outputs", True)))

    # Concrete model names are stored in several compatible locations because
    # older and newer capstone nodes look in slightly different namespaces.
    distiller_model = str(kwargs.get("distiller_model") or DEFAULT_DISTILLER_MODEL).strip() or DEFAULT_DISTILLER_MODEL
    validator_model = str(kwargs.get("validator_model") or DEFAULT_VALIDATOR_MODEL).strip() or DEFAULT_VALIDATOR_MODEL

    caption_common = {
        "base_seed": kwargs.get("base_seed", -1),
        "seed_mode": kwargs.get("seed_mode", "fixed"),
        "temperature_schedule": kwargs.get("temperature_schedule", DEFAULT_CAPTION_TEMPERATURE_SCHEDULE),
        "top_p_schedule": kwargs.get("top_p_schedule", DEFAULT_CAPTION_TOP_P_SCHEDULE),
        "top_k_schedule": kwargs.get("top_k_schedule", DEFAULT_CAPTION_TOP_K_SCHEDULE),
        "max_size": kwargs.get("max_size", DEFAULT_CAPTION_MAX_IMAGE_SIZE),
        "max_new_tokens": kwargs.get("max_new_tokens", DEFAULT_CAPTION_MAX_NEW_TOKENS),
    }
    for caption_key in ("caption_settings", "caption_generation", "pass_a_settings"):
        existing = plan.get(caption_key)
        if not isinstance(existing, dict):
            existing = {}
        existing.update(caption_common)
        plan[caption_key] = existing

    pass_b_distiller = plan.setdefault("pass_b_distiller", {})
    if isinstance(pass_b_distiller, dict):
        pass_b_distiller.update({
            "backend": "ollama",
            "model": distiller_model,
            "ollama_model": distiller_model,
            "seed": kwargs.get("distiller_base_seed", -1),
        })

    pass_c_vlm_validator = plan.setdefault("pass_c_vlm_validator", {})
    if isinstance(pass_c_vlm_validator, dict):
        pass_c_vlm_validator.update({
            "backend": "ollama",
            "model": validator_model,
            "ollama_model": validator_model,
            "seed": kwargs.get("validator_base_seed", -1),
        })

    distiller_common = {
        "model": distiller_model,
        "ollama_model": distiller_model,
        "model_family": distiller_model,
        "base_seed": kwargs.get("distiller_base_seed", -1),
        "seed_mode": kwargs.get("distiller_seed_mode", "fixed"),
        "strategy": kwargs.get("distiller_strategy", "single_pass"),
        "max_caption_chars_for_llm": kwargs.get("distiller_max_caption_chars_for_llm", 1536),
        "num_predict": kwargs.get("distiller_num_predict", 3096),
        "temperature": kwargs.get("distiller_temperature", 0.24),
        "top_p": kwargs.get("distiller_top_p", 0.90),
        "top_k": kwargs.get("distiller_top_k", 60),
        "write_prompt_jsonl": kwargs.get("distiller_write_prompt_jsonl", False),
        "preserve_raw_response": kwargs.get("distiller_preserve_raw_response", False),
    }
    plan["distiller"] = dict(distiller_common)
    plan["pass_b"] = dict(distiller_common)

    validator_common = {
        "model": validator_model,
        "ollama_model": validator_model,
        "model_family": validator_model,
        "base_seed": kwargs.get("validator_base_seed", -1),
        "seed_mode": kwargs.get("validator_seed_mode", "fixed"),
        "num_predict": kwargs.get("validator_num_predict", 2200),
        "temperature": kwargs.get("validator_temperature", 0.0),
        "top_p": kwargs.get("validator_top_p", 0.92),
        "top_k": kwargs.get("validator_top_k", 80),
        "write_prompt_jsonl": kwargs.get("validator_write_prompt_jsonl", False),
        "preserve_raw_vlm_response": kwargs.get("validator_preserve_raw_vlm_response", False),
    }
    plan["validator"] = dict(validator_common)
    plan["pass_c"] = dict(validator_common)

    plan["final"] = {
        "caption_style": kwargs.get("final_caption_style", "narrative"),
        "write_txt_sidecars": kwargs.get("final_write_txt_sidecars", True),
        "write_jsonl": kwargs.get("final_write_jsonl", True),
        "overwrite_outputs": kwargs.get("overwrite_outputs", True),
    }
    plan["output"] = {"overwrite_outputs": kwargs.get("overwrite_outputs", True)}
    return plan



def _patch_supported_caption_witnesses(
    plan: dict[str, Any],
    *,
    joy_runs: int,
    qwen_runs: int,
    ollama_runs: int,
) -> dict[str, Any]:
    """Normalize the current supported Pass A caption-node set.

    Current CaptionForge mainline exposes Joy, Qwen, and generic Ollama Caption
    nodes. SmolVLM is deliberately not exposed in this Planner revision, but
    deprecated compatibility keys are patched to zero so older planner-engine
    structures do not accidentally enable removed branches.

    The canonical Ollama key is ``ollama``. Each connected JLC CaptionForge
    Ollama Caption node uses this count with its own locally selected Ollama
    model tag.
    """
    if not isinstance(plan, dict):
        plan = {}

    counts = {
        "joy": max(0, int(joy_runs)),
        "qwen": max(0, int(qwen_runs)),
        "ollama": max(0, int(ollama_runs)),
    }
    deprecated_zero = {
        "smolvlm": 0,
        "smol": 0,
        "florence": 0,
        "llama_vision": 0,
        "llamavision": 0,
        "llama": 0,
    }

    # Top-level compatibility counters.
    for key, runs in counts.items():
        plan[f"{key}_runs_per_image"] = runs
        plan[f"{key}_captions_per_image"] = runs

    # Extra aliases accepted by the Ollama Caption node and earlier experiments.
    for alias in ("ollama_caption", "caption_ollama", "ollama_vlm"):
        plan[f"{alias}_runs_per_image"] = counts["ollama"]
        plan[f"{alias}_captions_per_image"] = counts["ollama"]

    for key, runs in deprecated_zero.items():
        plan[f"{key}_runs_per_image"] = runs
        plan[f"{key}_captions_per_image"] = runs

    shared = plan.setdefault("shared", {})
    if isinstance(shared, dict):
        shared["caption_witness_counts"] = dict(counts)
        shared["caption_node_counts"] = dict(counts)
        shared["supported_caption_witnesses"] = [key for key, runs in counts.items() if runs > 0]
        shared["supported_caption_nodes"] = [key for key, runs in counts.items() if runs > 0]
        shared["deprecated_caption_witnesses"] = list(deprecated_zero.keys())
        shared["deprecated_caption_nodes"] = list(deprecated_zero.keys())
        shared["ollama_caption_model_policy"] = "model_tag_selected_in_each_ollama_caption_node"

    def witness_record(model_key: str, runs: int) -> dict[str, Any]:
        return {
            "enabled": bool(runs),
            "model_key": model_key,
            "model_family": model_key,
            "runs_per_image": int(runs),
            "captions_per_image": int(runs),
        }

    def patch_family(container: dict[str, Any], key: str, model_key: str, runs: int) -> None:
        existing = container.get(key)
        if not isinstance(existing, dict):
            existing = {}
        merged = dict(existing)
        merged.update(witness_record(model_key, runs))
        container[key] = merged

    def patch_container(container: Any) -> None:
        if not isinstance(container, dict):
            return

        # Preserve any engine-provided per-family details, but force current
        # counts/enabled states.
        for key, runs in counts.items():
            patch_family(container, key, key, runs)

        # Aliases let existing/future Ollama nodes resolve the same canonical
        # count without making the Planner care about model tags.
        for alias in ("ollama_caption", "caption_ollama", "ollama_vlm"):
            patch_family(container, alias, "ollama", counts["ollama"])

        for key in deprecated_zero:
            patch_family(container, key, key, 0)

    # Common/possible namespaces used during CaptionForge planner evolution.
    for container_key in (
        "caption",
        "captions",
        "captioning",
        "caption_models",
        "caption_witnesses",
        "caption_nodes",
        "caption_model_families",
        "raw_caption_witnesses",
        "pass_a",
        "pass_a_raw_captions",
        "pass_a_captioners",
    ):
        patch_container(plan.setdefault(container_key, {}))

    return plan

def _reset_pass_a_jsonl_for_overwrite(plan: dict[str, Any], *, overwrite_outputs: bool) -> None:
    """Reset the planned Pass A JSONL before caption witnesses append.

    Caption witness nodes append to the shared A_RAW_CAPTIONS JSONL during a
    planned run so Joy, Qwen, Ollama, and future caption nodes can contribute to the
    same evidence file. That append behavior is correct inside a single run, but
    stale records from a previous run must not survive when the user has selected
    Output - overwrite outputs.

    The Pipeline Planner is the conservative place to do this because it runs
    upstream of every caption witness. Resetting here avoids making any individual
    witness responsible for knowing whether it is "first" in the graph.
    """
    if not overwrite_outputs or not isinstance(plan, dict):
        return

    paths = plan.get("paths")
    if not isinstance(paths, dict):
        return

    planned_paths: list[Path] = []
    seen: set[str] = set()
    for key in ("caption_jsonl", "pass_a_jsonl"):
        value = str(paths.get(key) or "").strip()
        if not value:
            continue
        candidate = Path(value)
        marker = str(candidate)
        if marker not in seen:
            planned_paths.append(candidate)
            seen.add(marker)

    if not planned_paths:
        return

    removed: list[str] = []
    prepared: list[str] = []

    for path in planned_paths:
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            if path.exists():
                if path.is_dir():
                    raise IsADirectoryError(
                        f"CaptionForge Planner cannot overwrite Pass A JSONL because path is a directory: {path}"
                    )
                path.unlink()
                removed.append(str(path))
            else:
                prepared.append(str(path))
        except Exception as exc:
            raise RuntimeError(
                "CaptionForge Pipeline Planner could not reset the planned Pass A "
                f"caption JSONL for overwrite mode: {path} ({exc})"
            ) from exc

    runtime = plan.setdefault("runtime", {})
    if isinstance(runtime, dict):
        runtime["pass_a_jsonl_reset_by_planner"] = True
        runtime["pass_a_jsonl_reset_paths"] = removed or prepared

    if removed:
        print(
            "[JLC CaptionForge Pipeline Planner] overwrite_outputs=True; "
            f"reset Pass A JSONL: {', '.join(removed)}",
            flush=True,
        )
    elif prepared:
        print(
            "[JLC CaptionForge Pipeline Planner] overwrite_outputs=True; "
            f"Pass A JSONL will be created fresh: {', '.join(prepared)}",
            flush=True,
        )


class JLC_CaptionForge_Pipeline_Planner:
    """Build a CAPTIONFORGE_PIPELINE_PLAN and optionally pass IMAGE through."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "Planner - enabled": (
                    "BOOLEAN",
                    {
                        "default": True,
                        "tooltip": (
                            "Enable CaptionForge Pipeline Planner mode. If disabled, this node "
                            "passes through the optional IMAGE and emits an empty/falsy plan so "
                            "downstream nodes can run in standalone mode without GUI bypassing."
                        ),
                    },
                ),

                # -----------------------------------------------------------------
                # Input routing first.
                # -----------------------------------------------------------------
                "Input - image path": (
                    "STRING",
                    {
                        "default": "",
                        "multiline": False,
                        "tooltip": (
                            "Image file or image folder/root for ordinary folder/file workflows. "
                            "This is also used as the validator image root. For quick single-image "
                            "workflows, connect IMAGE to the optional Input - single image socket."
                        ),
                    },
                ),
                "Input - recursive": (
                    "BOOLEAN",
                    {
                        "default": True,
                        "tooltip": "Whether captioning nodes should recurse when Input - image path is a folder.",
                    },
                ),
                "Input - filename glob": (
                    "STRING",
                    {
                        "default": "*",
                        "multiline": False,
                        "tooltip": "Filename glob for folder captioning, e.g. *.png, *.jpg, or *.",
                    },
                ),

                # -----------------------------------------------------------------
                # Output routing immediately after inputs.
                # -----------------------------------------------------------------
                "Output - folder": (
                    "STRING",
                    {
                        "default": _default_output_dir(),
                        "multiline": False,
                        "tooltip": (
                            "Output root folder. CaptionForge creates a run-specific working directory "
                            "inside this folder for JSON/JSONL/audit artifacts. Final TXT sidecars are "
                            "written beside their resolved source images."
                        ),
                    },
                ),
                "Output - run name": (
                    "STRING",
                    {
                        "default": "captionforge_run",
                        "multiline": False,
                        "tooltip": (
                            "Run-root used to name config JSON, output path JSON, caption JSONL, "
                            "distiller JSONL, prompt JSONLs, validator JSONL, and final JSONL."
                        ),
                    },
                ),
                "Output - overwrite outputs": (
                    "BOOLEAN",
                    {
                        "default": True,
                        "tooltip": "Overwrite generated run artifacts during capstone execution.",
                    },
                ),

                # -----------------------------------------------------------------
                # LoRA metadata immediately after outputs.
                # -----------------------------------------------------------------
                "LoRA - trigger word": (
                    "STRING",
                    {
                        "default": "",
                        "multiline": False,
                        "tooltip": "Optional shared LoRA trigger token/string preserved through the pipeline.",
                    },
                ),
                "LoRA - user caption anchor": (
                    "STRING",
                    {
                        "default": "",
                        "multiline": False,
                        "tooltip": "Optional user style/identity anchor passed to distiller and validator.",
                    },
                ),

                # -----------------------------------------------------------------
                # Caption-node controls. User-facing label avoids Pass A.
                # -----------------------------------------------------------------
                "Caption - Joy runs/image": (
                    CAPTION_RUNS,
                    {
                        "default": "2",
                        "tooltip": "Joy Caption runs per image. Set to Disabled to omit Joy from this run. Dropdown is capped at 5 to prevent accidental giant runs.",
                    },
                ),
                "Caption - Qwen runs/image": (
                    CAPTION_RUNS,
                    {
                        "default": "2",
                        "tooltip": "Qwen Caption runs per image. Set to Disabled to omit Qwen from this run. Dropdown is capped at 5 to prevent accidental giant runs.",
                    },
                ),
                "Caption - Ollama runs/image": (
                    CAPTION_RUNS,
                    {
                        "default": "Disabled",
                        "tooltip": (
                            "Ollama Caption runs per image for each connected JLC CaptionForge Ollama Caption node. "
                            "The actual Ollama model tag is selected in each Ollama Caption node. Connecting multiple "
                            "Ollama Caption nodes multiplies runtime, memory pressure, and raw-caption count."
                        ),
                    },
                ),
                "Caption - base seed": (
                    "INT",
                    {
                        "default": -1,
                        "min": -1,
                        "max": MAX_SEED_32,
                        "step": 1,
                        "tooltip": "Base seed for caption generation. -1 means unseeded when supported.",
                    },
                ),
                "Caption - seed mode": (
                    SEED_MODES,
                    {"default": "fixed"},
                ),
                "Caption - temperature schedule": (
                    "STRING",
                    {
                        "default": DEFAULT_CAPTION_TEMPERATURE_SCHEDULE,
                        "multiline": False,
                        "tooltip": "Comma-separated caption temperature schedule; final value repeats if needed.",
                    },
                ),
                "Caption - top p schedule": (
                    "STRING",
                    {"default": DEFAULT_CAPTION_TOP_P_SCHEDULE, "multiline": False},
                ),
                "Caption - top k schedule": (
                    "STRING",
                    {"default": DEFAULT_CAPTION_TOP_K_SCHEDULE, "multiline": False},
                ),
                "Caption - max image size": (
                    "INT",
                    {"default": DEFAULT_CAPTION_MAX_IMAGE_SIZE, "min": 0, "max": 4096, "step": 64},
                ),
                "Caption - max new tokens": (
                    "INT",
                    {"default": DEFAULT_CAPTION_MAX_NEW_TOKENS, "min": 16, "max": 12000, "step": 64},
                ),

                # -----------------------------------------------------------------
                # Distiller controls.
                # -----------------------------------------------------------------
                "Distiller - model": (
                    DISTILLER_MODEL_CHOICES,
                    {
                        "default": DEFAULT_DISTILLER_MODEL,
                        "tooltip": (
                            "Concrete Ollama text model tag for the distiller. Use custom to enter "
                            "any other installed Ollama text model."
                        ),
                    },
                ),
                "Distiller - custom Ollama model": (
                    "STRING",
                    {
                        "default": "",
                        "multiline": False,
                        "tooltip": "Used only when Distiller - model is custom, e.g. my-model:latest.",
                    },
                ),
                "Distiller - base seed": (
                    "INT",
                    {
                        "default": -1,
                        "min": -1,
                        "max": MAX_SEED_32,
                        "step": 1,
                        "tooltip": "Base seed for the distiller. -1 means omit seed.",
                    },
                ),
                "Distiller - seed mode": (
                    SEED_MODES,
                    {"default": "fixed"},
                ),
                "Distiller - strategy": (
                    DISTILLER_STRATEGIES,
                    {"default": "single_pass"},
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

                # -----------------------------------------------------------------
                # Validator controls in the same order pattern.
                # -----------------------------------------------------------------
                "Validator - model": (
                    VALIDATOR_MODEL_CHOICES,
                    {
                        "default": DEFAULT_VALIDATOR_MODEL,
                        "tooltip": (
                            "Concrete Ollama vision model tag for the image-aware validator. Use custom "
                            "to enter any other installed Ollama vision model."
                        ),
                    },
                ),
                "Validator - custom Ollama model": (
                    "STRING",
                    {
                        "default": "",
                        "multiline": False,
                        "tooltip": "Used only when Validator - model is custom, e.g. gemma4:e4b or another installed VLM tag.",
                    },
                ),
                "Validator - base seed": (
                    "INT",
                    {
                        "default": -1,
                        "min": -1,
                        "max": MAX_SEED_32,
                        "step": 1,
                        "tooltip": "Base seed for the VLM validator. -1 means omit seed.",
                    },
                ),
                "Validator - seed mode": (
                    SEED_MODES,
                    {"default": "fixed"},
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

                # -----------------------------------------------------------------
                # Final export controls.
                # -----------------------------------------------------------------
                "Final - caption style": (
                    FINAL_CAPTION_STYLES,
                    {"default": "narrative"},
                ),
                "Final - write TXT sidecars": (
                    "BOOLEAN",
                    {
                        "default": True,
                        "tooltip": (
                            "Final TXT sidecars use the associated image filename stem and .txt extension."
                        ),
                    },
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
                            "Optional IMAGE passthrough for quick single-image workflows. "
                            "The planner does not process this image; it simply returns it as output."
                        ),
                    },
                ),
            },
        }

    RETURN_TYPES = ("IMAGE", "CAPTIONFORGE_PIPELINE_PLAN", "STRING")
    RETURN_NAMES = ("single_image", "pipeline_plan", "pipeline_plan_json")
    FUNCTION = "plan"
    CATEGORY = "Captioning/CaptionForge"

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        return float("NaN")

    def plan(self, **kwargs):
        output_dir = str(kwargs.get("Output - folder", "") or "").strip() or _default_output_dir()
        run_name = _clean_run_name(kwargs.get("Output - run name", "captionforge_run"))
        single_image = kwargs.get("Input - single image", None)
        single_image_connected = single_image is not None

        distiller_model = _resolve_ollama_model_name(
            kwargs.get("Distiller - model", DEFAULT_DISTILLER_MODEL),
            kwargs.get("Distiller - custom Ollama model", ""),
            DEFAULT_DISTILLER_MODEL,
        )
        validator_model = _resolve_ollama_model_name(
            kwargs.get("Validator - model", DEFAULT_VALIDATOR_MODEL),
            kwargs.get("Validator - custom Ollama model", ""),
            DEFAULT_VALIDATOR_MODEL,
        )

        planner_enabled = _as_bool(kwargs.get("Planner - enabled", True))
        joy_runs = _runs_per_image(kwargs.get("Caption - Joy runs/image", "2"), "2")
        qwen_runs = _runs_per_image(kwargs.get("Caption - Qwen runs/image", "2"), "2")
        ollama_runs = _runs_per_image(kwargs.get("Caption - Ollama runs/image", "Disabled"), "Disabled")
        smolvlm_runs = 0

        if not planner_enabled:
            plan: dict[str, Any] = {}
            return (single_image, plan, json.dumps(plan, ensure_ascii=False, indent=2))

        plan = _call_build_captionforge_pipeline_plan_compat(
            output_dir=output_dir,
            run_name=run_name,
            input_path=str(kwargs.get("Input - image path", "") or "").strip(),
            single_image_connected=single_image_connected,
            recursive=_as_bool(kwargs.get("Input - recursive", True)),
            filename_glob=str(kwargs.get("Input - filename glob", "*") or "*").strip() or "*",
            joy_runs_per_image=joy_runs,
            qwen_runs_per_image=qwen_runs,
            ollama_runs_per_image=ollama_runs,
            ollama_caption_runs_per_image=ollama_runs,
            caption_ollama_runs_per_image=ollama_runs,
            ollama_vlm_runs_per_image=ollama_runs,
            # Deprecated placeholders are kept at zero for compatibility with
            # older planner-engine signatures, but are no longer exposed in UI.
            smolvlm_runs_per_image=0,
            smol_runs_per_image=0,
            florence_runs_per_image=0,
            llama_vision_runs_per_image=0,
            base_seed=int(kwargs.get("Caption - base seed", -1) or -1),
            seed_mode=str(kwargs.get("Caption - seed mode", "fixed") or "fixed"),
            temperature_schedule=str(kwargs.get("Caption - temperature schedule", DEFAULT_CAPTION_TEMPERATURE_SCHEDULE) or DEFAULT_CAPTION_TEMPERATURE_SCHEDULE),
            top_p_schedule=str(kwargs.get("Caption - top p schedule", DEFAULT_CAPTION_TOP_P_SCHEDULE) or DEFAULT_CAPTION_TOP_P_SCHEDULE),
            top_k_schedule=str(kwargs.get("Caption - top k schedule", DEFAULT_CAPTION_TOP_K_SCHEDULE) or DEFAULT_CAPTION_TOP_K_SCHEDULE),
            max_size=int(kwargs.get("Caption - max image size", DEFAULT_CAPTION_MAX_IMAGE_SIZE) or DEFAULT_CAPTION_MAX_IMAGE_SIZE),
            max_new_tokens=int(kwargs.get("Caption - max new tokens", DEFAULT_CAPTION_MAX_NEW_TOKENS) or DEFAULT_CAPTION_MAX_NEW_TOKENS),
            trigger_word=str(kwargs.get("LoRA - trigger word", "") or "").strip(),
            user_caption_anchor=str(kwargs.get("LoRA - user caption anchor", "") or "").strip(),
            distiller_model=distiller_model,
            distiller_model_family=distiller_model,
            distiller_base_seed=int(kwargs.get("Distiller - base seed", -1) or -1),
            distiller_seed_mode=str(kwargs.get("Distiller - seed mode", "fixed") or "fixed"),
            distiller_strategy=str(kwargs.get("Distiller - strategy", "single_pass") or "single_pass"),
            distiller_max_caption_chars_for_llm=int(kwargs.get("Distiller - max caption chars for LLM", 1536) or 1536),
            distiller_num_predict=int(kwargs.get("Distiller - num predict", 3096) or 3096),
            distiller_temperature=float(kwargs.get("Distiller - temperature", 0.24) or 0.0),
            distiller_top_p=float(kwargs.get("Distiller - top p", 0.90) or 0.90),
            distiller_top_k=int(kwargs.get("Distiller - top k", 60) or 60),
            distiller_write_prompt_jsonl=_as_bool(kwargs.get("Distiller - write prompt JSONL", False)),
            distiller_preserve_raw_response=_as_bool(kwargs.get("Distiller - preserve raw response", False)),
            validator_model=validator_model,
            validator_model_family=validator_model,
            validator_base_seed=int(kwargs.get("Validator - base seed", -1) or -1),
            validator_seed_mode=str(kwargs.get("Validator - seed mode", "fixed") or "fixed"),
            validator_num_predict=int(kwargs.get("Validator - num predict", 2200) or 2200),
            validator_temperature=float(kwargs.get("Validator - temperature", 0.0) or 0.0),
            validator_top_p=float(kwargs.get("Validator - top p", 0.92) or 0.92),
            validator_top_k=int(kwargs.get("Validator - top k", 80) or 80),
            validator_write_prompt_jsonl=_as_bool(kwargs.get("Validator - write prompt JSONL", False)),
            validator_preserve_raw_vlm_response=_as_bool(kwargs.get("Validator - preserve raw VLM response", False)),
            final_caption_style=str(kwargs.get("Final - caption style", "narrative") or "narrative"),
            final_write_txt_sidecars=_as_bool(kwargs.get("Final - write TXT sidecars", True)),
            final_write_jsonl=_as_bool(kwargs.get("Final - write JSONL", True)),
            overwrite_outputs=_as_bool(kwargs.get("Output - overwrite outputs", True)),
        )
        plan = _patch_supported_caption_witnesses(
            plan,
            joy_runs=joy_runs,
            qwen_runs=qwen_runs,
            ollama_runs=ollama_runs,
        )

        overwrite_outputs = _as_bool(kwargs.get("Output - overwrite outputs", True))
        _reset_pass_a_jsonl_for_overwrite(plan, overwrite_outputs=overwrite_outputs)

        return (single_image, plan, json.dumps(plan, ensure_ascii=False, indent=2))


NODE_CLASS_MAPPINGS = {
    "JLC_CaptionForge_Pipeline_Planner": JLC_CaptionForge_Pipeline_Planner,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "JLC_CaptionForge_Pipeline_Planner": "\u2003JLC CaptionForge Pipeline Planner",
}
