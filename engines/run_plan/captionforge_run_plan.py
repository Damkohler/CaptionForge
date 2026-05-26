"""
CaptionForge Run Plan Helpers — Shared Pass A Configuration Utilities

- CaptionForge
  - This module is part of **CaptionForge**, a model-agnostic captioning
    framework for ComfyUI developed by **J. L. Córdova**.

  - Repository
    https://github.com/Damkohler/CaptionForge

- Module Purpose
    - This module contains the shared, model-agnostic logic behind the
      JLC CaptionForge Run Plan node.

    - It does not load models.

    - It does not caption images.

    - It does not know about Qwen, Joy, or any specific captioning engine.

    - It only:
            • builds a normalized CaptionForge run config dictionary
            • parses connected run-config objects or JSON strings
            • expands schedules into per-caption-instance settings
            • derives per-caption-instance seeds
            • provides a small immutable CaptionForgeRun dataclass

- Pass A Role
    - CaptionForge Pass A generates caption evidence records.

    - A single Run Plan may be shared by multiple independent captioning engines
      so that Qwen, Joy, and future engines generate comparable evidence under
      the same run-level constraints.

- Run Expansion Contract
    - A connected CAPTIONFORGE_RUN_CONFIG overrides matching node-local widget
      values.

    - If no Run Plan is connected, caption nodes may still call this module with
      widget fallbacks and behave as standalone tools.

    - Schedules are intentionally forgiving:
            • blank schedule -> use node-local scalar fallback
            • shorter-than-n schedule -> repeat final schedule value
            • malformed schedule values -> ignored

    - This supports practical workflows such as varying temperature while
      keeping top-p and top-k constant.

- Seed Contract
    - seed_mode controls per-caption-instance seed generation:
            • fixed      -> same seed for all ensemble runs
            • increment  -> base_seed + run_index
            • decrement  -> base_seed - run_index, clamped to zero
            • random     -> pseudo-random sequence from base_seed, or
                            nondeterministic seeds when base_seed is -1

    - Matching run indices across different caption engines intentionally receive
      matching seeds so evidence records are paired by ensemble_run_index.

- Output Directory Contract
    - output_dir is carried through the run config and expanded runs.

    - The Run Plan node requires it in CaptionForge mode.

    - Standalone caption nodes may still use their own fallback directories when
      no Run Plan is connected.

- Token Budget Contract
    - max_new_tokens is treated as a shared constant, not a schedule.

    - This avoids biasing downstream claim extraction by letting one caption
      instance produce much more verbose or speculative text than another.

- Design Philosophy
    - This helper is intentionally small and strict.

    - It centralizes CaptionForge run-plan semantics so Qwen, Joy, Lite nodes,
      and future caption engines do not each reimplement seed/schedule logic.

- ⚠️ Development Status
    - Early CaptionForge Pass A infrastructure.
    - Config schema may evolve before CaptionForge v1.0.0.

- Attribution & License
  - Concept and implementation by **J. L. Córdova**
    with development assistance from **ChatGPT (OpenAI)**.

  - Copyright (c) 2026 J. L. Córdova

  - Released under the **MIT License**.
"""

from __future__ import annotations

MANIFEST = {
    "name": "CaptionForge Run Plan Helpers",
    "version": (0, 1, 0),
    "author": "J. L. Córdova",
    "description": (
        "Shared model-agnostic helper module for CaptionForge Pass A run "
        "planning. Builds normalized run-config dictionaries, parses connected "
        "CAPTIONFORGE_RUN_CONFIG objects or JSON strings, expands per-caption "
        "sampling schedules, derives per-caption-instance seeds, carries the "
        "required shared output_dir, and returns immutable CaptionForgeRun "
        "objects for Qwen, Joy, Lite nodes, and future captioning engines. "
        "Centralizes run-plan semantics so independent caption nodes can produce "
        "coordinated JSONL evidence records for downstream claim extraction and "
        "final CaptionForge caption synthesis."
    ),
}

import json
import random
from dataclasses import dataclass
from typing import Any


MAX_SEED_32 = 0xFFFFFFFF


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
    """
    CaptionForge schedule rule:
    - run index inside list range: use that item
    - run index beyond list range: repeat last item
    """
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


def _seed_for_run(base_seed: int, seed_mode: str, index: int) -> int | None:
    """
    Existing Qwen/Joy widgets use -1 as nondeterministic/no fixed seed.

    For random:
    - base_seed < 0 gives nondeterministic seeds.
    - base_seed >= 0 gives a reproducible pseudo-random seed sequence.
    """
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
        for _ in range(index + 1):
            out = rng.randint(0, MAX_SEED_32)
        return out

    return base_seed


def normalize_captionforge_run_config(config: Any) -> dict[str, Any]:
    """
    Accept either:
    - dict from a connected CAPTIONFORGE_RUN_CONFIG socket
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
) -> dict[str, Any]:
    return {
        "captionforge_config_type": "captionforge_run_plan",
        "captionforge_config_version": 1,

        "captions_per_image": _coerce_int(captions_per_image, 1, 1, 100),
        "base_seed": _coerce_int(base_seed, -1, -1, MAX_SEED_32),
        "seed_mode": _normalize_seed_mode(seed_mode),

        "temperature_schedule": str(temperature_schedule or "").strip(),
        "top_p_schedule": str(top_p_schedule or "").strip(),
        "top_k_schedule": str(top_k_schedule or "").strip(),

        "max_size": _coerce_int(max_size, 1024, 0, 4096),
        "max_new_tokens": _coerce_int(max_new_tokens, 512, 16, 4096),
        "trigger_word": str(trigger_word or "").strip(),
        "output_dir": str(output_dir or "").strip(),
    }


def expand_captionforge_runs(
    captionforge_run_config: Any = None,
    *,
    widget_captions_per_image: int = 1,
    widget_seed: int = -1,
    widget_temperature: float = 0.75,
    widget_top_p: float = 0.90,
    widget_top_k: int = 50,
    widget_max_new_tokens: int = 512,
    widget_max_size: int = 1024,
    widget_trigger_word: str = "",
    widget_output_dir: str = "",
) -> list[CaptionForgeRun]:
    """
    Expand connected Run Plan + node widget fallbacks into per-caption settings.

    Connected config overrides matching values. Blank schedules fall back to
    widget scalar values. Short schedules repeat the last provided value.
    """
    cfg = normalize_captionforge_run_config(captionforge_run_config)

    captions_per_image = _coerce_int(
        cfg.get("captions_per_image", widget_captions_per_image),
        widget_captions_per_image,
        1,
        100,
    )

    base_seed = _coerce_int(
        cfg.get("base_seed", widget_seed),
        widget_seed,
        -1,
        MAX_SEED_32,
    )

    seed_mode = _normalize_seed_mode(cfg.get("seed_mode", "fixed"))

    max_size = _coerce_int(
        cfg.get("max_size", widget_max_size),
        widget_max_size,
        0,
        4096,
    )

    max_new_tokens = _coerce_int(
        cfg.get("max_new_tokens", widget_max_new_tokens),
        widget_max_new_tokens,
        16,
        4096,
    )

    trigger_word = str(cfg.get("trigger_word", widget_trigger_word) or "").strip()
    output_dir = str(cfg.get("output_dir", widget_output_dir) or "").strip()

    temperatures = _parse_schedule(
        cfg.get("temperature_schedule", ""),
        float,
        _coerce_float(widget_temperature, 0.75, 0.0, 2.0),
    )
    top_ps = _parse_schedule(
        cfg.get("top_p_schedule", ""),
        float,
        _coerce_float(widget_top_p, 0.90, 0.0, 1.0),
    )
    top_ks = _parse_schedule(
        cfg.get("top_k_schedule", ""),
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
            )
        )

    return runs