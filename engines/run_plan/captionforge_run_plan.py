"""
CaptionForge Run Plan helpers

Small shared utility for optional Pass A ensemble-run configuration.

This module is intentionally model-agnostic. It does not know about Qwen,
Joy, or any future captioning engine. It only normalizes a shared run plan
and expands it into per-caption generation settings.

Design rules:
- The Run Plan is optional.
- Connected Run Plan values override matching node widgets.
- Blank schedule fields fall back to node-local widget values.
- Short schedules repeat their last value.
- max_new_tokens is a shared constant, not a diversity schedule.
- max_size is a shared workload guard.
- trigger_word is a shared LoRA training identity token.
- input_path / recursive / filename_glob are shared dataset-routing controls.
"""

from __future__ import annotations

import json
import random
from dataclasses import dataclass
from typing import Any


MANIFEST = {
    "name": "CaptionForge Run Plan Helpers",
    "version": (0, 2, 0),
    "author": "J. L. Córdova",
    "description": (
        "Shared model-agnostic helper module for CaptionForge Pass A run planning. "
        "Builds normalized run-config dictionaries, parses connected "
        "CAPTIONFORGE_RUN_CONFIG objects or JSON strings, expands per-caption "
        "sampling schedules, derives per-caption-instance seeds, carries shared "
        "output_dir and optional input_path/recursive/filename_glob routing, and "
        "returns immutable CaptionForgeRun objects for Qwen, Joy, Lite nodes, and "
        "future captioning engines."
    ),
}


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
        out = base_seed
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
    input_path: str = "",
    recursive: bool = True,
    filename_glob: str = "*",
) -> dict[str, Any]:
    return {
        "captionforge_config_type": "captionforge_run_plan",
        "captionforge_config_version": 2,

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

        "input_path": str(input_path or "").strip(),
        "recursive": _coerce_bool(recursive, True),
        "filename_glob": str(filename_glob or "*").strip() or "*",
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
    widget_input_path: str = "",
    widget_recursive: bool = True,
    widget_filename_glob: str = "*",
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
    input_path = str(cfg.get("input_path", widget_input_path) or "").strip()
    recursive = _coerce_bool(cfg.get("recursive", widget_recursive), bool(widget_recursive))
    filename_glob = str(cfg.get("filename_glob", widget_filename_glob) or "*").strip() or "*"

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
                input_path=input_path,
                recursive=recursive,
                filename_glob=filename_glob,
            )
        )

    return runs
