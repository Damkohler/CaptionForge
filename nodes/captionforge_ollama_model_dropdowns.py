"""
CaptionForge Ollama Model Dropdown Helper

- CaptionForge
  - This helper is part of **CaptionForge**, a model-agnostic captioning
    framework for ComfyUI developed by **J. L. Córdova**.

  - Repository:
    https://github.com/Damkohler/CaptionForge

- CaptionForge focuses on practical dataset-captioning infrastructure for
  LoRA dataset preparation, using multi-engine caption generation, JSONL
  audit trails, claim extraction and refinement, text-LLM distillation,
  image-aware VLM validation, and consensus-oriented caption improvement
  to produce grounded, auditable training captions.

- Helper Purpose
    - The **CaptionForge Ollama Model Dropdown Helper** centralizes loading of
      user-editable Ollama model dropdown choices from:

            config/captionforge_ollama_models.json

    - It keeps Ollama model-list configuration out of individual node classes
      and out of backend engine code.

    - It provides explicit Ollama model tags and defaults for CaptionForge
      nodes that expose Ollama-backed model selectors, including text-LLM
      distillation, image-aware VLM validation/capstone stages, and optional
      Ollama caption-witness nodes.

    - It preserves a custom-model escape hatch so users can type locally
      installed or pullable Ollama tags that are not yet listed in the JSON
      configuration file.

- CaptionForge Pipeline Role
    - This helper is not itself a ComfyUI node and does not perform captioning.

    - It supports CaptionForge's Ollama-backed UI surfaces by providing one
      shared source of truth for model dropdown choices, defaults, config-path
      resolution, fallback values, de-duplication, and custom-entry behavior.

    - The actual model execution remains in the relevant CaptionForge node or
      engine. Ollama-backed nodes use local Ollama HTTP API behavior; this
      helper only supplies normalized model tag choices.

- Configuration Model
    - Dropdown values are concrete Ollama tags and are used exactly as written.

    - This helper intentionally avoids family aliases, shorthand names, or
      automatic model remapping.

    - Engines and nodes remain model-agnostic: they receive resolved model tags
      from their callers and do not need to know how dropdown choices were
      loaded.

- Design Philosophy
    - CaptionForge should avoid duplicated dropdown constants spread across
      multiple node wrappers.

    - A single helper makes pre-release cleanup easier, keeps ComfyUI widgets
      consistent, and allows users to edit one JSON file rather than patching
      several Python files.

    - The helper favors explicitness, reproducibility, and predictable fallback
      behavior over clever model discovery.

- Development Status
    - CaptionForge v0.1.0 experimental developer-preview infrastructure.
    - This helper is active shared infrastructure, but the exact JSON schema may
      evolve before the first stable CaptionForge release.

- Attribution & License
  - Concept and implementation by **J. L. Córdova**
    with development assistance from **ChatGPT (OpenAI)**.

  - Ollama-facing CaptionForge components are designed around local Ollama HTTP
    API behavior. This helper does not call Ollama directly; it only normalizes
    local configuration used by those components.

  - Designed for use with:
    https://github.com/comfyanonymous/ComfyUI

  - Copyright (c) 2026 J. L. Córdova

  - Released under the **MIT License**.
"""

from __future__ import annotations

from ..captionforge_version import CAPTIONFORGE_VERSION

MANIFEST = {
    "name": "CaptionForge Ollama Model Dropdown Helper",
    "version": CAPTIONFORGE_VERSION,
    "author": "J. L. Córdova",
    "description": (
        "Shared CaptionForge helper for loading user-editable Ollama model tag "
        "dropdown choices from config/captionforge_ollama_models.json. Provides "
        "centralized defaults, fallback values, de-duplication, and custom-entry "
        "handling for Ollama-backed CaptionForge model selectors while keeping "
        "model-list configuration out of individual node wrappers and engines."
    ),
}

import json
from pathlib import Path
from typing import Any

FALLBACK_DISTILLER_MODELS = ["llama3.1:8b"]
FALLBACK_VALIDATOR_MODELS = ["gemma4:e4b"]
FALLBACK_FORMAT_MODELS = ["llama3.1:8b"]
FALLBACK_CAPTION_MODELS = ["gemma4:26b", "qwen3.6:35B-A3B"]

FALLBACK_DISTILLER_DEFAULT = "llama3.1:8b"
FALLBACK_VALIDATOR_DEFAULT = "gemma4:e4b"
FALLBACK_FORMAT_DEFAULT = "llama3.1:8b"
FALLBACK_CAPTION_DEFAULT = "gemma4:26b"

CUSTOM_VALUE = "custom"


def _dedupe_keep_order(values: list[Any]) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for value in values:
        text = str(value or "").strip()
        if not text or text in seen:
            continue
        seen.add(text)
        out.append(text)
    return out


def _as_bool(value: Any, default: bool = True) -> bool:
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


def _project_root_from_node_file(node_file: str | Path) -> Path:
    """Return CaptionForge repo root from a node/helper file path.

    Supports files located at:
        CaptionForge/nodes/*.py
        CaptionForge/nodes/caption_nodes/*.py
        CaptionForge/*.py

    The function prefers a discovered CaptionForge root containing either
    captionforge_version.py or config/captionforge_ollama_models.json.
    """
    p = Path(node_file).resolve()
    start = p.parent if p.is_file() else p

    for candidate in (start, *start.parents):
        if (candidate / "captionforge_version.py").is_file():
            return candidate
        if (candidate / "config" / "captionforge_ollama_models.json").is_file():
            return candidate

    # Conservative structural fallbacks.
    if start.name.lower() == "caption_nodes" and start.parent.name.lower() == "nodes":
        return start.parent.parent
    if start.name.lower() == "nodes":
        return start.parent

    return start


def _models_from_config(
    data: dict[str, Any],
    key: str,
    fallback: list[str],
    *,
    legacy_key: str | None = None,
) -> list[str]:
    raw = data.get(key)

    if raw is None and legacy_key:
        raw = data.get(legacy_key)

    if isinstance(raw, list):
        cleaned = _dedupe_keep_order(raw)
        if cleaned:
            return cleaned

    return list(fallback)


def _default_from_config(
    defaults: dict[str, Any],
    key: str,
    fallback: str,
    *,
    legacy_key: str | None = None,
) -> str:
    value = defaults.get(key)

    if value is None and legacy_key:
        value = defaults.get(legacy_key)

    text = str(value or "").strip()
    return text or fallback


def _ensure_default_in_choices(choices: list[str], default: str) -> list[str]:
    default = str(default or "").strip()
    if default and default not in choices:
        return [default, *choices]
    return choices


def _append_custom_choice(choices: list[str], include_custom: bool) -> list[str]:
    out = list(choices)
    if include_custom and not any(str(x).strip().lower() == CUSTOM_VALUE for x in out):
        out.append(CUSTOM_VALUE)
    return out


def load_ollama_model_dropdowns(node_file: str | Path) -> dict[str, Any]:
    root = _project_root_from_node_file(node_file)
    config_path = root / "config" / "captionforge_ollama_models.json"

    data: dict[str, Any] = {}

    try:
        if config_path.exists() and config_path.is_file():
            loaded = json.loads(config_path.read_text(encoding="utf-8"))
            if isinstance(loaded, dict):
                data = loaded
    except Exception as exc:
        print(
            f"[CaptionForge] WARNING: Could not load Ollama model dropdown config: "
            f"{config_path} | {exc}",
            flush=True,
        )

    defaults = data.get("defaults")
    if not isinstance(defaults, dict):
        defaults = {}

    include_custom = _as_bool(data.get("include_custom"), True)

    distiller_models = _models_from_config(
        data,
        "distiller_models",
        FALLBACK_DISTILLER_MODELS,
    )
    validator_models = _models_from_config(
        data,
        "validator_models",
        FALLBACK_VALIDATOR_MODELS,
    )
    format_models = _models_from_config(
        data,
        "format_models",
        FALLBACK_FORMAT_MODELS,
    )
    caption_models = _models_from_config(
        data,
        "caption_models",
        FALLBACK_CAPTION_MODELS,
        legacy_key="ollama_caption_models",
    )

    distiller_default = _default_from_config(
        defaults,
        "distiller_model",
        FALLBACK_DISTILLER_DEFAULT,
    )
    validator_default = _default_from_config(
        defaults,
        "validator_model",
        FALLBACK_VALIDATOR_DEFAULT,
    )
    format_default = _default_from_config(
        defaults,
        "format_model",
        FALLBACK_FORMAT_DEFAULT,
    )
    caption_default = _default_from_config(
        defaults,
        "caption_model",
        FALLBACK_CAPTION_DEFAULT,
        legacy_key="ollama_caption_model",
    )

    distiller_models = _append_custom_choice(
        _ensure_default_in_choices(distiller_models, distiller_default),
        include_custom,
    )
    validator_models = _append_custom_choice(
        _ensure_default_in_choices(validator_models, validator_default),
        include_custom,
    )
    format_models = _append_custom_choice(
        _ensure_default_in_choices(format_models, format_default),
        include_custom,
    )
    caption_models = _append_custom_choice(
        _ensure_default_in_choices(caption_models, caption_default),
        include_custom,
    )

    return {
        "distiller_models": distiller_models,
        "validator_models": validator_models,
        "format_models": format_models,
        "caption_models": caption_models,
        "distiller_default": distiller_default,
        "validator_default": validator_default,
        "format_default": format_default,
        "caption_default": caption_default,
        "include_custom": include_custom,
        "config_path": str(config_path),
    }
