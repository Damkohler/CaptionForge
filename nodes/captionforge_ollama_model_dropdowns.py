"""
CaptionForge Ollama Model Dropdown Loader

- CaptionForge
  - Shared ComfyUI node helper for loading user-editable Ollama model dropdown
    choices for the CaptionForge Pipeline Planner and JLC CaptionForge capstone
    node.

  - Repository
    https://github.com/Damkohler/CaptionForge

- Purpose
    - Reads config/captionforge_ollama_models.json at ComfyUI custom-node load
      time.
    - Provides explicit Ollama model tags for Distiller and Validator dropdowns.
    - Keeps dropdown configuration out of the engines and out of duplicate node
      constants.
    - Preserves a custom escape hatch for installed models not listed in JSON.

- Design Notes
    - This helper intentionally does not use family aliases or shorthand names.
    - Dropdown values are concrete Ollama tags and are used exactly as written.
    - The reusable engines remain model-agnostic and simply receive resolved
      model names through their existing config objects.

- Attribution & License
  - CaptionForge is an original concept and implementation by J. L. Córdova
    with development assistance from ChatGPT (OpenAI). It is not derived from
    or based on another ComfyUI workflow.
  - Designed for use with:
    https://github.com/comfyanonymous/ComfyUI
  - Copyright (c) 2026 J. L. Córdova
  - Released under the MIT License.
"""

from __future__ import annotations

MANIFEST = {
    "name": "CaptionForge Ollama Model Dropdown Loader",
    "version": (0, 1, 0),
    "author": "J. L. Córdova",
    "description": (
        "Shared CaptionForge helper that loads explicit Ollama model tags from "
        "config/captionforge_ollama_models.json for the Pipeline Planner and "
        "JLC CaptionForge capstone node dropdowns. Uses no family aliases; "
        "dropdown values are concrete model tags used exactly as written."
    ),
}

import json
from pathlib import Path
from typing import Any

FALLBACK_DISTILLER_MODELS = ["llama3.1:8b"]
FALLBACK_VALIDATOR_MODELS = ["gemma4:e4b"]
FALLBACK_DISTILLER_DEFAULT = "llama3.1:8b"
FALLBACK_VALIDATOR_DEFAULT = "gemma4:e4b"
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
    """Return CaptionForge repo root from a node file path.

    Node wrappers usually live in CaptionForge/nodes, so the repo root is the
    parent of the node directory. If the file is placed directly in the root
    during manual testing, this still returns a sensible parent.
    """
    p = Path(node_file).resolve()
    return p.parent.parent if p.parent.name.lower() == "nodes" else p.parent


def load_ollama_model_dropdowns(node_file: str | Path) -> dict[str, Any]:
    root = _project_root_from_node_file(node_file)
    config_path = root / "config" / "captionforge_ollama_models.json"

    distiller_models = list(FALLBACK_DISTILLER_MODELS)
    validator_models = list(FALLBACK_VALIDATOR_MODELS)
    distiller_default = FALLBACK_DISTILLER_DEFAULT
    validator_default = FALLBACK_VALIDATOR_DEFAULT
    include_custom = True

    try:
        if config_path.exists() and config_path.is_file():
            data = json.loads(config_path.read_text(encoding="utf-8"))
            if isinstance(data, dict):
                if isinstance(data.get("distiller_models"), list):
                    distiller_models = _dedupe_keep_order(data["distiller_models"])
                if isinstance(data.get("validator_models"), list):
                    validator_models = _dedupe_keep_order(data["validator_models"])

                defaults = data.get("defaults")
                if isinstance(defaults, dict):
                    distiller_default = str(defaults.get("distiller_model") or distiller_default).strip()
                    validator_default = str(defaults.get("validator_model") or validator_default).strip()

                include_custom = _as_bool(data.get("include_custom"), True)
    except Exception as exc:
        print(
            f"[CaptionForge] WARNING: Could not load Ollama model dropdown config: "
            f"{config_path} | {exc}",
            flush=True,
        )

    if not distiller_models:
        distiller_models = list(FALLBACK_DISTILLER_MODELS)
    if not validator_models:
        validator_models = list(FALLBACK_VALIDATOR_MODELS)

    if distiller_default and distiller_default not in distiller_models:
        distiller_models.insert(0, distiller_default)
    if validator_default and validator_default not in validator_models:
        validator_models.insert(0, validator_default)

    if include_custom:
        if CUSTOM_VALUE not in distiller_models:
            distiller_models.append(CUSTOM_VALUE)
        if CUSTOM_VALUE not in validator_models:
            validator_models.append(CUSTOM_VALUE)

    return {
        "distiller_models": distiller_models,
        "validator_models": validator_models,
        "distiller_default": distiller_default or FALLBACK_DISTILLER_DEFAULT,
        "validator_default": validator_default or FALLBACK_VALIDATOR_DEFAULT,
        "config_path": str(config_path),
    }
