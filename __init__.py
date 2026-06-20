"""
CaptionForge — ComfyUI Package Entry Point

- CaptionForge
  - This package is part of **CaptionForge**, a model-agnostic captioning
    framework for ComfyUI developed by **J. L. Córdova**.

  - Repository
    https://github.com/Damkohler/CaptionForge

- CaptionForge focuses on practical dataset-captioning infrastructure for
  LoRA dataset preparation, using multi-engine caption generation, JSONL
  audit trails, claim extraction and refinement, text-LLM distillation,
  image-aware VLM validation, and consensus-oriented caption improvement
  to produce grounded, auditable training captions.

- Package Purpose
    - This file is the ComfyUI registration entry point for the CaptionForge
      custom node package.

    - It exposes:
            • NODE_CLASS_MAPPINGS
            • NODE_DISPLAY_NAME_MAPPINGS

      so ComfyUI can:
            • discover the Python node classes
            • display the nodes in the Add Node menu
            • register the package as a unified CaptionForge node collection

    - The package currently registers:
            • JLC Qwen Caption
            • JLC Joy Caption
            • JLC Qwen Caption (Lite)
            • JLC Joy Caption (Lite)
            • JLC CaptionForge Claim Extractor

- Package Structure
    - CaptionForge keeps ComfyUI-facing node wrappers separate from reusable
      engine modules.

    - Node wrappers define ComfyUI widgets, IMAGE input handling, output strings,
      categories, and node mappings.

    - Engine modules handle model loading, prompt resolution, generation,
      cleanup, folder traversal, TXT sidecars, JSONL audit records, run-config
      export, model cache behavior, and Pass B claim extraction.

- Web / Icon Assets
    - JavaScript/icon registration is intentionally disabled in this entry point
      for now to avoid frontend branding conflicts while the package structure
      stabilizes.

    - A future version may re-enable WEB_DIRECTORY after confirming that the
      CaptionForge frontend assets do not conflict with other JLC node packages.

- Attribution & License
  - Concept and implementation by **J. L. Córdova**
    with development assistance from **ChatGPT (OpenAI)**.

  - Designed for use with:
    https://github.com/comfyanonymous/ComfyUI

  - Copyright (c) 2026 J. L. Córdova

  - Released under the **MIT License**.
"""

from __future__ import annotations

# ######################
# Planner node
# ######################
from .nodes.jlc_captionforge_pipeline_planner_node import (
    NODE_CLASS_MAPPINGS as PIPELINE_PLANNER_CLASS_MAPPINGS,
    NODE_DISPLAY_NAME_MAPPINGS as PIPELINE_PLANNER_DISPLAY_NAME_MAPPINGS,
)

# ######################
# Caption nodes
# ######################
# # Helper
from .nodes.jlc_captionforge_template_options import (
    NODE_CLASS_MAPPINGS as CAPTIONFORGE_EXTRA_OPTIONS_CLASS_MAPPINGS,
    NODE_DISPLAY_NAME_MAPPINGS as CAPTIONFORGE_EXTRA_OPTIONS_DISPLAY_NAME_MAPPINGS,
)

# Joy
from .nodes.caption_nodes.jlc_captionforge_joy_caption_node import (
    NODE_CLASS_MAPPINGS as JOY_NODE_CLASS_MAPPINGS,
    NODE_DISPLAY_NAME_MAPPINGS as JOY_NODE_DISPLAY_NAME_MAPPINGS,
)

# Ollama Caption Models
from .nodes.caption_nodes.jlc_captionforge_ollama_caption_node import (
    NODE_CLASS_MAPPINGS as OLLAMA_NODE_CLASS_MAPPINGS,
    NODE_DISPLAY_NAME_MAPPINGS as OLLAMA_NODE_DISPLAY_NAME_MAPPINGS,
)

#Qwen
from .nodes.caption_nodes.jlc_captionforge_qwen_caption_node import (
    NODE_CLASS_MAPPINGS as QWEN_NODE_CLASS_MAPPINGS,
    NODE_DISPLAY_NAME_MAPPINGS as QWEN_NODE_DISPLAY_NAME_MAPPINGS,
)


# ######################
# CaptionForge node
# ######################
from .nodes.jlc_captionforge_node import (
    NODE_CLASS_MAPPINGS as CAPTIONFORGE_CLASS_MAPPINGS,
    NODE_DISPLAY_NAME_MAPPINGS as CAPTIONFORGE_DISPLAY_NAME_MAPPINGS,
)

# #####################
# Class mappings
# #####################
NODE_CLASS_MAPPINGS = {}
NODE_CLASS_MAPPINGS.update(PIPELINE_PLANNER_CLASS_MAPPINGS)
NODE_CLASS_MAPPINGS.update(CAPTIONFORGE_EXTRA_OPTIONS_CLASS_MAPPINGS)
NODE_CLASS_MAPPINGS.update(JOY_NODE_CLASS_MAPPINGS)
NODE_CLASS_MAPPINGS.update(OLLAMA_NODE_CLASS_MAPPINGS)
NODE_CLASS_MAPPINGS.update(QWEN_NODE_CLASS_MAPPINGS)
NODE_CLASS_MAPPINGS.update(CAPTIONFORGE_CLASS_MAPPINGS)

# ######################
# Display name mappings
# ######################
NODE_DISPLAY_NAME_MAPPINGS = {}
NODE_DISPLAY_NAME_MAPPINGS.update(CAPTIONFORGE_EXTRA_OPTIONS_DISPLAY_NAME_MAPPINGS)
NODE_DISPLAY_NAME_MAPPINGS.update(PIPELINE_PLANNER_DISPLAY_NAME_MAPPINGS)
NODE_DISPLAY_NAME_MAPPINGS.update(JOY_NODE_DISPLAY_NAME_MAPPINGS)
NODE_DISPLAY_NAME_MAPPINGS.update(OLLAMA_NODE_DISPLAY_NAME_MAPPINGS)
NODE_DISPLAY_NAME_MAPPINGS.update(QWEN_NODE_DISPLAY_NAME_MAPPINGS)
NODE_DISPLAY_NAME_MAPPINGS.update(CAPTIONFORGE_DISPLAY_NAME_MAPPINGS)


WEB_DIRECTORY = "./web"

CAPTIONFORGE_ICON = "⚒"
CAPTIONFORGE_PREFIX = f"{CAPTIONFORGE_ICON}  CaptionForge"
print(f"{CAPTIONFORGE_PREFIX} loaded ({len(NODE_CLASS_MAPPINGS)} nodes)")

__all__ = [
    "NODE_CLASS_MAPPINGS",
    "NODE_DISPLAY_NAME_MAPPINGS",
    "WEB_DIRECTORY",
]