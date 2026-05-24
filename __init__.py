"""
CaptionForge — ComfyUI Package Entry Point

- CaptionForge
  - This package is part of **CaptionForge**, a model-agnostic captioning
    framework for ComfyUI developed by **J. L. Córdova**.

  - Repository
    https://github.com/Damkohler/CaptionForge

  - CaptionForge focuses on practical dataset-captioning infrastructure for:
        • LoRA dataset preparation
        • multi-engine caption generation
        • JSONL audit trails
        • claim extraction and refinement
        • consensus-oriented caption improvement

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

from .jlc_qwen_caption_CUI_node import (
    NODE_CLASS_MAPPINGS as QWEN_NODE_CLASS_MAPPINGS,
    NODE_DISPLAY_NAME_MAPPINGS as QWEN_NODE_DISPLAY_NAME_MAPPINGS,
)

from .jlc_joy_caption_CUI_node import (
    NODE_CLASS_MAPPINGS as JOY_NODE_CLASS_MAPPINGS,
    NODE_DISPLAY_NAME_MAPPINGS as JOY_NODE_DISPLAY_NAME_MAPPINGS,
)

from .jlc_qwen_caption_lite_CUI_node import (
    NODE_CLASS_MAPPINGS as QWEN_LITE_CLASS_MAPPINGS,
    NODE_DISPLAY_NAME_MAPPINGS as QWEN_LITE_DISPLAY_NAME_MAPPINGS,
)

from .jlc_joy_caption_lite_CUI_node import (
    NODE_CLASS_MAPPINGS as JOY_LITE_CLASS_MAPPINGS,
    NODE_DISPLAY_NAME_MAPPINGS as JOY_LITE_DISPLAY_NAME_MAPPINGS,
)

from .jlc_captionforge_claim_extractor_CUI_node import (
    NODE_CLASS_MAPPINGS as CLAIM_EXTRACTOR_CLASS_MAPPINGS,
    NODE_DISPLAY_NAME_MAPPINGS as CLAIM_EXTRACTOR_DISPLAY_NAME_MAPPINGS,
)

NODE_CLASS_MAPPINGS = {}
NODE_CLASS_MAPPINGS.update(QWEN_NODE_CLASS_MAPPINGS)
NODE_CLASS_MAPPINGS.update(JOY_NODE_CLASS_MAPPINGS)
NODE_CLASS_MAPPINGS.update(QWEN_LITE_CLASS_MAPPINGS)
NODE_CLASS_MAPPINGS.update(JOY_LITE_CLASS_MAPPINGS)
NODE_CLASS_MAPPINGS.update(CLAIM_EXTRACTOR_CLASS_MAPPINGS)

NODE_DISPLAY_NAME_MAPPINGS = {}
NODE_DISPLAY_NAME_MAPPINGS.update(QWEN_NODE_DISPLAY_NAME_MAPPINGS)
NODE_DISPLAY_NAME_MAPPINGS.update(JOY_NODE_DISPLAY_NAME_MAPPINGS)
NODE_DISPLAY_NAME_MAPPINGS.update(QWEN_LITE_DISPLAY_NAME_MAPPINGS)
NODE_DISPLAY_NAME_MAPPINGS.update(JOY_LITE_DISPLAY_NAME_MAPPINGS)
NODE_DISPLAY_NAME_MAPPINGS.update(CLAIM_EXTRACTOR_DISPLAY_NAME_MAPPINGS)

# WEB_DIRECTORY
WEB_DIRECTORY = "./web"

CAPTIONFORGE_ICON = "⚒"
CAPTIONFORGE_PREFIX = f"{CAPTIONFORGE_ICON}  CaptionForge"
print(f"{CAPTIONFORGE_PREFIX} loaded ({len(NODE_CLASS_MAPPINGS)} nodes)")

__all__ = [
    "NODE_CLASS_MAPPINGS",
    "NODE_DISPLAY_NAME_MAPPINGS",
    "WEB_DIRECTORY",
]