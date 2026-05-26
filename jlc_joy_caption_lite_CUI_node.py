"""
JLC Joy Caption (Lite) — ComfyUI Node Wrapper

- CaptionForge
  - This node is part of **CaptionForge**, a model-agnostic captioning
    framework for ComfyUI developed by **J. L. Córdova**.

  - Repository
    https://github.com/Damkohler/CaptionForge

  - CaptionForge focuses on practical dataset-captioning infrastructure for:
        • LoRA dataset preparation
        • multi-engine caption generation
        • JSONL audit trails
        • claim extraction and refinement
        • consensus-oriented caption improvement

- Node Purpose
    - The **JLC Joy Caption (Lite)** node provides a deliberately small
      daily-use ComfyUI frontend for JoyCaption/LLaVA-family captioning.

    - It is intended for quick direct IMAGE captioning when the full CaptionForge
      Joy node is more detailed than needed.

    - This file is the **ComfyUI-facing Lite wrapper**, not the reusable
      captioning engine. It is responsible for:
            • minimal ComfyUI INPUT_TYPES / widget definitions
            • direct IMAGE tensor input
            • IMAGE tensor conversion to PIL images
            • fixed model-root integration under `models/LLM/`
            • basic generation controls
            • Joy memory-mode selection
            • keep-loaded cache control
            • conservative system prompt routing
            • direct caption string output
            • node display name, category, and mapping registration
            • passing user settings into the shared Joy caption engine

    - The actual reusable captioning implementation lives in:
            jlc_joy_caption_engine.py

- CaptionForge Pass Role
    - This node participates in **Pass A** of the CaptionForge pipeline.

    - Unlike the full Joy node, this Lite node is optimized for quick
      direct-image captioning and does not expose the full file/folder,
      sidecar, JSONL, run-config, Joy template, prompt-file, or cleanup
      interface.

    - Its captions can still be used as caption evidence, but the Lite node is
      primarily designed for fast interactive use inside ComfyUI workflows.

- Lite Workflow Model
    - The node accepts:
            • direct ComfyUI IMAGE input only

    - The node returns:
            • one caption string containing captions for the input image batch

    - The Lite interface exposes only:
            • model
            • memory_mode
            • keep_loaded
            • caption instruction text
            • max_new_tokens
            • temperature
            • top_p
            • top_k

- Model and Memory Behavior
    - JoyCaption/LLaVA-family models are loaded from:
            ComfyUI/models/LLM/JLC_JoyCaption/

    - The Lite node uses the shared CaptionForge model cache through the Joy
      engine so repeated runs can reuse a loaded model when `keep_loaded` is
      enabled.

    - **Balanced (8-bit)** is the recommended CaptionForge default for Joy on
      constrained VRAM systems and maps to the shared engine’s bitsandbytes
      memory-efficient loading path.

- Design Philosophy
    - The Lite node exists for everyday captioning convenience.

    - It preserves the same engine-democratic CaptionForge architecture as the
      full node while avoiding unnecessary UI surface area for simple direct
      IMAGE captioning tasks.

    - The full Joy node remains the preferred interface for batch dataset work,
      JSONL audit output, TXT sidecars, run-config logging, Joy-native template
      controls, and downstream Pass B claim extraction workflows.

- ⚠️ Development Status
    - This is early CaptionForge Pass A Lite wrapper infrastructure.
    - The Lite interface is intentionally minimal and may remain smaller than
      the full node even as CaptionForge gains additional refinement passes.
    - The node is intended for local interactive captioning, not full dataset
      audit orchestration.

- Attribution & License
  - Concept and implementation by **J. L. Córdova**
    with development assistance from **ChatGPT (OpenAI)**.

  - JoyCaption/LLaVA model loading is designed around compatible Hugging Face
    Transformers interfaces and publicly available JoyCaption-family checkpoints.

  - Designed for use with:
    https://github.com/comfyanonymous/ComfyUI

  - Copyright (c) 2026 J. L. Córdova

  - Released under the **MIT License**.
"""

from __future__ import annotations

MANIFEST = {
    "name": "JLC Joy Caption (Lite)",
    "version": (1, 0, 1),
    "author": "J. L. Córdova",
    "description": (
        "Minimal direct-IMAGE ComfyUI frontend for JoyCaption/LLaVA-family captioning "
        "inside CaptionForge Pass A. Provides a compact daily-use interface with model "
        "selection, memory-mode selection, keep-loaded cache control, a single caption "
        "instruction field, conservative system prompt routing, basic generation controls, "
        "direct IMAGE tensor handling, and caption string output. Delegates reusable "
        "Joy/LLaVA model loading, generation, 8-bit memory-efficient behavior, cleanup, "
        "ComfyUI model-management bridge behavior, and shared cache integration to "
        "jlc_joy_caption_engine.py while preserving a small practical interface for "
        "interactive captioning workflows."
    ),
}

from pathlib import Path

import numpy as np
import torch
from PIL import Image

import folder_paths

try:
    from .jlc_joy_caption_engine import (
        CleanupConfig,
        GenerationConfig,
        JoyCaptionConfig,
        JoyCaptionEngine,
        MEMORY_EFFICIENT_CONFIGS,
        MODEL_REGISTRY,
    )
except ImportError:
    from jlc_joy_caption_engine import (
        CleanupConfig,
        GenerationConfig,
        JoyCaptionConfig,
        JoyCaptionEngine,
        MEMORY_EFFICIENT_CONFIGS,
        MODEL_REGISTRY,
    )

JLC_JOY_MODEL_ROOT = Path(folder_paths.models_dir) / "LLM" / "JLC_JoyCaption"

DEFAULT_LITE_PROMPT = (
    "You are an image captioning assistant. Describe the image in a highly "
    "detailed, literal, and visually grounded way. Focus only on visible details. "
    "Include subject appearance, clothing, pose, body position, hands, facial "
    "expression, hairstyle, accessories, lighting, background, textures, colors, "
    "and spatial relationships. Avoid speculation, avoid backstory, avoid "
    "opinions, and avoid mentioning things not clearly visible."
)


LITE_SYSTEM_PROMPT = (
    "You are a helpful image-captioning assistant. Describe only what is visible "
    "in the image. Do not invent unseen context."
)


def _tensor_to_pil(image_tensor) -> list[Image.Image]:
    """Convert ComfyUI IMAGE tensor [B,H,W,C] float 0..1 into PIL RGB images."""
    if image_tensor is None:
        return []

    if isinstance(image_tensor, torch.Tensor):
        image_tensor = image_tensor.detach().cpu()

    if image_tensor.ndim == 3:
        image_tensor = image_tensor.unsqueeze(0)

    images: list[Image.Image] = []
    for img in image_tensor:
        arr = img.numpy()
        arr = np.clip(arr * 255.0, 0, 255).astype(np.uint8)
        images.append(Image.fromarray(arr).convert("RGB"))

    return images


class JLC_JoyCaptionLite:
    """Minimal direct-image JoyCaption node for everyday CaptionForge use."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": (
                    list(MODEL_REGISTRY.keys()),
                    {
                        "default": "llama-joycaption-beta-one-hf-llava",
                        "tooltip": (
                            "Select the JoyCaption/LLaVA-family model. Models are "
                            "loaded from ComfyUI/models/LLM/JLC_JoyCaption/."
                        ),
                    },
                ),
                "memory_mode": (
                    list(MEMORY_EFFICIENT_CONFIGS.keys()),
                    {
                        "default": (
                            "Balanced (8-bit)"
                            if "Balanced (8-bit)" in MEMORY_EFFICIENT_CONFIGS
                            else list(MEMORY_EFFICIENT_CONFIGS.keys())[0]
                        ),
                        "tooltip": (
                            "Joy model memory mode. Balanced (8-bit) uses bitsandbytes "
                            "load-time quantization and is the recommended CaptionForge "
                            "default for 16 GB VRAM systems."
                        ),
                    },
                ),
                "keep_loaded": (
                    "BOOLEAN",
                    {
                        "default": True,
                        "tooltip": (
                            "Keep the model cached after captioning for faster repeated "
                            "runs. Disable to unload after this node finishes."
                        ),
                    },
                ),
                "system_prompt": (
                    "STRING",
                    {
                        "default": DEFAULT_LITE_PROMPT,
                        "multiline": True,
                        "tooltip": (
                            "Lite caption instruction. This is passed as the Joy caption "
                            "prompt while a conservative internal system message is used."
                        ),
                    },
                ),
                "max_new_tokens": (
                    "INT",
                    {
                        "default": 384,
                        "min": 16,
                        "max": 4096,
                        "step": 8,
                        "tooltip": "Maximum number of new tokens to generate for each caption.",
                    },
                ),
                "temperature": (
                    "FLOAT",
                    {
                        "default": 0.75,
                        "min": 0.0,
                        "max": 2.0,
                        "step": 0.01,
                        "tooltip": "Sampling temperature. Use 0.0 for greedy/non-sampling output.",
                    },
                ),
                "top_p": (
                    "FLOAT",
                    {
                        "default": 0.90,
                        "min": 0.0,
                        "max": 1.0,
                        "step": 0.01,
                        "tooltip": "Nucleus sampling threshold.",
                    },
                ),
                "top_k": (
                    "INT",
                    {
                        "default": 50,
                        "min": 0,
                        "max": 500,
                        "step": 1,
                        "tooltip": "Top-k sampling limit. Set to 0 to disable top-k filtering.",
                    },
                ),
            },
            "optional": {
                "image": (
                    "IMAGE",
                    {
                        "tooltip": "Image or batch of images to caption.",
                    },
                ),
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("caption",)
    FUNCTION = "caption"
    CATEGORY = "JLC/Captioning"

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        return float("NaN")

    def caption(
        self,
        model,
        memory_mode,
        keep_loaded,
        system_prompt,
        max_new_tokens,
        temperature,
        top_p,
        top_k,
        image=None,
    ):
        pil_images = _tensor_to_pil(image)
        if not pil_images:
            raise RuntimeError("No IMAGE input found. Connect an image to JLC Joy Caption (Lite).")

        prompt = (system_prompt or DEFAULT_LITE_PROMPT).strip() or DEFAULT_LITE_PROMPT

        generation = GenerationConfig(
            max_new_tokens=int(max_new_tokens),
            temperature=float(temperature),
            top_p=float(top_p),
            top_k=int(top_k),
            repetition_penalty=1.0,
            seed=None,
        )

        cleanup = CleanupConfig(
            trigger="",
            prefix="",
            suffix="",
            forbidden_phrases=[],
            replacement_rules=[],
        )

        joy_config = JoyCaptionConfig(
            model_name=model,
            model_path="",
            model_root=str(JLC_JOY_MODEL_ROOT),
            memory_mode=memory_mode,
            dtype="bf16",
            device="auto",
            trust_remote_code=True,
            keep_loaded=bool(keep_loaded),
            quiet_transformers_load=True,
            system_prompt=LITE_SYSTEM_PROMPT,
            prompt=prompt,
            allow_download=True,
            use_comfy_model_management=True,
        )

        engine = JoyCaptionEngine(
            config=joy_config,
            generation=generation,
            cleanup=cleanup,
        )

        engine.load()
        captions: list[str] = []
        for index, pil in enumerate(pil_images):
            final_caption, _raw_caption = engine.caption_pil(pil)
            captions.append(final_caption)
            print(f"[JLC Joy Caption Lite] Captioned IMAGE {index + 1}/{len(pil_images)}")

        if not keep_loaded:
            engine.unload()

        return ("\n\n".join(captions),)


NODE_CLASS_MAPPINGS = {
    "JLC_JoyCaptionLite": JLC_JoyCaptionLite,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "JLC_JoyCaptionLite": "\u2003JLC Joy Caption (Lite)",
}
