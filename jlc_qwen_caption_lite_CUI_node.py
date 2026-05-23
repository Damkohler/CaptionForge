"""
JLC Qwen Caption (Lite) — ComfyUI Node Wrapper

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
    - The **JLC Qwen Caption (Lite)** node provides a deliberately small
      daily-use ComfyUI frontend for Qwen-family vision-language captioning.

    - It is intended for quick direct IMAGE captioning when the full CaptionForge
      Qwen node is more detailed than needed.

    - This file is the **ComfyUI-facing Lite wrapper**, not the reusable
      captioning engine. It is responsible for:
            • minimal ComfyUI INPUT_TYPES / widget definitions
            • direct IMAGE tensor input
            • IMAGE tensor conversion to PIL images
            • fixed model-root integration under `models/LLM/`
            • basic generation controls
            • optional bitsandbytes 8-bit loading control
            • keep-loaded cache control
            • direct caption string output
            • node display name, category, and mapping registration
            • passing user settings into the shared Qwen caption engine

    - The actual reusable captioning implementation lives in:
            jlc_qwen_caption_engine.py

- CaptionForge Pass Role
    - This node participates in **Pass A** of the CaptionForge pipeline.

    - Unlike the full Qwen node, this Lite node is optimized for quick
      direct-image captioning and does not expose the full file/folder,
      sidecar, JSONL, run-config, prompt-file, or cleanup interface.

    - Its captions can still be used as caption evidence, but the Lite node is
      primarily designed for fast interactive use inside ComfyUI workflows.

- Lite Workflow Model
    - The node accepts:
            • direct ComfyUI IMAGE input only

    - The node returns:
            • one caption string containing captions for the input image batch

    - The Lite interface exposes only:
            • model
            • Qwen quantization mode
            • keep_loaded
            • caption instruction text
            • max_new_tokens
            • temperature
            • top_p
            • top_k

- Model and Memory Behavior
    - Qwen models are loaded from:
            ComfyUI/models/LLM/JLC_QwenCaption/

    - The Lite node uses the shared CaptionForge model cache through the Qwen
      engine so repeated runs can reuse a loaded model when `keep_loaded` is
      enabled.

    - Optional **Balanced (8-bit)** mode maps to bitsandbytes 8-bit loading in
      the shared engine and is intended mainly for larger Qwen-family models on
      limited-VRAM systems.

- Design Philosophy
    - The Lite node exists for everyday captioning convenience.

    - It preserves the same engine-democratic CaptionForge architecture as the
      full node while avoiding unnecessary UI surface area for simple direct
      IMAGE captioning tasks.

    - The full Qwen node remains the preferred interface for batch dataset work,
      JSONL audit output, TXT sidecars, run-config logging, and downstream Pass B
      claim extraction workflows.

- ⚠️ Development Status
    - This is early CaptionForge Pass A Lite wrapper infrastructure.
    - The Lite interface is intentionally minimal and may remain smaller than
      the full node even as CaptionForge gains additional refinement passes.
    - The node is intended for local interactive captioning, not full dataset
      audit orchestration.

- Attribution & License
  - Concept and implementation by **J. L. Córdova**
    with development assistance from **ChatGPT (OpenAI)**.

  - Qwen-family model loading is designed around compatible Hugging Face
    Transformers interfaces and publicly available Qwen-family checkpoints.

  - Designed for use with:
    https://github.com/comfyanonymous/ComfyUI

  - Copyright (c) 2026 J. L. Córdova

  - Released under the **MIT License**.
"""

from __future__ import annotations

MANIFEST = {
    "name": "JLC Qwen Caption (Lite)",
    "version": (1, 0, 0),
    "author": "J. L. Córdova",
    "description": (
        "Minimal direct-IMAGE ComfyUI frontend for Qwen-family vision-language captioning "
        "inside CaptionForge Pass A. Provides a compact daily-use interface with model "
        "selection, optional bitsandbytes 8-bit loading, keep-loaded cache control, a single "
        "caption instruction field, basic generation controls, direct IMAGE tensor handling, "
        "and caption string output. Delegates reusable Qwen model loading, generation, "
        "quantization behavior, cleanup, and shared cache integration to "
        "jlc_qwen_caption_engine.py while preserving a small practical interface for "
        "interactive captioning workflows."
    ),
}

from pathlib import Path

import numpy as np
import torch
from PIL import Image

import folder_paths

try:
    from .jlc_qwen_caption_engine import (
        CleanupConfig,
        GenerationConfig,
        MODEL_REGISTRY,
        QwenCaptionConfig,
        QwenCaptionEngine,
    )
except ImportError:
    from jlc_qwen_caption_engine import (
        CleanupConfig,
        GenerationConfig,
        MODEL_REGISTRY,
        QwenCaptionConfig,
        QwenCaptionEngine,
    )


JLC_QWEN_MODEL_ROOT = Path(folder_paths.models_dir) / "LLM" / "JLC_QwenCaption"

DEFAULT_LITE_PROMPT = (
    "You are an image captioning assistant. Describe the image in a highly "
    "detailed, literal, and visually grounded way. Focus only on visible details. "
    "Include subject appearance, clothing, pose, body position, hands, facial "
    "expression, hairstyle, accessories, lighting, background, textures, colors, "
    "and spatial relationships. Write a dense descriptive prompt suitable for "
    "image captioning. Avoid speculation, avoid backstory, avoid opinions, and "
    "avoid mentioning things not clearly visible."
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


class JLC_QwenCaptionLite:
    """Minimal direct-image Qwen VLM node for everyday CaptionForge use."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": (
                    list(MODEL_REGISTRY.keys()),
                    {
                        "default": "Qwen2.5-VL-3B-Instruct",
                        "tooltip": (
                            "Select the Qwen vision-language model. Models are "
                            "loaded from ComfyUI/models/LLM/JLC_QwenCaption/."
                        ),
                    },
                ),
                "qwen_quantization": (
                    ["none", "Balanced (8-bit)"],
                    {
                        "default": "none",
                        "tooltip": (
                            "Optional bitsandbytes 8-bit loading. Intended mainly for "
                            "Qwen 7B-class models on limited VRAM. May also work on other "
                            "compatible Qwen-family models, but compatibility and speed are "
                            "environment-dependent."
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
                            "Lite caption instruction. For Qwen this is passed as the "
                            "caption prompt consumed by the shared engine."
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
        qwen_quantization,
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
            raise RuntimeError("No IMAGE input found. Connect an image to JLC Qwen Caption (Lite).")

        prompt = (system_prompt or DEFAULT_LITE_PROMPT).strip() or DEFAULT_LITE_PROMPT

        generation = GenerationConfig(
            max_new_tokens=int(max_new_tokens),
            temperature=float(temperature),
            top_p=float(top_p),
            top_k=int(top_k),
            repetition_penalty=1.08,
            seed=None,
        )

        cleanup = CleanupConfig(
            trigger="",
            prefix="",
            suffix="",
            forbidden_phrases=[],
            replacement_rules=[],
        )

        qwen_quantization_value = {
            "None": "none",
            "Balanced (8-bit)": "bnb_8bit",
        }.get(str(qwen_quantization), "none")

        qwen_config = QwenCaptionConfig(
            model_name=model,
            model_path="",
            model_root=str(JLC_QWEN_MODEL_ROOT),
            dtype="auto",
            device="auto",
            device_map="auto",
            quantization=qwen_quantization_value,
            trust_remote_code=True,
            keep_loaded=bool(keep_loaded),
            quiet_transformers_load=True,
            patch_lm_head_weight=True,
            ignore_mismatched_sizes=True,
            max_size=1024,
            prompt=prompt,
            allow_download=True,
        )

        engine = QwenCaptionEngine(
            config=qwen_config,
            generation=generation,
            cleanup=cleanup,
        )

        engine.load()
        captions: list[str] = []
        for index, pil in enumerate(pil_images):
            final_caption, _raw_caption = engine.caption_pil(pil)
            captions.append(final_caption)
            print(f"[JLC Qwen Caption Lite] Captioned IMAGE {index + 1}/{len(pil_images)}")

        if not keep_loaded:
            engine.unload()

        return ("\n\n".join(captions),)


NODE_CLASS_MAPPINGS = {
    "JLC_QwenCaptionLite": JLC_QwenCaptionLite,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "JLC_QwenCaptionLite": "\u2003JLC Qwen Caption (Lite)",
}
