"""
JLC CaptionForge Pipeline Planner — ComfyUI Node Wrapper

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
    - The **JLC CaptionForge Pipeline Planner** node provides the shared
      configuration object for coordinated CaptionForge captioning workflows.

    - This file is the **ComfyUI-facing wrapper**, not the reusable planning
      engine. It is responsible for:
            • ComfyUI INPUT_TYPES / widget definitions
            • shared project/run output-folder selection
            • optional shared file/folder `input_path` routing
            • recursive and filename-glob dataset routing controls
            • raw-caption witness count selection for caption model families
            • optional validator selection
            • optional final-caption LLM polish selection
            • shared seed and sampling schedules
            • shared image-size and token-budget guards
            • shared LoRA trigger-word routing
            • emitting a CAPTIONFORGE_PIPELINE_PLAN object
            • returning a readable JSON representation of the emitted plan
            • node display name, category, and mapping registration
            • passing user settings into the shared pipeline planner engine

    - The actual reusable planning implementation lives in:
            captionforge_pipeline_planner_engine.py

      That engine handles:
            • pipeline-plan normalization
            • typed CAPTIONFORGE_PIPELINE_PLAN dictionary creation
            • schedule parsing
            • seed expansion
            • per-caption-instance run expansion
            • model-family run-count lookup
            • shared routing resolution
            • shared max-size and token-budget resolution
            • optional-model enable/disable normalization
            • temporary compatibility aliases during CaptionForge development

- CaptionForge Pass Role
    - This node coordinates the shared CaptionForge pipeline plan used by
      Pass A caption witnesses and downstream CaptionForge processing.

    - Raw caption witnesses generate auditable caption evidence records from one
      or more captioning engines.

    - The planner defines which caption model families participate in raw
      caption generation and how many caption witnesses each family contributes
      per image.

    - Current raw-caption family controls include:
            • Joy Caption
            • Qwen Caption
            • Florence
            • Llama Vision

    - Joy and Qwen are treated as foundational raw-caption witnesses. Optional
      additional model families may be enabled as CaptionForge expands.

    - The planner may also declare future optional validation and final-polish
      stages, while leaving exact model-version and memory-mode choices to the
      individual captioning nodes.

- Node Workflow Model
    - The planner emits a single CAPTIONFORGE_PIPELINE_PLAN object.

    - Captioning nodes consume this object to coordinate:
            • shared output directory
            • optional shared input path
            • recursive folder traversal
            • filename filtering
            • per-family raw-caption run counts
            • per-caption-instance seed values
            • per-caption-instance sampling values
            • shared max image size
            • shared max token budget
            • shared LoRA trigger word

    - The planner does not caption images, load models, validate masks, extract
      claims, or synthesize final captions.

    - The planner is intentionally model-family oriented. Exact model versions,
      quantization modes, memory modes, prompts, and engine-specific settings
      remain on the individual captioning nodes.

- User-Facing Pipeline Language
    - The node avoids exposing internal pass labels such as Pass A, Pass B, or
      Pass C in its widget labels where possible.

    - User-facing planner sections are expressed in practical workflow terms:
            • Routing
            • Raw Captions
            • Validation
            • Final Caption
            • Sampling
            • LoRA

    - Internal plan fields may still use pass-oriented keys where useful for
      downstream compatibility and code organization.

- Final Caption Policy
    - Deterministic CaptionForge synthesis is mandatory in the downstream final
      caption stage.

    - The planner does not expose a deterministic on/off switch.

    - Optional LLM polish, such as GPT-OSS polish, is treated as a post-process
      after deterministic synthesis rather than a replacement for deterministic
      claim-grounded caption construction.

- Design Philosophy
    - This node keeps CaptionForge pipeline coordination separate from individual
      caption model implementation.

    - CaptionForge is engine-democratic: no single caption model is treated as
      canonical. The planner coordinates multiple caption witnesses so downstream
      claim extraction, consensus, validation, and final-caption synthesis can
      work from auditable evidence rather than one model's unverified output.

    - The planner prioritizes clear workflow configuration, reproducibility,
      auditability, scalable hardware usage, and clean separation between the
      ComfyUI interface and shared planner logic.

- ⚠️ Development Status
    - This is early CaptionForge pipeline-planning infrastructure.
    - The UI, model-family list, validator options, polish options, emitted plan
      schema, and compatibility aliases may evolve as the multi-pass
      CaptionForge pipeline matures.
    - The node is intended for local dataset preparation and controlled caption
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
    "name": "JLC CaptionForge Pipeline Planner",
    "version": (0, 3, 0),
    "author": "J. L. Córdova",
    "description": (
        "ComfyUI-facing planner node for coordinated CaptionForge captioning workflows. "
        "Defines shared routing, raw-caption witness counts by model family, optional "
        "validator and final-polish selections, seed and sampling schedules, image-size "
        "and token-budget guards, and LoRA trigger-word routing. Emits a "
        "CAPTIONFORGE_PIPELINE_PLAN object plus a readable JSON representation. Delegates "
        "plan normalization, schedule parsing, seed expansion, model-family run-count "
        "resolution, and per-capINPUT_TYPEStion-instance run expansion to "
        "captionforge_pipeline_planner_engine.py so Joy, Qwen, Florence, Llama Vision, "
        "and future CaptionForge nodes can share one auditable pipeline plan."
    ),
}

import json

from ..engines.captionforge_pipeline_planner_engine import (
    MAX_SEED_32,
    build_captionforge_pipeline_plan,
    default_semantic_profile,
    discover_semantic_profiles,
)


RUN_COUNT_REQUIRED = ["1", "2", "3", "4", "5"]
RUN_COUNT_OPTIONAL = ["disabled", "1", "2", "3", "4", "5"]

class JLC_CaptionForgePipelinePlanner:
    @classmethod
    def INPUT_TYPES(cls):
        profiles = discover_semantic_profiles()
        if not profiles:
            profiles = ["disabled"]

        default_profile = default_semantic_profile()
        if default_profile not in profiles:
            default_profile = profiles[0]

        return {
            "required": {
                "Routing - output folder": (
                    "STRING",
                    {
                        "default": "",
                        "multiline": False,
                        "tooltip": "Required CaptionForge project/run folder.",
                    },
                ),
                "Routing - input path": (
                    "STRING",
                    {
                        "default": "",
                        "multiline": False,
                        "tooltip": "Optional shared input image file or folder.",
                    },
                ),
                "Routing - recursive": (
                    "BOOLEAN",
                    {
                        "default": True,
                        "tooltip": "If input path is a folder, search subfolders too.",
                    },
                ),
                "Routing - filename glob": (
                    "STRING",
                    {
                        "default": "*",
                        "multiline": False,
                        "tooltip": "Filename filter for folder input path, such as *.png or *.jpg.",
                    },
                ),

                "Raw Captions - Joy": (
                    ["1", "2", "3", "4", "5"],
                    {
                        "default": "2",
                        "tooltip": "Number of Joy Caption raw caption witnesses per image.",
                    },
                ),
                "Raw Captions - Qwen": (
                    ["1", "2", "3", "4", "5"],
                    {
                        "default": "1",
                        "tooltip": "Number of Qwen raw caption witnesses per image.",
                    },
                ),
                "Raw Captions - Florence": (
                    ["disabled", "1", "2", "3", "4", "5"],
                    {
                        "default": "disabled",
                        "tooltip": "Optional Florence raw caption witnesses per image.",
                    },
                ),
                "Raw Captions - Llama Vision": (
                    ["disabled", "1", "2", "3", "4", "5"],
                    {
                        "default": "disabled",
                        "tooltip": "Optional Llama Vision raw caption witnesses per image.",
                    },
                ),

                "Validation - SAM3": (
                    ["disabled", "sam3"],
                    {
                        "default": "disabled",
                        "tooltip": "Optional future validator. This does not write captions.",
                    },
                ),

                "Final Caption - LLM polish": (
                    ["disabled", "gpt_oss"],
                    {
                        "default": "disabled",
                        "tooltip": "Optional LLM polish after deterministic CaptionForge synthesis.",
                    },
                ),

                "Final Caption - semantic profile": (
                    profiles,
                    {
                        "default": default_profile,
                        "tooltip": (
                            "Semantic profile used by the final CaptionForge synthesis stage. "
                            "Profiles are discovered from CaptionForge/semantic_profiles/. "
                            "Add custom profiles there to make them appear in this dropdown."
                        ),
                    },
                ),

                "Sampling - base seed": (
                    "INT",
                    {
                        "default": 0,
                        "min": -1,
                        "max": MAX_SEED_32,
                        "step": 1,
                        "tooltip": "Base seed for the raw-caption ensemble.",
                    },
                ),
                "Sampling - seed mode": (
                    ["fixed", "increment", "decrement", "random"],
                    {
                        "default": "increment",
                        "tooltip": "How per-caption seeds are derived.",
                    },
                ),
                "Sampling - temperature schedule": (
                    "STRING",
                    {
                        "default": "0.35, 0.60, 0.85",
                        "multiline": False,
                        "tooltip": "Comma-separated temperatures; short schedules repeat the final value.",
                    },
                ),
                "Sampling - top-p schedule": (
                    "STRING",
                    {
                        "default": "0.90",
                        "multiline": False,
                        "tooltip": "Comma-separated top-p values.",
                    },
                ),
                "Sampling - top-k schedule": (
                    "STRING",
                    {
                        "default": "50",
                        "multiline": False,
                        "tooltip": "Comma-separated top-k values.",
                    },
                ),
                "Sampling - max image size": (
                    "INT",
                    {
                        "default": 1024,
                        "min": 0,
                        "max": 4096,
                        "step": 64,
                        "tooltip": "Shared image resize guard. Source files are not modified.",
                    },
                ),
                "Sampling - max new tokens": (
                    "INT",
                    {
                        "default": 512,
                        "min": 16,
                        "max": 4096,
                        "step": 8,
                        "tooltip": "Shared token budget for each raw caption.",
                    },
                ),

                "LoRA - trigger word": (
                    "STRING",
                    {
                        "default": "",
                        "multiline": False,
                        "tooltip": "Optional shared LoRA trigger token inserted into evidence captions.",
                    },
                ),
            }
        }

    RETURN_TYPES = ("CAPTIONFORGE_PIPELINE_PLAN", "STRING")
    RETURN_NAMES = ("pipeline_plan", "pipeline_plan_json")
    FUNCTION = "build"
    CATEGORY = "JLC/Captioning"

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        return float("NaN")

    def build(self, **kwargs):
        output_dir = str(kwargs.get("Routing - output folder", "") or "").strip()
        if not output_dir:
            raise RuntimeError(
                "CaptionForge Pipeline Planner requires a Routing - output folder."
            )

        cfg = build_captionforge_pipeline_plan(
            output_dir=output_dir,
            input_path=kwargs.get("Routing - input path", ""),
            recursive=kwargs.get("Routing - recursive", True),
            filename_glob=kwargs.get("Routing - filename glob", "*"),

            joy_runs_per_image=kwargs.get("Raw Captions - Joy", "2"),
            qwen_runs_per_image=kwargs.get("Raw Captions - Qwen", "1"),
            florence_runs_per_image=kwargs.get("Raw Captions - Florence", "disabled"),
            llama_vision_runs_per_image=kwargs.get("Raw Captions - Llama Vision", "disabled"),

            validator_model=kwargs.get("Validation - VLM inspection", "disabled"),

            pass_c_polish_model=kwargs.get("Final Caption - LLM polish", "disabled"),
            semantic_profile=kwargs.get("Final Caption - semantic profile", default_semantic_profile()),

            base_seed=kwargs.get("Sampling - base seed", 0),
            seed_mode=kwargs.get("Sampling - seed mode", "increment"),
            temperature_schedule=kwargs.get("Sampling - temperature schedule", "0.35, 0.60, 0.85"),
            top_p_schedule=kwargs.get("Sampling - top-p schedule", "0.90"),
            top_k_schedule=kwargs.get("Sampling - top-k schedule", "50"),
            max_size=kwargs.get("Sampling - max image size", 1024),
            max_new_tokens=kwargs.get("Sampling - max new tokens", 512),

            trigger_word=kwargs.get("LoRA - trigger word", ""),
        )

        return (cfg, json.dumps(cfg, ensure_ascii=False, indent=2))


# Compatibility alias for older internal references. Do not map it as a second node.
JLC_CaptionForgeRunPlanner = JLC_CaptionForgePipelinePlanner
JLC_CaptionForgeRunPlan = JLC_CaptionForgePipelinePlanner


NODE_CLASS_MAPPINGS = {
    "JLC_CaptionForgePipelinePlanner": JLC_CaptionForgePipelinePlanner,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "JLC_CaptionForgePipelinePlanner": "\u2003JLC CaptionForge Pipeline Planner",
}
