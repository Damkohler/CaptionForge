"""
JLC CaptionForge Run Plan — ComfyUI Pass A Coordination Node

- CaptionForge
  - This node is part of **CaptionForge**, a model-agnostic captioning
    framework for ComfyUI developed by **J. L. Córdova**.

  - Repository
    https://github.com/Damkohler/CaptionForge

  - CaptionForge focuses on practical dataset-captioning infrastructure for:
        • LoRA dataset preparation
        • multi-engine caption generation
        • JSONL evidence trails
        • claim extraction and refinement
        • consensus-oriented caption improvement
        • final human-grade sidecar caption synthesis

- Node Purpose
    - The **JLC CaptionForge Run Plan** node emits a shared Pass A run
      configuration object for Qwen, Joy, and future CaptionForge captioning
      engines.

    - It does not caption images directly.

    - It does not run claim extraction.

    - It does not produce final user-facing sidecar captions.

    - Instead, it coordinates upstream caption evidence generation by defining:
            • shared project/run output folder
            • captions per image
            • per-caption seed behavior
            • sampling schedules
            • shared max image size
            • shared max token budget
            • shared LoRA trigger word

- CaptionForge Pass Role
    - This node coordinates **Pass A** of the CaptionForge pipeline.

    - Pass A generates auditable caption evidence records from one or more
      captioning engines.

    - The JSONL evidence produced under this Run Plan is intended to be consumed
      by downstream CaptionForge stages, including:
            • claim extraction
            • semantic profile filtering
            • caption evidence comparison
            • final caption synthesis

- Run Plan Contract
    - The node outputs:
            • CAPTIONFORGE_RUN_CONFIG
            • a human-readable JSON string copy of the same config

    - Caption nodes connected to this Run Plan should treat it as authoritative
      for CaptionForge-mode execution.

    - In CaptionForge mode, connected caption nodes should override local
      standalone-node settings where necessary so that evidence generation is
      coordinated and reproducible.

    - In particular, CaptionForge-mode caption nodes should force:
            • write_jsonl = True
            • also_jsonl = False
            • skip_existing_txt = False
            • skip_existing_jsonl_images = False
            • overwrite = True

      This prevents TXT sidecars or stale JSONL skip settings from silently
      suppressing required evidence records.

- Output Directory Philosophy
    - output_dir is required in Run Plan mode.

    - Standalone Qwen/Joy caption nodes may use their own fallback folders, but
      a CaptionForge evidence run must write to one shared project/run folder.

    - This allows downstream nodes to consume one canonical evidence pool rather
      than searching multiple engine-specific output folders.

- Seed Philosophy
    - The Run Plan emits one shared base seed and seed mode.

    - Caption nodes expand the plan into per-caption-instance seeds.

    - With seed_mode="increment", for example:
            run 0 -> base_seed
            run 1 -> base_seed + 1
            run 2 -> base_seed + 2

    - Qwen and Joy intentionally use matching seeds for matching ensemble run
      indices. This creates paired evidence records across engines.

- Schedule Philosophy
    - temperature_schedule, top_p_schedule, and top_k_schedule are comma-separated
      lists.

    - If a schedule is shorter than captions_per_image, the last value repeats.

    - This allows users to vary only one parameter while keeping others fixed.

- Design Philosophy
    - This is not a mega-node.

    - It is a compact Pass A coordination node that lets independent captioning
      nodes behave as a coherent CaptionForge evidence ensemble.

    - The user-facing captioning nodes remain useful as standalone tools.

- ⚠️ Development Status
    - This is early CaptionForge Run Plan infrastructure.
    - The config schema may evolve before CaptionForge v1.0.0.
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
    "name": "JLC CaptionForge Run Plan",
    "version": (0, 1, 0),
    "author": "J. L. Córdova",
    "description": (
        "Optional CaptionForge Pass A coordination node. Emits a shared "
        "CAPTIONFORGE_RUN_CONFIG object for Qwen, Joy, and future captioning "
        "engines. Defines one required project/run output folder, captions per "
        "image, per-caption-instance seed behavior, sampling schedules, shared "
        "max image size, shared max generation token budget, and a shared LoRA "
        "trigger word. Connected caption nodes should treat the Run Plan as "
        "authoritative in CaptionForge mode, forcing JSONL evidence output and "
        "disabling TXT/JSONL skip behavior that could suppress required evidence "
        "records. The node does not caption images or synthesize final sidecars; "
        "it coordinates reproducible multi-engine evidence generation for "
        "downstream claim extraction and final CaptionForge caption synthesis."
    ),
}

import json

from ..engines.run_plan.captionforge_run_plan import (
    build_captionforge_run_config,
    MAX_SEED_32,
)

class JLC_CaptionForgeRunPlan:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "output_dir": (
                    "STRING",
                    {
                        "default": "",
                        "multiline": False,
                        "tooltip": (
                            "Required in CaptionForge Run Plan mode. Choose one project/run "
                            "folder where all connected caption nodes write shared JSONL "
                            "evidence, TXT audit sidecars, and run configs. Downstream claim "
                            "extraction and final CaptionForge synthesis should read from this "
                            "same folder."
                        )
                    },
                ),
                "captions_per_image": (
                    "INT",
                    {
                        "default": 3,
                        "min": 1,
                        "max": 100,
                        "step": 1,
                        "tooltip": (
                            "Number of raw caption records to generate per image, per connected "
                            "captioning engine. For example, 3 with Qwen + Joy yields up to "
                            "6 first pass raw caption evidence records per image."
                        ),
                    },
                ),

                "base_seed": (
                    "INT",
                    {
                        "default": 0,
                        "min": -1,
                        "max": MAX_SEED_32,
                        "step": 1,
                        "tooltip": (
                            "Base seed for the CaptionForge ensemble. Seeds are expanded per "
                            "caption instance according to seed_mode. Use -1 for nondeterministic "
                            "behavior."
                        ),
                    },
                ),

                "seed_mode": (
                    ["fixed", "increment", "decrement", "random"],
                    {
                        "default": "increment",
                        "tooltip": (
                            "How each per-caption-instance seed is derived. fixed reuses "
                            "base_seed for all caption instances. increment uses base_seed, "
                            "base_seed+1, base_seed+2, etc. decrement counts down. random derives "
                            "a pseudo-random sequence from base_seed, or nondeterministic seeds "
                            "when base_seed is -1."
                        ),
                    },
                ),

                "temperature_schedule": (
                    "STRING",
                    {
                        "default": "0.35, 0.60, 0.85",
                        "multiline": False,
                        "tooltip": (
                            "Comma-separated temperatures applied per caption instance. If the "
                            "list is shorter than captions_per_image, the final value repeats. "
                            "Example: with 3 captions, '0.35, 0.75' means 0.35, 0.75, 0.75."
                        ),
                    },
                ),

                "top_p_schedule": (
                    "STRING",
                    {
                        "default": "0.90",
                        "multiline": False,
                        "tooltip": (
                            "Comma-separated top-p values applied per caption instance. If the "
                            "list is shorter than captions_per_image, the final value repeats. "
                            "Use a single value to keep top-p constant while varying temperature."
                        ),
                    },
                ),

                "top_k_schedule": (
                    "STRING",
                    {
                        "default": "50",
                        "multiline": False,
                         "tooltip": (
                            "Comma-separated top-k values applied per caption instance. If the "
                            "list is shorter than captions_per_image, the final value repeats. "
                            "Use a single value to keep top-k constant across the ensemble."
                        ),
                    },
                ),

                "max_size": (
                    "INT",
                    {
                        "default": 1024,
                        "min": 0,
                        "max": 4096,
                        "step": 64,
                        "tooltip": (
                            "Shared image workload guard for all connected caption nodes. The "
                            "longest image side is resized in memory only; source files are not "
                            "modified. Prevents one captioner from accidentally processing huge "
                            "mixed-in images. Use 0 to disable resizing."
                        ),
                    },
                ),

                "max_new_tokens": (
                    "INT",
                    {
                        "default": 512,
                        "min": 16,
                        "max": 4096,
                        "step": 8,
                        "tooltip": (
                            "Shared constant generation token budget for each caption. Kept "
                            "constant across the ensemble so downstream claim extraction is not "
                            "biased by one caption being allowed to ramble more than another. "
                            "Increase only if captions are visibly truncated."
                        ),
                    },
                ),

                "trigger_word": (
                    "STRING",
                    {
                        "default": "",
                        "multiline": False,
                        "tooltip": (
                            "Shared LoRA training trigger token, such as JessJennFlux2 or "
                            "DolLoraSugarSkulls. Connected caption nodes inject this as the "
                            "common prefix for all evidence captions in the run."
                        ),
                    },
                ),
            },
        }

    RETURN_TYPES = ("CAPTIONFORGE_RUN_CONFIG", "STRING")
    RETURN_NAMES = ("run_config", "run_config_json")
    FUNCTION = "build"
    CATEGORY = "JLC/Captioning"

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        return float("NaN")

    def build(
        self,
        output_dir,
        captions_per_image,
        base_seed,
        seed_mode,
        temperature_schedule,
        top_p_schedule,
        top_k_schedule,
        max_size,
        max_new_tokens,
        trigger_word,
    ):
        output_dir = str(output_dir or "").strip()
        if not output_dir:
            raise RuntimeError(
                "CaptionForge Run Plan requires output_dir. "
                "Choose a project/run folder so Qwen, Joy, claim extraction, "
                "and final CaptionForge synthesis share one evidence pool."
            )
    
        cfg = build_captionforge_run_config(
            captions_per_image=captions_per_image,
            base_seed=base_seed,
            seed_mode=seed_mode,
            temperature_schedule=temperature_schedule,
            top_p_schedule=top_p_schedule,
            top_k_schedule=top_k_schedule,
            max_size=max_size,
            max_new_tokens=max_new_tokens,
            trigger_word=trigger_word,
            output_dir=output_dir,
        )

        return (cfg, json.dumps(cfg, ensure_ascii=False, indent=2))


NODE_CLASS_MAPPINGS = {
    "JLC_CaptionForgeRunPlan": JLC_CaptionForgeRunPlan,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "JLC_CaptionForgeRunPlan": "\u2003JLC CaptionForge Run Plan",
}