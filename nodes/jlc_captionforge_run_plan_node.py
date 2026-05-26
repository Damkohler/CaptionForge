"""
JLC CaptionForge Run Plan — ComfyUI Pass A Coordination Node

This node emits a shared CAPTIONFORGE_RUN_CONFIG object for Qwen, Joy, Lite,
and future CaptionForge captioning nodes.

It does not caption images. It coordinates Pass A evidence generation:
- shared project/run output folder
- optional shared input file/folder path
- recursive and filename-glob dataset routing
- captions per image
- per-caption-instance seed behavior
- sampling schedules
- shared max image size
- shared max token budget
- shared LoRA trigger word

When connected, caption nodes should treat this Run Plan as authoritative for
shared routing and evidence policy. JSONL evidence must not be silently skipped
because of existing TXT sidecars or previous JSONL records.
"""

from __future__ import annotations

import json

from ..engines.run_plan.captionforge_run_plan import (
    MAX_SEED_32,
    build_captionforge_run_config,
)


MANIFEST = {
    "name": "JLC CaptionForge Run Plan",
    "version": (0, 2, 0),
    "author": "J. L. Córdova",
    "description": (
        "Optional CaptionForge Pass A coordination node. Emits a shared "
        "CAPTIONFORGE_RUN_CONFIG object for Qwen, Joy, Lite, and future "
        "captioning engines. Defines one required project/run output folder, "
        "optional shared input_path routing, recursive and filename-glob dataset "
        "controls, captions per image, per-caption-instance seed behavior, "
        "sampling schedules, shared max image size, shared generation token "
        "budget, and a shared LoRA trigger word."
    ),
}


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
                            "Required CaptionForge project/run folder. All connected caption "
                            "nodes write shared JSONL evidence, TXT audit sidecars, and run "
                            "configs here. Downstream claim extraction and final synthesis read "
                            "from this same evidence pool."
                        ),
                    },
                ),

                "input_path": (
                    "STRING",
                    {
                        "default": "",
                        "multiline": False,
                        "tooltip": (
                            "Optional shared input image file or folder. When provided, connected "
                            "caption nodes use this path instead of their local input_path widgets. "
                            "Leave blank to use direct IMAGE inputs from the ComfyUI graph."
                        ),
                    },
                ),

                "recursive": (
                    "BOOLEAN",
                    {
                        "default": True,
                        "tooltip": (
                            "Shared folder traversal setting for input_path. If input_path is a "
                            "folder, search subfolders too. Ignored for direct IMAGE input and "
                            "single-file input_path."
                        ),
                    },
                ),

                "filename_glob": (
                    "STRING",
                    {
                        "default": "*",
                        "multiline": False,
                        "tooltip": (
                            "Shared filename filter for folder input_path. Examples: *, *.png, "
                            "*.jpg, JessJenn_*.webp, *_closeup.*. Ignored for direct IMAGE input."
                        ),
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
                            "Number of caption evidence records to generate per image, per "
                            "connected captioning engine. For example, 3 with Qwen + Joy yields "
                            "up to 6 Pass A evidence records per image."
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
                            "Base seed for the ensemble. Seeds are expanded per caption instance "
                            "according to seed_mode. Use -1 for nondeterministic behavior."
                        ),
                    },
                ),

                "seed_mode": (
                    ["fixed", "increment", "decrement", "random"],
                    {
                        "default": "increment",
                        "tooltip": (
                            "How each per-caption-instance seed is derived. fixed reuses base_seed. "
                            "increment uses base_seed, base_seed+1, base_seed+2, etc. decrement "
                            "counts down. random derives a pseudo-random sequence from base_seed, "
                            "or nondeterministic seeds when base_seed is -1."
                        ),
                    },
                ),

                "temperature_schedule": (
                    "STRING",
                    {
                        "default": "0.35, 0.60, 0.85",
                        "multiline": False,
                        "tooltip": (
                            "Comma-separated temperatures applied per caption instance. If shorter "
                            "than captions_per_image, the final value repeats. Example: with 3 "
                            "captions, '0.35, 0.75' means 0.35, 0.75, 0.75."
                        ),
                    },
                ),

                "top_p_schedule": (
                    "STRING",
                    {
                        "default": "0.90",
                        "multiline": False,
                        "tooltip": (
                            "Comma-separated top-p values applied per caption instance. If shorter "
                            "than captions_per_image, the final value repeats. Use a single value "
                            "to keep top-p constant while varying temperature."
                        ),
                    },
                ),

                "top_k_schedule": (
                    "STRING",
                    {
                        "default": "50",
                        "multiline": False,
                        "tooltip": (
                            "Comma-separated top-k values applied per caption instance. If shorter "
                            "than captions_per_image, the final value repeats. Use a single value "
                            "to keep top-k constant across the ensemble."
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
                            "modified. Use 0 to disable resizing."
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
                            "biased by one caption being allowed to ramble more than another."
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
                            "DolLoraSugarSkulls. Connected caption nodes inject this as the common "
                            "prefix for all evidence captions in the run."
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
        input_path,
        recursive,
        filename_glob,
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
                "CaptionForge Run Plan requires output_dir. Choose a project/run "
                "folder so Qwen, Joy, claim extraction, and final CaptionForge "
                "synthesis share one evidence pool."
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
            input_path=input_path,
            recursive=recursive,
            filename_glob=filename_glob,
        )

        return (cfg, json.dumps(cfg, ensure_ascii=False, indent=2))


NODE_CLASS_MAPPINGS = {
    "JLC_CaptionForgeRunPlan": JLC_CaptionForgeRunPlan,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "JLC_CaptionForgeRunPlan": "\u2003JLC CaptionForge Run Plan",
}
