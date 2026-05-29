"""
JLC CaptionForge Claim Extractor — ComfyUI Node Wrapper

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
    - The **JLC CaptionForge Claim Extractor** is the ComfyUI-facing wrapper for
      CaptionForge Pass B claim extraction.

    - It consumes the shared Pass A caption JSONL produced by CaptionForge
      caption engines, groups records by `image_key`, extracts atomic visual
      claims, normalizes rough equivalents, preserves source references, flags
      simple conflicts, and writes one Pass B claim JSONL record per image.

    - This node is intentionally text-only:
            • It does not load Qwen models
            • It does not load Joy models
            • It does not perform image captioning
            • It does not inspect image pixels directly
            • It operates only on Pass A caption audit records

    - The underlying engine module is:
            captionforge_claim_engine.py

- CaptionForge Pass Role
    - This node participates in **Pass B** of the CaptionForge pipeline.

    - Pass A generates caption evidence from one or more caption engines.

    - Pass B decomposes that caption evidence into structured claim records that
      can be inspected, compared, normalized, scored, and refined by later
      CaptionForge passes.

    - The node exposes Pass B controls for:
            • input Pass A JSONL path
            • output Pass B JSONL path
            • overwrite / append behavior
            • dry-run mode
            • image limit
            • error-record inclusion
            • low-confidence claim inclusion
            • conflict detection
            • maximum claims per caption
            • backend selection
            • image-key filtering
            • future-LLM prompt bundle export
            • raw LLM/manual-response preservation

- LLM-Ready Infrastructure
    - v0.4.0 exposes the current Pass B backend contract through ComfyUI.

    - Supported backends:
            • heuristic
                Deterministic rule-based extraction for immediate local use.

            • manual_json
                Development backend for parser and schema testing using
                precomputed JSON/JSONL responses.

    - The node can optionally export one future-LLM prompt bundle per image key,
      allowing the prompt contract to be tested before live runtime integration
      with a local text-only LLM backend.

    - Future backends may include systems such as Ollama, Transformers,
      llama.cpp, or other local providers.

- Audit and Output Behavior
    - The node returns:
            • a human-readable summary string
            • JSONL record text
            • resolved output JSONL path

    - Pass B records preserve the evidence chain from normalized claims back to
      the originating Pass A caption records.

    - v0.4.0 supports parser-audit fields from the engine:
            • llm_parse_result
            • parser_warnings
            • rejected_claims
            • raw_llm_response

- Design Philosophy
    - This node treats caption text as evidence rather than final truth.

    - The goal is not to replace human judgment with a single model output, but
      to expose intermediate caption claims so later passes can compare support,
      identify contradictions, normalize language, and produce more reliable
      LoRA-ready captions.

    - The node preserves CaptionForge’s engine-democratic design by consuming
      Pass A records from any compatible captioning engine rather than assuming
      a single canonical caption source.

- ⚠️ Development Status
    - This is Pass B v0.4.0 ComfyUI wrapper infrastructure.
    - The default backend is deterministic and heuristic.
    - Live LLM claim extraction is not yet integrated.
    - The UI, backend list, and output schema may evolve as future LLM-powered
      and consensus-based refinement passes are added.

- Attribution & License
  - Concept and implementation by **J. L. Córdova**
    with development assistance from **ChatGPT (OpenAI)**.

  - Copyright (c) 2026 J. L. Córdova

  - Released under the **MIT License**.
"""

from __future__ import annotations

MANIFEST = {
    "name": "JLC CaptionForge Claim Extractor",
    "version": (0, 4, 0),
    "author": "J. L. Córdova",
    "description": (
        "ComfyUI wrapper for CaptionForge Pass B text-only claim extraction. Consumes "
        "shared Pass A caption JSONL records, groups captions by image_key, extracts "
        "atomic visual claims, normalizes rough equivalents, preserves source references, "
        "flags simple conflicts, and writes one Pass B claim JSONL record per image. "
        "Exposes v0.4 LLM-ready infrastructure including heuristic and manual_json "
        "backends, optional future-LLM prompt bundle export, parser-audit fields, "
        "image-key filtering, dry-run output, and raw response preservation for local "
        "claim parser development."
    ),
}

from pathlib import Path

try:
    import folder_paths
except Exception:
    folder_paths = None

try:
    from .engines.captionforge_claim_engine import (
        BatchClaimConfig,
        ClaimExtractionConfig,
        default_output_path,
        default_prompt_output_path,
        extract_claims_batch,
    )
except ImportError:
    from .engines.captionforge_claim_engine import (
        BatchClaimConfig,
        ClaimExtractionConfig,
        default_output_path,
        default_prompt_output_path,
        extract_claims_batch,
    )


class JLC_CaptionForgeClaimExtractor:
    """
    ComfyUI wrapper for CaptionForge Pass B claim extraction.

    This node is intentionally text-only. It does not load image models and does
    not depend on the Qwen/Joy engines.
    """

    @classmethod
    def INPUT_TYPES(cls):
        default_output = ""
        default_prompts = ""
        if folder_paths is not None:
            try:
                out_dir = Path(folder_paths.get_output_directory())
                default_output = str(out_dir / "captionforge_claims_pass_b.jsonl")
                default_prompts = str(out_dir / "captionforge_pass_b_llm_prompts.jsonl")
            except Exception:
                default_output = ""
                default_prompts = ""

        return {
            "required": {
                "input_jsonl": (
                    "STRING",
                    {
                        "default": "",
                        "multiline": False,
                        "tooltip": (
                            "Path to the Pass A captions JSONL file produced by the Qwen/Joy "
                            "caption nodes, e.g. C:\\\\...\\\\sample_pics3\\\\captions.jsonl. "
                            "This must be the JSONL file, not the image folder."
                        ),
                    },
                ),
                "output_jsonl": (
                    "STRING",
                    {
                        "default": default_output,
                        "multiline": False,
                        "tooltip": "Path for the Pass B claim JSONL. Leave blank to write beside input_jsonl as *_claims_pass_b.jsonl.",
                    },
                ),
                "write_jsonl": (
                    "BOOLEAN",
                    {
                        "default": True,
                        "tooltip": "Write Pass B claim records to output_jsonl.",
                    },
                ),
                "overwrite": (
                    "BOOLEAN",
                    {
                        "default": True,
                        "tooltip": "Overwrite output_jsonl. Disable to append.",
                    },
                ),
                "dry_run": (
                    "BOOLEAN",
                    {
                        "default": False,
                        "tooltip": "Run extraction and return JSONL text without writing files.",
                    },
                ),
                "limit_images": (
                    "INT",
                    {
                        "default": 0,
                        "min": 0,
                        "max": 100000,
                        "step": 1,
                        "tooltip": "Maximum grouped image_key records to process. 0 means no limit.",
                    },
                ),
                "include_error_pass_a_records": (
                    "BOOLEAN",
                    {
                        "default": False,
                        "tooltip": "Include Pass A records with status != ok. Normally disabled.",
                    },
                ),
                "include_low_confidence": (
                    "BOOLEAN",
                    {
                        "default": True,
                        "tooltip": "Keep lower-confidence heuristic claims. Useful during early audit development.",
                    },
                ),
                "detect_conflicts": (
                    "BOOLEAN",
                    {
                        "default": True,
                        "tooltip": "Flag simple mutually exclusive attribute conflicts, such as multiple hair colors.",
                    },
                ),
                "max_claims_per_caption": (
                    "INT",
                    {
                        "default": 80,
                        "min": 1,
                        "max": 500,
                        "step": 1,
                        "tooltip": "Maximum atomic claims extracted from each source caption.",
                    },
                ),
                "text_llm_backend": (
                    ["heuristic", "manual_json"],
                    {
                        "default": "heuristic",
                        "tooltip": (
                            "Text-only extraction backend. heuristic uses the deterministic scaffold. "
                            "manual_json reads precomputed JSON/JSONL responses using the same schema expected from a future LLM."
                        ),
                    },
                ),
                "manual_json_path": (
                    "STRING",
                    {
                        "default": "",
                        "multiline": False,
                        "tooltip": (
                            "Optional JSON/JSONL file containing manual/future-LLM claim responses. "
                            "Required only when text_llm_backend is manual_json."
                        ),
                    },
                ),
                "write_llm_prompt_jsonl": (
                    "BOOLEAN",
                    {
                        "default": False,
                        "tooltip": "Export one future-LLM prompt bundle per image_key for backend testing/auditing.",
                    },
                ),
                "llm_prompt_jsonl": (
                    "STRING",
                    {
                        "default": default_prompts,
                        "multiline": False,
                        "tooltip": "Output path for prompt bundle JSONL. Leave blank to write beside input_jsonl as *_pass_b_llm_prompts.jsonl.",
                    },
                ),
                "image_key_filter": (
                    "STRING",
                    {
                        "default": "",
                        "multiline": False,
                        "tooltip": (
                            "Optional debug filter. Process/export only image_keys that exactly match or contain this text. "
                            "Useful for testing one LLM prompt/response at a time."
                        ),
                    },
                ),
                "preserve_raw_llm_response": (
                    "BOOLEAN",
                    {
                        "default": False,
                        "tooltip": (
                            "For manual_json/future LLM backends, store the raw response text inside the Pass B record. "
                            "Useful for debugging, but can make JSONL large."
                        ),
                    },
                ),
                "text_llm_model": (
                    "STRING",
                    {
                        "default": "",
                        "multiline": False,
                        "tooltip": "Reserved model identifier/path for future runtime LLM backends. Stored in Pass B params.",
                    },
                ),
                "text_llm_prompt": (
                    "STRING",
                    {
                        "default": (
                            "Extract atomic visible visual claims from the provided image captions. "
                            "Return strict JSON only. Preserve source references. Do not add facts "
                            "that are not supported by the captions."
                        ),
                        "multiline": True,
                        "tooltip": "Prompt instruction stored in Pass B params and used in exported future-LLM prompt bundles.",
                    },
                ),
            }
        }

    RETURN_TYPES = ("STRING", "STRING", "STRING")
    RETURN_NAMES = ("summary", "jsonl_records", "output_jsonl")
    FUNCTION = "extract_claims"
    CATEGORY = "JLC/Captioning"

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        return float("NaN")

    def extract_claims(
        self,
        input_jsonl,
        output_jsonl,
        write_jsonl,
        overwrite,
        dry_run,
        limit_images,
        include_error_pass_a_records,
        include_low_confidence,
        detect_conflicts,
        max_claims_per_caption,
        text_llm_backend,
        manual_json_path,
        write_llm_prompt_jsonl,
        llm_prompt_jsonl,
        image_key_filter,
        preserve_raw_llm_response,
        text_llm_model,
        text_llm_prompt,
    ):
        input_jsonl = (input_jsonl or "").strip()
        if not input_jsonl:
            raise RuntimeError("input_jsonl is required. Select the Pass A captions.jsonl file, not the image folder.")

        input_path = Path(input_jsonl)
        if input_path.is_dir():
            raise RuntimeError(
                "input_jsonl points to a folder, not a JSONL file:\n"
                f"  {input_path}\n"
                "Select the Pass A captions JSONL file, for example captions.jsonl."
            )

        resolved_output = (output_jsonl or "").strip()
        if not resolved_output:
            resolved_output = str(default_output_path(input_jsonl))

        resolved_prompt_jsonl = (llm_prompt_jsonl or "").strip()
        if not resolved_prompt_jsonl:
            try:
                resolved_prompt_jsonl = str(default_prompt_output_path(input_jsonl))
            except Exception:
                resolved_prompt_jsonl = ""

        config = ClaimExtractionConfig(
            include_low_confidence=bool(include_low_confidence),
            detect_conflicts=bool(detect_conflicts),
            max_claims_per_caption=int(max_claims_per_caption),
            text_llm_backend=(text_llm_backend or "heuristic").strip(),
            text_llm_model=(text_llm_model or "").strip(),
            text_llm_prompt=(text_llm_prompt or "").strip(),
            preserve_raw_llm_response=bool(preserve_raw_llm_response),
        )

        batch = BatchClaimConfig(
            input_jsonl=input_jsonl,
            output_jsonl=resolved_output,
            write_jsonl=bool(write_jsonl),
            dry_run=bool(dry_run),
            limit_images=int(limit_images),
            include_error_pass_a_records=bool(include_error_pass_a_records),
            overwrite=bool(overwrite),
            manual_json_path=(manual_json_path or "").strip(),
            write_llm_prompt_jsonl=bool(write_llm_prompt_jsonl),
            llm_prompt_jsonl=resolved_prompt_jsonl,
            image_key_filter=(image_key_filter or "").strip(),
        )

        result = extract_claims_batch(batch, config)

        total_atomic = sum(len(r.atomic_claims) for r in result.records)
        total_normalized = sum(len(r.normalized_claims) for r in result.records)
        total_conflicts = sum(len(r.conflicts) for r in result.records)
        total_rejected = sum(len(getattr(r, "rejected_claims", []) or []) for r in result.records)
        total_parser_warnings = sum(len(getattr(r, "parser_warnings", []) or []) for r in result.records)

        prompt_line = ""
        if write_llm_prompt_jsonl:
            prompt_line = f"LLM prompt JSONL: {resolved_prompt_jsonl}\n"

        summary = (
            "CaptionForge Pass B claim extraction complete.\n"
            f"Node version: 0.4.0\n"
            f"Engine version: {config.engine_version}\n"
            f"Input JSONL: {input_jsonl}\n"
            f"Output JSONL: {resolved_output if write_jsonl and not dry_run else '(not written)'}\n"
            f"{prompt_line}"
            f"Images processed: {result.processed}\n"
            f"Failed groups: {result.failed}\n"
            f"Atomic claims: {total_atomic}\n"
            f"Normalized claims: {total_normalized}\n"
            f"Conflict records: {total_conflicts}\n"
            f"Backend: {config.text_llm_backend}"
        )

        return (summary, result.jsonl_text, resolved_output)


NODE_CLASS_MAPPINGS = {
    "JLC_CaptionForgeClaimExtractor": JLC_CaptionForgeClaimExtractor,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "JLC_CaptionForgeClaimExtractor": "\u2003JLC CaptionForge Claim Extractor",
}
