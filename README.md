# CaptionForge

**CaptionForge** is a model-agnostic captioning framework for ComfyUI, designed to generate cleaner, more consistent, auditable captions for LoRA dataset preparation and image dataset analysis.

Rather than treating any single captioning model as the final authority, CaptionForge is built around the idea that caption quality can be improved through structured comparison, repeated observation, audit trails, claim extraction, statistical agreement, LLM-assisted reasoning, and consensus-based refinement.

The long-term goal is to approximate human-quality descriptive captions by using existing vision and language models inside a more deliberate captioning infrastructure.

## Current Status

CaptionForge now includes its first working ComfyUI implementation.

The current codebase provides:

- Qwen-family captioning through a full ComfyUI node
- JoyCaption/LLaVA-family captioning through a full ComfyUI node
- Lite daily-use caption nodes for Qwen and Joy
- Shared model cache management
- Pass A JSONL caption audit records
- Pass B text-only claim extraction
- LLM-ready Pass B scaffolding for future local text-model integration
- Frontend icon branding for CaptionForge nodes

This is still early development, but the repository now contains working code rather than only a project placeholder.

## Vision

Image captioning models are powerful, but individual captions can still contain:

- hallucinated visual details
- missing important features
- inconsistent wording
- repetition
- awkward grammar
- overconfident but unsupported claims
- model-specific biases

CaptionForge aims to reduce these problems by treating captions as evidence, not answers.

Multiple engines can describe the same image. Their outputs can then be audited, decomposed into visual claims, compared, normalized, scored, and recombined into cleaner final captions.

## Current Nodes

CaptionForge currently registers the following ComfyUI nodes:

### JLC Qwen Caption

Full Qwen-family vision-language captioning node.

Supports direct ComfyUI `IMAGE` input, file/folder captioning, prompt presets, custom prompts, prompt files, TXT sidecars, JSONL audit output, run configuration logging, model download probing, and optional bitsandbytes 8-bit loading.

### JLC Joy Caption

Full JoyCaption/LLaVA-family captioning node.

Supports direct ComfyUI `IMAGE` input, file/folder captioning, CaptionForge prompt presets, JoyCaption-native template controls, TXT sidecars, JSONL audit output, run configuration logging, model download probing, and memory-efficient 8-bit loading.

### JLC Qwen Caption (Lite)

Minimal direct-image Qwen captioning node for daily interactive use.

This Lite node exposes only the core controls needed for quick captioning inside ComfyUI.

### JLC Joy Caption (Lite)

Minimal direct-image JoyCaption node for daily interactive use.

This Lite node exposes only the core controls needed for quick captioning inside ComfyUI.

### JLC CaptionForge Claim Extractor

Text-only Pass B node.

Consumes shared Pass A caption JSONL, groups records by `image_key`, extracts atomic visual claims, normalizes rough equivalents, preserves source references, flags simple conflicts, and writes one Pass B claim JSONL record per image.

## Current Files

The first code seed includes:

```text
__init__.py
captionforge_model_cache.py
captionforge_claim_engine.py
jlc_qwen_caption_engine.py
jlc_joy_caption_engine.py
jlc_qwen_caption_CUI_node.py
jlc_joy_caption_CUI_node.py
jlc_qwen_caption_lite_CUI_node.py
jlc_joy_caption_lite_CUI_node.py
jlc_captionforge_claim_extractor_CUI_node.py
web/jlc_captionforge_icons.js
web/assets/icons/jlc-comfyui-nodes_Logo-Dark-0128.png

## Pipeline Direction

CaptionForge is being designed around a multi-pass strategy.

### Pass A — Caption Generation

Generate one or more captions from one or more engines.

Current Pass A engines include:

- Qwen-family VLM captioning
- JoyCaption/LLaVA-family captioning

Pass A records are written as JSONL audit entries containing model metadata, prompt settings, generation parameters, raw captions, cleaned captions, and image keys.

### Pass B — Claim Extraction

Convert caption text into atomic visual claims.

Current Pass B status:

- deterministic heuristic backend
- manual JSON backend for parser testing
- LLM-ready prompt and response schema
- parser warnings
- rejected claim records
- optional raw response preservation
- future-LLM prompt bundle export

Live LLM extraction is not yet integrated.

### Future Passes

Planned future stages may include:

- normalizing equivalent claims
- comparing claims across engines and repeated runs
- detecting contradictions
- scoring claims by support and usefulness
- validating captions against claim evidence
- text-only LLM cleanup/refinement
- producing final LoRA-ready captions
- preserving all intermediate evidence in auditable records

The final caption should be less like a single model guess and more like a consensus summary of supported visual evidence.

## Model-Agnostic Direction

CaptionForge is intended to remain engine-democratic.

The framework should be able to incorporate different kinds of captioning, tagging, or visual reasoning systems, such as:

- local VLM captioners
- specialized aesthetic or tagging models
- general-purpose multimodal models
- LLM-based text cleanup passes
- validator models
- future visual reasoning systems
- additional robustness engines as they become useful

No single model is assumed to be canonical.

Qwen and Joy are the first working engines, not the final boundary of the project.

## Installation

Clone or copy this repository into your ComfyUI `custom_nodes` folder:

`ComfyUI/custom_nodes/CaptionForge`

Restart ComfyUI after installation.

Model files are not included in this repository. CaptionForge nodes use local model folders under ComfyUI’s `models/LLM/` directory and may offer download-probe behavior through the node UI.

## Model Locations

Current model roots:

`ComfyUI/models/LLM/JLC_QwenCaption/`

`ComfyUI/models/LLM/JLC_JoyCaption/`

Qwen and Joy models are Hugging Face model directories, not single checkpoint files.

Large model weights are intentionally not stored in this repository.

## Notes

CaptionForge is designed for local workflows and practical dataset preparation.

The current implementation favors:

- reproducible settings
- auditable intermediate records
- explicit JSONL outputs
- separation between ComfyUI node wrappers and reusable engines
- local model execution
- careful VRAM/cache management
- incremental development toward consensus-based caption refinement

## Development Status

This repository is in early active development.

APIs, node names, file layout, prompt presets, model registries, and output schemas may evolve before a stable public release.

## Attribution & License

Concept and implementation by **J. L. Córdova**, with development assistance from **ChatGPT (OpenAI)**.

Copyright (c) 2026 J. L. Córdova

Released under the **MIT License**. See [`LICENSE`](./LICENSE) for details.