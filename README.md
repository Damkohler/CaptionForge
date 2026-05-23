# CaptionForge

**CaptionForge** is a model-agnostic captioning framework for ComfyUI, designed to help generate cleaner, more consistent, human-reviewable captions for LoRA dataset preparation and image dataset analysis.

Rather than treating any single captioning model as the final authority, CaptionForge is built around the idea that caption quality can be improved through structured comparison, repeated observation, audit trails, statistical agreement, LLM-assisted reasoning, and consensus-based refinement.

The long-term goal is to approximate human-quality descriptive captions by using existing vision and language models inside a more deliberate captioning infrastructure.

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

## Current focus

CaptionForge is currently focused on ComfyUI nodes and shared captioning engines for local dataset preparation.

Initial capabilities include:

- Folder-based image captioning
- Optional direct ComfyUI `IMAGE` input support
- `.txt` sidecar caption writing
- JSONL audit output
- Run configuration logging for reproducibility
- Lightweight daily-use node variants
- Model cache management to reduce unnecessary reloads and VRAM pressure
- Support for multiple captioning engines within one framework

The first repository seed is expected to include early ComfyUI caption nodes, shared engine code, and audit-oriented output utilities.

## Model-agnostic direction

CaptionForge is intended to remain engine-democratic.

The framework should be able to incorporate different kinds of captioning or vision-language systems, such as:

- local VLM captioners
- specialized aesthetic or tagging models
- general-purpose multimodal models
- LLM-based text cleanup passes
- future visual reasoning systems
- additional robustness engines as they become useful

No single model is assumed to be canonical. The infrastructure is designed so different engines can contribute observations to a broader captioning and refinement process.

## Planned multi-pass pipeline

CaptionForge is being designed around a multi-pass strategy:

- **Pass A:** Generate one or more captions from one or more engines
- **Pass B:** Extract atomic visual claims from caption text
- **Pass C:** Normalize equivalent claims  
  Example: `blonde hair`, `light blond hair`, and `golden hair` may refer to the same visual attribute
- **Pass D:** Compare claims across engines, prompts, and repeated runs
- **Pass E:** Score claims by agreement, confidence, and usefulness
- **Pass F:** Produce cleaner LoRA-ready captions
- **Pass G:** Preserve all intermediate evidence in auditable JSONL records

The final caption should be less like a single model guess and more like a consensus summary of supported visual evidence.

## Philosophy

CaptionForge favors:

- local-first workflows
- reproducible outputs
- auditable intermediate data
- model-agnostic design
- practical LoRA dataset preparation
- human-reviewable decisions
- clear separation between engines, node wrappers, and refinement logic
- consensus over single-model authority

## Status

CaptionForge is in early development.

The current repository is a preliminary project home. Code, APIs, node names, and folder structure may change before the first stable release.

## Attribution & License

Concept and implementation by **J. L. Córdova**, with development assistance from **ChatGPT (OpenAI)**.

Copyright (c) 2026 J. L. Córdova

Released under the **MIT License**. See [`LICENSE`](./LICENSE) for details.
