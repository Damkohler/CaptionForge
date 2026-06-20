# CaptionForge — experimental ComfyUI developer preview

> **CaptionForge is an experimental ComfyUI captioning framework for LoRA dataset preparation.**
>
> The current v0.1.0 preview is ready for ComfyUI users and node developers to install, test, compare, critique, and improve.
>
> CaptionForge runs as a practical multi-pass captioning workflow: raw caption witnesses, text-LLM distillation, image-aware VLM validation, and auditable TXT/JSONL exports.
>
> This is still an early public preview, so node names, JSONL schemas, prompts, model defaults, and documentation may evolve before a more formal release. Caption quality may also vary by dataset and model choice.
>
> The main question this preview is trying to answer is simple: can multi-engine witness captions, text-LLM distillation, image-aware VLM validation, and JSONL audit trails produce better LoRA training captions than a single strong captioner alone?
>
> Feedback, caption-quality comparisons, model recommendations, bug reports, and workflow suggestions are welcome.

---

<p align="center">
  <img src="assets/icons/jlc-comfyui-nodes_Logo-0512.png" width="120">
  &nbsp;&nbsp;&nbsp;
  <img src="assets/icons/jlc-comfyui-nodes_Logo-Dark-0512.png" width="120">
</p>

[![ComfyUI](https://img.shields.io/badge/ComfyUI-Custom%20Nodes-blue)]()
[![License](https://img.shields.io/badge/license-MIT-green)]()
![Status](https://img.shields.io/badge/status-experimental-orange)
![Version](https://img.shields.io/badge/version-0.1.0-orange)

## What CaptionForge is

**CaptionForge** is a local, model-agnostic captioning framework for building richer, more auditable captions for LoRA dataset preparation inside ComfyUI.

A single vision-language model caption can be useful, but it is not always reliable. CaptionForge treats raw captions as imperfect witness statements, then uses a multi-pass process to merge, validate, correct, and export final training captions.

The current developer preview is tuned primarily for character and style LoRA captioning: detailed human, humanoid, illustrated, rendered, doll-like, pageant, cosplay, fashion, portrait, and stylized subjects where clothing, pose, body shape, facial traits, hair, eyes, makeup, materials, color, lighting, framing, and visible style cues matter.

Other image domains may work, but will require custom prompts, model choices, or workflow tuning. These are in the works and will be added in the near future.

## Current status

CaptionForge v0.1.0 is an **experimental developer preview**.

That means:

- the project is intended to be installed and test-driven
- the workflow may still change
- node names and categories may still be refined
- JSONL schemas and audit fields may evolve
- model behavior depends heavily on the selected backend models
- it may or may not outperform a strong standalone captioner for your dataset
- bug reports and comparison feedback are valuable

CaptionForge does **not** ship model weights. Large Joy, Qwen, Ollama, and other model downloads remain user-controlled.

Experimental or unsupported branches should remain unregistered by default and should not appear as active ComfyUI nodes unless explicitly imported by the package.

## Main workflow

The active CaptionForge workflow is a multi-pass pipeline:

```text
Pass A: Raw witness captions
  Joy Python caption witness xN
  Qwen Python caption witness xN
  optional Ollama VLM caption witness xN

Pass B: Text-LLM distillation
  a text-only Ollama LLM treats raw captions as witness ballots
  it emits accepted claims, plausible singleton candidates, rejected conflicts,
  and rich/taggy draft captions

Pass C: Image-aware VLM validation
  an Ollama VLM checks the Pass B evidence against the actual image
  it removes unsupported details, corrects visible errors, preserves supported
  LoRA-relevant detail, and writes validated final-caption candidates

Pass D: Export
  natural captions and taggy/comma-style captions are exported as TXT/JSONL sidecars
```

The intended behavior is not merely “summarize several captions.” The goal is to preserve useful trainable visual detail while reducing unsupported hallucinations through later image-aware validation.

## Why use this instead of standalone Joy or some other standard captioning method/approach?

You may not need to.

JoyCaption can be excellent by itself, especially with a good local setup or a strong hosted implementation. CaptionForge is an experiment in whether a structured local pipeline can do better often enough to justify the extra workflow complexity.

CaptionForge may be useful when:

- one raw captioner is strong but misses details
- different captioners notice different useful details
- you want JSONL audit trails of intermediate records
- you want a text LLM to consolidate repeated witness evidence
- you want a VLM to validate the draft against the source image
- you want natural and taggy exports from the same audited run
- you are preparing LoRA training captions and care about visible, trainable detail

A useful negative result is still useful: if a dataset is better served by standalone Joy, Qwen, or another captioner, that is exactly the kind of comparison this preview is meant to surface.

## What CaptionForge tries to optimize

CaptionForge currently favors captions that are:

- richer than a single generic image caption
- less hallucinated than unvalidated text-only synthesis
- useful for LoRA training
- auditable through JSONL sidecars
- locally runnable
- prompt-configurable
- model-agnostic enough to swap better witnesses, distillers, validators, and formatters over time

Useful caption details often include:

- subject type and visible style
- face shape and facial traits
- hair color and hairstyle
- eye color and makeup as separate details
- expression and pose
- hands and body position
- body shape and visible proportions when relevant
- clothing construction, layers, and fit
- accessories, jewelry, nails, props, and distinctive details
- colors, materials, textures, lighting, background, framing, and crop

Visible sensual, glamour, swimwear, lingerie, revealing clothing, cleavage, side openings, exposed midriff, or similar styling may be described neutrally when it is actually visible and relevant to the dataset. CaptionForge prompts should not invent hidden anatomy, unseen clothing, explicit acts, or contradicted details.

## Active node families

Node categories are being normalized under:

```text
Captioning/CaptionForge
```

with active caption nodes under:

```text
Captioning/CaptionForge/Caption Nodes
```

### Pipeline and orchestration

#### JLC CaptionForge Pipeline Planner

The central planning node for normal runs.

It coordinates:

- input image path or direct image passthrough
- recursive folder traversal
- filename glob filtering
- output directory
- run name
- overwrite behavior
- Pass A witness run counts
- seed schedules
- sampling schedules
- max image size
- max token budget
- LoRA trigger word
- user caption anchor
- distiller settings
- validator settings
- final export settings
- derived JSONL/TXT/config paths

#### JLC CaptionForge

The main capstone/orchestration node.

Current target behavior:

1. consume Pass A raw caption records
2. run Pass B text-LLM distillation
3. run Pass C image-aware VLM validation
4. export final natural and taggy captions

The natural caption should come from the image-aware VLM validation pass. A text-only stage may be used for formatting/taggy output, but it should not blindly rewrite the natural final caption.

### Caption witnesses

#### JLC CaptionForge Joy Caption

Python-based JoyCaption/LLaVA-family Pass A witness.

Joy remains one of the strongest raw caption witnesses and is treated as a major first-class CaptionForge caption source.

#### JLC CaptionForge Qwen Caption

Python-based Qwen-family Pass A witness.

Qwen is useful as a second caption voice, especially when its model behavior complements Joy. Optional 8-bit loading may be available where supported.

#### JLC CaptionForge Ollama Caption

Ollama-backed VLM Pass A witness.

This node delegates model execution to a local Ollama server rather than loading Hugging Face/PyTorch model weights inside ComfyUI. It can use configured Ollama VLM tags such as:

```text
gemma4:26b
qwen3.6:35B-A3B
huihui_ai/gemma-4-abliterated:26b
```

The Ollama Caption node is an optional raw-caption witness. It does not replace the later VLM validator/capstone role.

### Prompt and option helpers

#### JLC CaptionForge Template Options

Shared prompt-option sidecar for caption nodes.

It is intended to help request consistent LoRA-relevant detail across caption witnesses without forcing every backend model into the same prompt implementation.

#### CaptionForge Ollama model dropdown config

The file:

```text
config/captionforge_ollama_models.json
```

defines user-editable Ollama model tags for dropdowns used by distiller, validator, formatter, and Ollama caption-witness nodes.

## Output files

CaptionForge writes auditable sidecars during planned runs. Current conventions include:

```text
<run_name>__A_RAW_CAPTIONS.jsonl
<run_name>__B_DISTILL.jsonl
<run_name>__B_DISTILL_readable.jsonl
<run_name>__B_DISTILL_readable.json
<run_name>__B_DISTILL_prompts.jsonl
<run_name>__C_VLM_VALIDATED.jsonl
<run_name>__C_VLM_VALIDATED_readable/
<run_name>__C_VLM_VALIDATOR_prompts.jsonl
<run_name>__D_FINAL_EXPORT.jsonl
<run_name>__TXT/
<run_name>__output_paths.json
<run_name>__run_config.json
```

Exact filenames and schemas may change during the developer-preview phase.

Final outputs are expected to include:

```text
Natural caption:  VLM-validated prose
Taggy caption:    comma-separated LoRA-style caption
```

## Model configuration

CaptionForge uses two model ecosystems:

1. **Python / Hugging Face model folders** for Joy and Qwen witness engines.
2. **Ollama models** for text-LLM distillation, image-aware VLM validation, optional formatting, and Ollama-backed caption witnesses.

A typical Ollama dropdown config:

```json
{
  "_meta": {
    "name": "CaptionForge Ollama Model Dropdowns",
    "version": "0.1.0",
    "description": "User-editable Ollama model dropdown configuration for CaptionForge nodes and engines.",
    "consumed_by": [
      "nodes/captionforge_ollama_model_dropdowns.py",
      "CaptionForge Pipeline Planner",
      "JLC CaptionForge capstone",
      "JLC CaptionForge Ollama Caption"
    ],
    "notes": [
      "Values should be concrete Ollama model tags used exactly as written.",
      "distiller_models are used for text-only LLM distillation and formatting stages.",
      "validator_models are used for image-aware VLM validation.",
      "caption_models are used by Ollama-backed Pass A caption witness nodes.",
      "Set include_custom to true to expose a custom model-tag entry in supported nodes."
    ]
  },
  "distiller_models": [
    "mistral-small:24b",
    "VladimirGav/gemma4-26b-16GB-VRAM-Uncensored",
    "deepseek-r1:32b",
    "tarruda/neuraldaredevil-8b-abliterated:fp16",
    "gpt-oss:20b"
  ],
  "validator_models": [
    "gemma4:26b",
    "qwen3.6:35B-A3B",
    "huihui_ai/gemma-4-abliterated:26b"
  ],
  "format_models": [
    "mistral-small:24b",
    "VladimirGav/gemma4-26b-16GB-VRAM-Uncensored",
    "gpt-oss:20b",
    "deepseek-r1:32b"
  ],
  "caption_models": [
    "gemma4:26b",
    "qwen3.6:35B-A3B",
    "huihui_ai/gemma-4-abliterated:26b"
  ],
  "defaults": {
    "distiller_model": "mistral-small:24b",
    "validator_model": "gemma4:26b",
    "format_model": "mistral-small:24b",
    "caption_model": "gemma4:26b"
  },
  "include_custom": true
}
```

Terminology:

```text
distiller_model   text-only LLM for Pass B distillation
validator_model   image-aware VLM for Pass C validation
format_model      text-only LLM for formatting/taggy conversion when used
caption_model     Ollama-backed Pass A image-caption witness model
```

## Model locations

Large model weights are intentionally not stored in this repository.

Python-based witness models are expected under ComfyUI model folders, for example:

```text
ComfyUI/models/LLM/JLC_QwenCaption/
ComfyUI/models/LLM/JLC_JoyCaption/
```

Ollama models must be installed and runnable through Ollama outside this repository.

CaptionForge does not require every supported backend to be installed for every workflow. Users can test smaller subsets first.

## Installation

### ComfyUI custom node install

Clone or copy CaptionForge into your ComfyUI custom nodes directory:

```text
ComfyUI/custom_nodes/CaptionForge
```

Then restart ComfyUI.

### Recommended test posture

CaptionForge follows the normal risk profile of experimental ComfyUI custom nodes: it may require Python dependency setup, local model folders, Ollama model pulls, and enough VRAM for the selected workflow.

For heavily customized or mission-critical ComfyUI environments, test in a separate ComfyUI install first.

This is especially sensible if your main ComfyUI environment has tightly pinned CUDA, PyTorch, xformers, transformers, bitsandbytes, or other local-AI dependencies.

## Dependencies

Python dependencies are declared in `pyproject.toml`.

Typical local use may involve:

```text
torch
transformers
accelerate
huggingface-hub
pillow
numpy
safetensors
qwen-vl-utils
```

Optional quantization support may involve:

```text
bitsandbytes
```

Ollama-backed stages require a working local Ollama installation and installed Ollama model tags.

## Hardware notes

CaptionForge is designed for local workflows, but strong results may require large local models.

Practical performance depends on:

- GPU VRAM
- system RAM
- model size
- quantization mode
- Ollama version
- context length
- image size
- number of Pass A witness runs
- whether models are kept loaded or unloaded between runs

The author's active development environment includes an RTX 4090 Laptop GPU with 16 GB VRAM. Larger models may be slow, may require careful quantization, or may need more capable hardware.

## Experimental branches

Some experimental or unsupported code may exist in the repository for future A/B testing or research.

Experimental branches should be:

- clearly labeled
- kept out of the normal ComfyUI registration path
- not imported by `__init__.py`
- not shown as mainline nodes unless deliberately enabled
- treated as unsupported starting points rather than stable user features

The active public workflow should be the main Planner → Pass A witnesses → Distiller → VLM Validator → Export path.

## Development principles

CaptionForge currently prioritizes:

- local execution
- auditable intermediate records
- JSONL sidecars
- reusable engines separated from ComfyUI node wrappers
- planner-driven workflows
- model cache and VRAM hygiene
- strong defaults for LoRA captioning
- explicit prompt roles
- model-agnostic backends
- visible, trainable detail over generic caption prose
- practical feedback from real datasets

## Feedback wanted

Useful feedback includes:

- comparisons against standalone JoyCaption, Qwen, or other captioners
- examples where CaptionForge improves caption quality
- examples where CaptionForge makes captions worse
- hallucination reports
- missed-detail reports
- model recommendations
- prompt improvements
- broken node reports
- workflow usability feedback
- VRAM/performance observations
- JSONL/audit trail suggestions

Please include enough context to reproduce the issue or evaluate the result: selected nodes, model tags, relevant settings, whether the run used direct IMAGE input or a folder path, and a small sample of generated captions when possible.

## Attribution & License

Concept and implementation by **J. L. Córdova**, with development assistance from **ChatGPT (OpenAI)**.

CaptionForge's Joy/template-option workflow is locally adapted and was inspired in part by the practical template interface pattern used by the public JoyCaption Beta One Hugging Face Space:

```text
https://huggingface.co/spaces/fffiloni/JoyCaption-Beta-One
```

Copyright (c) 2026 J. L. Córdova

Released under the **MIT License**. See [`LICENSE`](./LICENSE) for details.
