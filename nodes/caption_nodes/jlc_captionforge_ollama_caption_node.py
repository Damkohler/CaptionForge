"""
JLC CaptionForge Ollama Caption — ComfyUI Node Wrapper

- CaptionForge
  - This node is part of **CaptionForge**, a model-agnostic captioning
    framework for ComfyUI developed by **J. L. Córdova**.

  - Repository
    https://github.com/Damkohler/CaptionForge

- CaptionForge focuses on practical dataset-captioning infrastructure for
  LoRA dataset preparation, using multi-engine caption generation, JSONL
  audit trails, claim extraction and refinement, text-LLM distillation,
  image-aware VLM validation, and consensus-oriented caption improvement
  to produce grounded, auditable training captions.

- Node Purpose
    - The **JLC CaptionForge Ollama Caption** node provides a ComfyUI
      frontend for Ollama-hosted vision-language image captioning inside
      CaptionForge.

    - This file is the **ComfyUI-facing wrapper**, not an Ollama model
      implementation. It is responsible for:
            • ComfyUI INPUT_TYPES / widget definitions
            • optional direct IMAGE tensor input
            • IMAGE tensor conversion to PIL/base64 payloads
            • optional Pipeline Planner file/folder `input_path` routing
            • Ollama model tag dropdown and custom model tag handling
            • local Ollama availability checks and best-effort remote tag probes
            • optional Ollama pull behavior for missing models
            • clear template-vs-custom prompt controls
            • CaptionForge Pipeline Planner consumption through `pipeline_plan`
            • CaptionForge Template Options consumption through `template_options`
            • TXT audit sidecar writing in planned runs
            • shared JSONL audit output in planned runs
            • direct ComfyUI caption and resolved-prompt string outputs
            • IMAGE and pipeline-plan passthrough for clean graph chaining
            • node display name, category, and mapping registration

    - Model execution is delegated to a local Ollama server. The node uses the
      Ollama HTTP API instead of loading Hugging Face/PyTorch model weights
      inside the ComfyUI Python process.

- CaptionForge Pipeline Role
    - This node participates in **Pass A** of the CaptionForge pipeline.

    - Pass A generates auditable caption evidence records from one or more
      captioning engines.

    - The Ollama Caption node contributes Ollama-hosted VLM caption evidence
      compatible with downstream CaptionForge claim extraction, semantic
      synthesis, and final caption construction.

    - CaptionForge audit fields include:
            • captionforge_pass
            • model_family
            • ensemble_run_index
            • image_key
            • raw_caption
            • final cleaned caption
            • generation parameters
            • system prompt and prompt metadata
            • model metadata

- Node Workflow Model
    - The node supports two image sources:
            • direct ComfyUI IMAGE input
            • file/folder routing supplied by the CaptionForge Pipeline Planner

    - These may be used independently or together.

    - In standalone mode, the node behaves as a direct IMAGE captioning node:
            • captions the connected IMAGE or IMAGE batch
            • returns caption text and resolved prompt text
            • remains file-silent from the user's point of view
            • passes the IMAGE through unchanged

    - When connected to the CaptionForge Pipeline Planner, this node consumes
      the shared CAPTIONFORGE_PIPELINE_PLAN through the `pipeline_plan` pin and
      uses the Ollama caption family plan to determine:
            • whether Ollama captioning participates in the run
            • how many Ollama raw-caption records to generate per image
            • shared output directory
            • optional shared input path
            • recursive folder traversal
            • filename filtering
            • per-caption-instance seed values
            • per-caption-instance sampling values
            • shared max image size
            • shared max token budget
            • shared LoRA trigger word

    - In planned mode, the node can write:
            • TXT audit sidecar captions
            • JSONL audit records
            • run-configuration JSON files

- Prompting Model
    - The node supports two explicit prompt modes:
            • caption_template_mode
            • custom_prompt_mode

    - `caption_template_mode` uses caption_type, caption_length, and optional
      Template Options supplied by the `template_options` pin.

    - `custom_prompt_mode` uses custom_prompt when provided, otherwise a local
      prompt_preset fallback. The custom_prompt widget is prepopulated with the
      default LoRA-oriented prompt so users can see exactly what the default
      custom mode would send.

    - If both booleans are enabled, custom_prompt_mode intentionally takes
      precedence. If both are disabled, the node falls back to template mode.

    - Local extra-option widgets are not duplicated. Shared template modifiers
      live only in the CaptionForge Template Options sidecar node.

- Model and Dependency Notes
    - Ollama model tags are not Hugging Face repository folders. They are served
      by the local Ollama daemon, usually at http://127.0.0.1:11434.

    - The node uses Ollama `/api/chat` as the primary image-conditioned VLM
      endpoint, with `/api/generate` as a fallback for compatibility.

    - The visible `max_new_tokens` widget maps to Ollama's `num_predict`
      parameter. The name is kept consistent with the other CaptionForge nodes
      and the Pipeline Planner.

    - Runtime behavior depends on the installed Ollama application, available
      model tags, the active Ollama server URL, Pillow, NumPy, and PyTorch only
      for ComfyUI IMAGE tensor conversion.

- Design Philosophy
    - This node keeps Ollama-hosted VLM captioning as one independent captioning
      voice inside CaptionForge while presenting the same workflow shape as the
      Joy and Qwen Caption nodes.

    - CaptionForge is engine-democratic: Ollama captions contribute evidence to
      the broader audit and consensus pipeline alongside Joy, Qwen, cleanup
      LLMs, validators, or other robustness engines.

    - The node prioritizes reproducibility, auditability, low UI clutter, and
      clean separation between the ComfyUI interface and the backend model
      service.

- ⚠️ Development Status
    - This is early CaptionForge Ollama-backed raw-caption infrastructure.
    - The UI, model-tag config, prompt behavior, and output audit fields may
      evolve as the multi-pass CaptionForge pipeline matures.
    - The node is intended for local dataset preparation and controlled caption
      audit workflows.

- Attribution & License
  - Concept and implementation by **J. L. Córdova**
    with development assistance from **ChatGPT (OpenAI)**.

  - CaptionForge's template-option workflow is locally adapted and was inspired
    in part by the practical template interface pattern used by the public
    JoyCaption Beta One Hugging Face Space at:
    https://huggingface.co/spaces/fffiloni/JoyCaption-Beta-One.

  - Ollama model execution is designed around local Ollama HTTP API behavior.

  - Designed for use with:
    https://github.com/comfyanonymous/ComfyUI

  - Copyright (c) 2026 J. L. Córdova

  - Released under the **MIT License**.
"""

from __future__ import annotations
from ...captionforge_version import CAPTIONFORGE_VERSION

MANIFEST = {
    "name": "JLC CaptionForge Ollama Caption",
    "version": CAPTIONFORGE_VERSION,
    "author": "J. L. Córdova",
    "description": (
        "Ollama-backed CaptionForge frontend for local vision-language image captioning. "
        "Provides direct IMAGE captioning while also consuming CAPTIONFORGE_PIPELINE_PLAN "
        "objects from the CaptionForge Pipeline Planner through the pipeline_plan input. "
        "Template extras are consumed only through the template_options input from the "
        "CaptionForge Template Options sidecar. The UI separates caption_template_mode and "
        "custom_prompt_mode for clearer prompt routing while delegating model availability, "
        "pulling, and image-conditioned generation to a local Ollama server. Uses Ollama "
        "/api/chat as the primary VLM endpoint, with /api/generate as compatibility fallback."
    ),
}

import base64
import io
import json
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image

import folder_paths

from ...engines.captionforge_pipeline_planner_engine import expand_captionforge_runs
from ...engines.captionforge_caption_prompt_kit import (
    CAPTION_LENGTH_CHOICES,
    CAPTION_TYPE_CHOICES,
    build_caption_prompt,
)
from ..jlc_captionforge_template_options import resolve_effective_extra_options
from ..captionforge_ollama_model_dropdowns import (
    CUSTOM_VALUE,
    load_ollama_model_dropdowns,
)

try:
    from ...engines.captionforge_model_cache import (
        cache_size as _captionforge_cache_size,
        unload_all as _captionforge_unload_all,
    )
except Exception:  # pragma: no cover - keeps direct/local smoke tests importable
    _captionforge_cache_size = None
    _captionforge_unload_all = None


DEFAULT_OLLAMA_URL = "http://127.0.0.1:11434"
DEFAULT_JSONL_FILENAME = "captions.jsonl"
_SUPPORTED_IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff"}


def _evict_python_models_before_ollama_if_needed(caller: str) -> None:
    """Unload resident Python/HF CaptionForge models before handing off to Ollama.

    Ollama model residency is owned by the Ollama daemon, not by CaptionForge's
    process-local Python model cache. Therefore this only runs when the Python
    cache is non-empty and naturally becomes a no-op between Ollama models.
    """
    if _captionforge_cache_size is None or _captionforge_unload_all is None:
        return

    try:
        resident = int(_captionforge_cache_size(include_keep=True))
    except Exception as exc:
        print(f"[{caller}] WARNING: Could not inspect CaptionForge Python model cache before Ollama handoff: {exc}", flush=True)
        return

    if resident <= 0:
        return

    try:
        evicted = int(
            _captionforge_unload_all(
                include_keep=True,
                reason="handoff_to_ollama",
                safe=True,
            )
        )
        if evicted:
            print(f"[{caller}] Evicted {evicted} CaptionForge Python model(s) before Ollama handoff.", flush=True)
    except Exception as exc:
        print(f"[{caller}] WARNING: Python model cache eviction before Ollama handoff failed: {exc}", flush=True)


FEMALE_CHARACTER_LORA_SYSTEM_PROMPT = """You are a multimodal image captioning engine for female character LoRA dataset preparation.

Your job is to inspect the image and write one strong standalone caption.

Rules:
- Describe only visible image content.
- Output exactly one paragraph and nothing else.
- Do not output analysis, reasoning, notes, bullet points, labels, or a thinking trace.
- Do not roleplay, address the user, ask follow-up questions, or include safety disclaimers.
- If a detail is uncertain, omit it rather than hedge.
- Use direct dataset-caption language, not chatty commentary."""

GENERAL_CHARACTER_SYSTEM_PROMPT = """You are a multimodal image captioning engine for character dataset preparation.

Your job is to inspect the image and write one strong standalone caption.

Rules:
- Describe only visible image content.
- Output exactly one paragraph and nothing else.
- Do not output analysis, reasoning, notes, bullet points, labels, or a thinking trace.
- Do not roleplay, address the user, ask follow-up questions, or include safety disclaimers.
- If a detail is uncertain, omit it rather than hedge.
- Use direct dataset-caption language, not chatty commentary."""

DEFAULT_SYSTEM_PROMPT = FEMALE_CHARACTER_LORA_SYSTEM_PROMPT

FEMALE_CHARACTER_LORA_PROMPT = """Write one clean caption for a female character LoRA dataset.

Caption priorities:
- Be specific rather than generic.
- Prioritize female-character-LoRA-relevant visual traits: subject type, pose, facial structure, expression, hair color and hairstyle, eye color, makeup color as distinct from eye color, lip shape and fullness, skin appearance, neck length, shoulder line, bust/chest shape, waist definition, hip width, body proportions, silhouette, legs, clothing construction, accessories, dominant colors, lighting, background, framing, and visual style.
- Include visible anatomy and figure information when clear and relevant. Do not omit body-shape or facial-structure details merely because they are sensual, revealing, or anatomy-related.
- If the image presents glamour, sensuality, revealing clothing, lingerie, swimwear, cleavage, underboob, sideboob, bare midriff, exposed hips, thighs, or other sexualized styling, describe that directly and matter-of-factly when visible.
- Use strong, open dataset-caption language rather than timid euphemisms, but do not turn the caption into roleplay, commentary, or erotic prose.
- Do not invent nudity, explicit acts, hidden anatomy, or hidden clothing details.
- Do not force photo, render, doll, anime, realistic, or stylized language unless supported by the image.
- Avoid meta phrases like "this image shows" or "the picture depicts."
- One paragraph only.

Final caption:"""

GENERAL_CHARACTER_PROMPT = """Write one clean caption for a character dataset.

Caption priorities:
- Be specific rather than generic.
- Prioritize visible subject traits, pose, face, hair, eyes, clothing, accessories, colors, lighting, background, framing, and visual style.
- Include visible anatomy and figure information when clear and relevant.
- Do not invent hidden details.
- Avoid meta phrases like "this image shows" or "the picture depicts."
- One paragraph only.

Final caption:"""

DEFAULT_OLLAMA_PROMPT = FEMALE_CHARACTER_LORA_PROMPT

OLLAMA_PROMPT_PRESETS = {
    "female_character_lora": FEMALE_CHARACTER_LORA_PROMPT,
    "general_character": GENERAL_CHARACTER_PROMPT,
    "default_literal": (
        "Describe the image in a highly detailed, literal, and visually grounded way. "
        "Focus only on visible details. Include subject appearance, clothing, pose, "
        "body position, hands, facial expression, hairstyle, accessories, lighting, "
        "background, textures, colors, and spatial relationships. Write a dense "
        "descriptive prompt suitable for image captioning. Avoid speculation, avoid "
        "backstory, avoid opinions, and avoid mentioning things not clearly visible."
    ),
    "dense_lora_literal": (
        "Describe the image as a dense, literal LoRA dataset caption. Focus only on visible facts: "
        "subject, clothing, body position, pose, hands, face, hair, expression, accessories, background, "
        "composition, lighting, colors, textures, and spatial relationships. Do not roleplay, continue "
        "a conversation, add safety commentary, infer backstory, or mention anything not visible."
    ),
    "concise_literal": (
        "Describe the image literally and concisely. Include the main visible subject, pose, clothing, "
        "facial expression, hairstyle, background, lighting, and important visual details. Avoid speculation, "
        "backstory, opinions, and conversation."
    ),
}



@dataclass
class OllamaCaptionRecord:
    image: str
    caption: str
    raw_caption: str
    model_name: str
    model_path: str = ""
    prompt: str = ""
    system_prompt: str = ""
    seed: int = -1
    temperature: float = 0.18
    top_p: float = 0.92
    top_k: int = 60
    repetition_penalty: float = 1.03
    max_new_tokens: int = 1800
    max_size: int = 1024
    timestamp: str = ""
    captionforge_pass: str = "A"
    model_family: str = "ollama"
    ensemble_run_index: int = 0
    image_key: str = ""
    backend: str = "ollama"
    status: str = "ok"


_MODEL_DROPDOWNS = load_ollama_model_dropdowns(__file__)

OLLAMA_MODEL_CHOICES = _MODEL_DROPDOWNS["caption_models"]
DEFAULT_OLLAMA_CAPTION_MODEL = _MODEL_DROPDOWNS["caption_default"]
OLLAMA_MODEL_CONFIG_PATH = Path(_MODEL_DROPDOWNS["config_path"])


def _persist_caption_model_if_possible(model_tag: str) -> None:
    """Best-effort nice-to-have: add a successfully used custom tag to caption_models."""
    model_tag = str(model_tag or "").strip()
    if not model_tag or model_tag.lower() == CUSTOM_VALUE:
        return

    path = OLLAMA_MODEL_CONFIG_PATH

    try:
        data: dict[str, Any] = {}
        if path.exists():
            loaded = json.loads(path.read_text(encoding="utf-8"))
            if isinstance(loaded, dict):
                data = loaded

        models = data.get("caption_models")
        if not isinstance(models, list):
            models = [
                str(x).strip()
                for x in OLLAMA_MODEL_CHOICES
                if str(x).strip() and str(x).strip().lower() != CUSTOM_VALUE
            ]

        models = [
            str(x).strip()
            for x in models
            if str(x).strip() and str(x).strip().lower() != CUSTOM_VALUE
        ]

        if model_tag in models:
            return

        models.append(model_tag)
        data["caption_models"] = models
        data.setdefault("include_custom", True)

        defaults = data.get("defaults")
        if not isinstance(defaults, dict):
            defaults = {}
            data["defaults"] = defaults

        defaults.setdefault("caption_model", DEFAULT_OLLAMA_CAPTION_MODEL)

        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(data, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        print(f"[JLC CaptionForge Ollama Caption] Added '{model_tag}' to {path}")

    except Exception as exc:
        print(
            f"[JLC CaptionForge Ollama Caption] Could not persist model tag "
            f"'{model_tag}' to config: {exc}"
        )


def _resolve_model_tag(model: str, custom_model_tag: str) -> str:
    selected = str(model or "").strip()
    custom = str(custom_model_tag or "").strip()

    if selected.lower() == CUSTOM_VALUE:
        return custom or DEFAULT_OLLAMA_CAPTION_MODEL

    return selected or custom or DEFAULT_OLLAMA_CAPTION_MODEL


def _normalize_ollama_url(value: str) -> str:
    url = str(value or DEFAULT_OLLAMA_URL).strip() or DEFAULT_OLLAMA_URL
    return url.rstrip("/")


def _http_json(method: str, url: str, payload: dict[str, Any] | None = None, timeout: float = 30.0) -> dict[str, Any]:
    data = None
    headers = {"Accept": "application/json"}
    if payload is not None:
        data = json.dumps(payload).encode("utf-8")
        headers["Content-Type"] = "application/json"

    req = urllib.request.Request(url, data=data, headers=headers, method=method.upper())
    try:
        with urllib.request.urlopen(req, timeout=float(timeout)) as resp:
            raw = resp.read().decode("utf-8", errors="replace")
            return json.loads(raw) if raw.strip() else {}
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace") if exc.fp else ""
        raise RuntimeError(f"HTTP {exc.code} from {url}: {body or exc.reason}") from exc
    except urllib.error.URLError as exc:
        reason = getattr(exc, "reason", exc)
        raise RuntimeError(f"Could not reach Ollama at {url}: {reason}") from exc


def _ollama_tags(ollama_url: str, timeout: float) -> list[str]:
    data = _http_json("GET", f"{ollama_url}/api/tags", timeout=timeout)
    models = data.get("models", []) if isinstance(data, dict) else []
    names: list[str] = []
    if isinstance(models, list):
        for item in models:
            if isinstance(item, dict):
                name = str(item.get("name") or item.get("model") or "").strip()
                if name:
                    names.append(name)
    return names


def _is_model_installed(ollama_url: str, model_tag: str, timeout: float) -> bool:
    installed = _ollama_tags(ollama_url, timeout=timeout)
    return model_tag in installed


def _split_ollama_model_tag(model_tag: str) -> tuple[str, str]:
    text = str(model_tag or "").strip()
    if ":" in text:
        name, tag = text.rsplit(":", 1)
        return name.strip(), tag.strip() or "latest"
    return text, "latest"


def _registry_manifest_url(model_tag: str) -> str | None:
    name, tag = _split_ollama_model_tag(model_tag)
    if not name:
        return None
    repo = name if "/" in name else f"library/{name}"
    # Preserve slash separators while URL-encoding each component.
    repo_url = "/".join(urllib.parse.quote(part, safe="") for part in repo.split("/"))
    tag_url = urllib.parse.quote(tag, safe="")
    return f"https://registry.ollama.ai/v2/{repo_url}/manifests/{tag_url}"


def _check_remote_registry_manifest(model_tag: str, timeout: float = 15.0) -> tuple[bool | None, str]:
    """Best-effort remote existence probe without pulling model blobs.

    Returns:
        (True, msg)      remote manifest appears to exist
        (False, msg)     remote manifest appears not to exist
        (None, msg)      probe could not determine remote availability
    """
    url = _registry_manifest_url(model_tag)
    if not url:
        return None, "No model tag was supplied."

    headers = {
        "Accept": (
            "application/vnd.docker.distribution.manifest.v2+json, "
            "application/vnd.oci.image.manifest.v1+json, application/json"
        )
    }

    for method in ("HEAD", "GET"):
        req = urllib.request.Request(url, headers=headers, method=method)
        try:
            with urllib.request.urlopen(req, timeout=float(timeout)) as resp:
                if 200 <= int(resp.status) < 300:
                    return True, f"Remote Ollama registry manifest exists for '{model_tag}'."
        except urllib.error.HTTPError as exc:
            if exc.code == 404:
                return False, f"Remote Ollama registry does not have manifest '{model_tag}'. Check colon/hyphen/case."
            if exc.code in {401, 403}:
                return None, f"Remote registry returned HTTP {exc.code}; existence is indeterminate."
            last = f"Remote registry returned HTTP {exc.code}: {exc.reason}"
        except Exception as exc:
            last = f"Remote registry probe failed: {exc}"

    return None, last


def _probe_ollama_model(ollama_url: str, model_tag: str, timeout: float) -> str:
    print(f"[JLC CaptionForge Ollama Caption] Probing Ollama at {ollama_url}")
    try:
        installed = _ollama_tags(ollama_url, timeout=timeout)
    except Exception as exc:
        msg = (
            f"[JLC CaptionForge Ollama Caption] Ollama probe failed. Is Ollama installed and serving at "
            f"{ollama_url}? Error: {exc}"
        )
        print(msg)
        return msg

    if model_tag in installed:
        msg = f"[JLC CaptionForge Ollama Caption] OK: '{model_tag}' is installed in local Ollama."
        print(msg)
        return msg

    remote_ok, remote_msg = _check_remote_registry_manifest(model_tag, timeout=min(float(timeout), 20.0))
    if remote_ok is True:
        msg = (
            f"[JLC CaptionForge Ollama Caption] Ollama is reachable. '{model_tag}' is not installed locally, "
            f"but the remote tag appears to exist. Disable download_probe_only to pull and caption."
        )
    elif remote_ok is False:
        msg = (
            f"[JLC CaptionForge Ollama Caption] Ollama is reachable, but '{model_tag}' is not installed locally "
            f"and the remote tag probe says it does not exist. {remote_msg}"
        )
    else:
        msg = (
            f"[JLC CaptionForge Ollama Caption] Ollama is reachable. '{model_tag}' is not installed locally. "
            f"Remote tag existence could not be confirmed. {remote_msg}"
        )
    print(msg)
    return msg


def _pull_ollama_model(ollama_url: str, model_tag: str, timeout: float) -> None:
    print(f"[JLC CaptionForge Ollama Caption] Pulling Ollama model '{model_tag}' from {ollama_url} ...")
    payload = {"name": model_tag, "stream": True}
    req = urllib.request.Request(
        f"{ollama_url}/api/pull",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json", "Accept": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=float(timeout)) as resp:
            for raw_line in resp:
                line = raw_line.decode("utf-8", errors="replace").strip()
                if not line:
                    continue
                try:
                    item = json.loads(line)
                except Exception:
                    print(f"[JLC CaptionForge Ollama Caption] pull: {line}")
                    continue
                if "error" in item:
                    raise RuntimeError(str(item.get("error")))
                status = str(item.get("status") or "").strip()
                digest = str(item.get("digest") or "").strip()
                completed = item.get("completed")
                total = item.get("total")
                if status:
                    if isinstance(completed, int) and isinstance(total, int) and total > 0:
                        pct = 100.0 * completed / total
                        print(f"[JLC CaptionForge Ollama Caption] pull {status}: {pct:.1f}% {digest}")
                    else:
                        print(f"[JLC CaptionForge Ollama Caption] pull: {status} {digest}".rstrip())
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace") if exc.fp else ""
        raise RuntimeError(f"Ollama pull failed for '{model_tag}' with HTTP {exc.code}: {body or exc.reason}") from exc
    except urllib.error.URLError as exc:
        reason = getattr(exc, "reason", exc)
        raise RuntimeError(f"Ollama pull failed because Ollama is unreachable at {ollama_url}: {reason}") from exc


def _ensure_ollama_model(ollama_url: str, model_tag: str, timeout: float) -> None:
    try:
        if _is_model_installed(ollama_url, model_tag, timeout=min(float(timeout), 30.0)):
            return
    except Exception as exc:
        msg = (
            f"[JLC CaptionForge Ollama Caption] Could not contact Ollama before captioning. "
            f"Is Ollama installed and serving at {ollama_url}? Error: {exc}"
        )
        print(msg)
        raise RuntimeError(msg) from exc

    remote_ok, remote_msg = _check_remote_registry_manifest(model_tag, timeout=20.0)
    if remote_ok is False:
        msg = f"[JLC CaptionForge Ollama Caption] Refusing to pull likely-invalid model tag '{model_tag}'. {remote_msg}"
        print(msg)
        raise RuntimeError(msg)
    if remote_ok is None:
        print(
            f"[JLC CaptionForge Ollama Caption] Remote tag probe was indeterminate for '{model_tag}'. "
            f"Proceeding with Ollama pull anyway. {remote_msg}"
        )

    _pull_ollama_model(ollama_url, model_tag, timeout=max(float(timeout), 60.0))


def _ollama_options(
    *,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    top_k: int,
    repetition_penalty: float,
    seed: int,
) -> dict[str, Any]:
    """Build Ollama generation options.

    CaptionForge keeps the public widget name max_new_tokens for consistency
    with the Python caption nodes and Pipeline Planner. Ollama receives the
    same value as num_predict.
    """
    options: dict[str, Any] = {
        "num_predict": int(max_new_tokens),
        "temperature": float(temperature),
        "top_p": float(top_p),
        "top_k": int(top_k),
        "repeat_penalty": float(repetition_penalty),
    }
    if int(seed) >= 0:
        options["seed"] = int(seed)
    return options


def _extract_ollama_text(data: Any) -> str:
    if not isinstance(data, dict):
        return ""

    # /api/chat canonical shape: {"message": {"role": "assistant", "content": "..."}}
    message = data.get("message")
    if isinstance(message, dict):
        text = str(message.get("content") or "").strip()
        if text:
            return text

    # /api/generate canonical shape: {"response": "..."}
    text = str(data.get("response") or "").strip()
    if text:
        return text

    return ""


def _summarize_ollama_response(data: Any) -> str:
    if not isinstance(data, dict):
        return f"type={type(data).__name__}"
    parts = [f"keys={sorted(data.keys())}"]
    if data.get("done_reason"):
        parts.append(f"done_reason={data.get('done_reason')}")
    if isinstance(data.get("message"), dict):
        msg = data["message"]
        parts.append(f"message_keys={sorted(msg.keys())}")
        content = str(msg.get("content") or "")
        parts.append(f"message_content_len={len(content)}")
    response = str(data.get("response") or "")
    parts.append(f"response_len={len(response)}")
    if data.get("eval_count") is not None:
        parts.append(f"eval_count={data.get('eval_count')}")
    if data.get("prompt_eval_count") is not None:
        parts.append(f"prompt_eval_count={data.get('prompt_eval_count')}")
    return "; ".join(parts)


def _ollama_chat_caption(
    *,
    ollama_url: str,
    model_tag: str,
    system_prompt: str,
    prompt: str,
    image_b64: str,
    options: dict[str, Any],
    keep_loaded: bool,
    timeout: float,
) -> tuple[str, dict[str, Any]]:
    payload: dict[str, Any] = {
        "model": model_tag,
        "stream": False,
        "messages": [
            {"role": "system", "content": str(system_prompt or "")},
            {"role": "user", "content": str(prompt or ""), "images": [image_b64]},
        ],
        "think": False,
        "options": options,
        "keep_alive": "5m" if bool(keep_loaded) else "0s",
    }
    data = _http_json("POST", f"{ollama_url}/api/chat", payload=payload, timeout=timeout)
    return _extract_ollama_text(data), data


def _ollama_generate_caption_fallback(
    *,
    ollama_url: str,
    model_tag: str,
    system_prompt: str,
    prompt: str,
    image_b64: str,
    options: dict[str, Any],
    keep_loaded: bool,
    timeout: float,
) -> tuple[str, dict[str, Any]]:
    # Mirrors the known-good PS1 fallback shape: system + user prompt combined
    # into one prompt, image attached at top level.
    system = str(system_prompt or "").strip()
    user = str(prompt or "").strip()
    full_prompt = f"{system}\n\n{user}".strip() if system else user
    payload: dict[str, Any] = {
        "model": model_tag,
        "prompt": full_prompt,
        "stream": False,
        "images": [image_b64],
        "think": False,
        "options": options,
        "keep_alive": "5m" if bool(keep_loaded) else "0s",
    }
    data = _http_json("POST", f"{ollama_url}/api/generate", payload=payload, timeout=timeout)
    return _extract_ollama_text(data), data


def _ollama_generate_caption(
    *,
    ollama_url: str,
    model_tag: str,
    system_prompt: str,
    prompt: str,
    pil_image: Image.Image,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    top_k: int,
    repetition_penalty: float,
    seed: int,
    max_size: int,
    keep_loaded: bool,
    timeout: float,
) -> str:
    image_b64 = _pil_to_base64_png(pil_image, max_size=max_size)
    options = _ollama_options(
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        repetition_penalty=repetition_penalty,
        seed=seed,
    )

    chat_error: Exception | None = None
    chat_data: dict[str, Any] | None = None
    try:
        text, chat_data = _ollama_chat_caption(
            ollama_url=ollama_url,
            model_tag=model_tag,
            system_prompt=system_prompt,
            prompt=prompt,
            image_b64=image_b64,
            options=options,
            keep_loaded=keep_loaded,
            timeout=timeout,
        )
        if text:
            return text
        print(
            f"[JLC CaptionForge Ollama Caption] /api/chat returned empty text for '{model_tag}'. "
            f"Trying /api/generate fallback. {_summarize_ollama_response(chat_data)}"
        )
    except Exception as exc:
        chat_error = exc
        print(
            f"[JLC CaptionForge Ollama Caption] /api/chat failed for '{model_tag}': {exc}. "
            "Trying /api/generate fallback."
        )

    generate_error: Exception | None = None
    generate_data: dict[str, Any] | None = None
    try:
        text, generate_data = _ollama_generate_caption_fallback(
            ollama_url=ollama_url,
            model_tag=model_tag,
            system_prompt=system_prompt,
            prompt=prompt,
            image_b64=image_b64,
            options=options,
            keep_loaded=keep_loaded,
            timeout=timeout,
        )
        if text:
            return text
    except Exception as exc:
        generate_error = exc

    diagnostics = [
        f"Ollama returned an empty caption for '{model_tag}' via both /api/chat and /api/generate.",
        f"CaptionForge max_new_tokens maps to Ollama num_predict; current value: {int(max_new_tokens)}.",
    ]
    if chat_error is not None:
        diagnostics.append(f"chat_error={chat_error}")
    elif chat_data is not None:
        diagnostics.append(f"chat_response: {_summarize_ollama_response(chat_data)}")
    if generate_error is not None:
        diagnostics.append(f"generate_error={generate_error}")
    elif generate_data is not None:
        diagnostics.append(f"generate_response: {_summarize_ollama_response(generate_data)}")

    done_reasons = []
    for data in (chat_data, generate_data):
        if isinstance(data, dict) and data.get("done_reason"):
            done_reasons.append(str(data.get("done_reason")))
    if "length" in done_reasons:
        diagnostics.append("One endpoint stopped with done_reason=length. Increase max_new_tokens.")

    raise RuntimeError(" ".join(diagnostics))


def _resolve_prompt_preset(prompt_preset: str, custom_prompt: str) -> str:
    text = str(custom_prompt or "").strip()
    if text:
        return text
    return OLLAMA_PROMPT_PRESETS.get(str(prompt_preset or ""), DEFAULT_OLLAMA_PROMPT)


def _format_resolved_prompt(system_prompt: str, prompt: str) -> str:
    system = str(system_prompt or "").strip()
    user = str(prompt or "").strip()
    if not system:
        return user
    return f"SYSTEM:\n{system}\n\nUSER:\n{user}"


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


def _pil_for_model(pil: Image.Image, max_size: int) -> Image.Image:
    img = pil.convert("RGB")
    max_size = int(max_size or 0)
    if max_size > 0:
        w, h = img.size
        longest = max(w, h)
        if longest > max_size:
            scale = max_size / float(longest)
            new_size = (max(1, int(round(w * scale))), max(1, int(round(h * scale))))
            img = img.resize(new_size, Image.Resampling.LANCZOS)
    return img


def _pil_to_base64_png(pil: Image.Image, max_size: int) -> str:
    img = _pil_for_model(pil, max_size=max_size)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("ascii")


def _safe_source_name(value: str) -> str:
    cleaned = []
    for ch in value.replace("\\", "/"):
        if ch.isalnum() or ch in {"-", "_", "."}:
            cleaned.append(ch)
        elif ch == "/":
            cleaned.append("__")
        else:
            cleaned.append("_")
    out = "".join(cleaned).strip("._")
    return out or "image"


def _iter_input_path_images(input_path: str, recursive: bool, filename_glob: str) -> list[tuple[str, Path]]:
    root = Path(str(input_path or "").strip())
    if not root:
        return []
    if not root.exists():
        raise RuntimeError(f"CaptionForge input_path does not exist: {root}")

    glob_text = (filename_glob or "*").strip() or "*"

    if root.is_file():
        if root.suffix.lower() not in _SUPPORTED_IMAGE_SUFFIXES:
            raise RuntimeError(f"CaptionForge input_path is not a supported image file: {root}")
        return [(_safe_source_name(root.stem), root)]

    pattern_iter = root.rglob(glob_text) if recursive else root.glob(glob_text)
    paths = sorted(
        p for p in pattern_iter
        if p.is_file() and p.suffix.lower() in _SUPPORTED_IMAGE_SUFFIXES
    )

    items: list[tuple[str, Path]] = []
    for path in paths:
        rel = path.relative_to(root).with_suffix("")
        items.append((_safe_source_name(str(rel)), path))
    return items


def _open_image(path: Path) -> Image.Image:
    with Image.open(path) as im:
        return im.convert("RGB")


def _normalize_pipeline_plan(config):
    if config is None:
        return {}
    if isinstance(config, dict):
        return dict(config)
    if isinstance(config, str):
        text = config.strip()
        if not text:
            return {}
        try:
            obj = json.loads(text)
            return obj if isinstance(obj, dict) else {}
        except Exception:
            return {}
    return {}


def _planned_caption_jsonl_path(pipeline_plan) -> Path | None:
    """Return Planner-owned shared Pass A JSONL path, if present."""
    cfg = _normalize_pipeline_plan(pipeline_plan)
    paths = cfg.get("paths") if isinstance(cfg.get("paths"), dict) else {}
    for key in ("caption_jsonl", "pass_a_jsonl"):
        value = str(paths.get(key) or "").strip()
        if value:
            return Path(value)
    return None


def _run_txt_path(output_dir: Path, source_name: str, run_count: int, run_index: int) -> Path:
    if run_count <= 1:
        return output_dir / f"{source_name}.txt"
    return output_dir / f"{source_name}__cf_run_{run_index:02d}.txt"


def _use_template_mode(caption_template_mode: bool, custom_prompt_mode: bool) -> bool:
    if bool(custom_prompt_mode):
        return False
    if bool(caption_template_mode):
        return True
    print("[CaptionForge] No caption mode was selected; falling back to caption_template_mode.")
    return True


def _resolve_template_options(template_options):
    options, name_input, metadata = resolve_effective_extra_options(
        payload=template_options,
        local_options=[],
        local_name="",
    )
    return options, name_input, metadata


def _parse_forbidden_lines(value: str) -> list[str]:
    value = (value or "").strip()
    if not value:
        return []
    return [line.strip() for line in value.splitlines() if line.strip()]


def _parse_replace_pairs(value: str) -> list[tuple[str, str]]:
    """Parse ComfyUI-friendly replacement rules in old=>new form."""
    value = (value or "").strip()
    if not value:
        return []

    rules: list[tuple[str, str]] = []
    for line in value.splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=>" not in line:
            continue
        old, new = line.split("=>", 1)
        old = old.strip()
        new = new.strip()
        if old:
            rules.append((old, new))
    return rules


def _clean_caption(
    raw: str,
    *,
    trigger_word: str,
    forbidden_phrases: list[str],
    replacement_rules: list[tuple[str, str]],
) -> tuple[str, str]:
    text = str(raw or "").strip().strip('"').strip()
    for old, new in replacement_rules:
        text = text.replace(old, new)

    if forbidden_phrases:
        kept: list[str] = []
        for line in text.splitlines() or [text]:
            lowered = line.lower()
            if any(phrase.lower() in lowered for phrase in forbidden_phrases if phrase):
                continue
            kept.append(line)
        text = "\n".join(line.strip() for line in kept if line.strip()).strip()

    trigger = str(trigger_word or "").strip().strip(",")
    if trigger and text and not text.lower().startswith(trigger.lower()):
        text = f"{trigger}, {text}"

    return text, "ok" if text else "filtered"


def _append_jsonl_records(path: Path, records: list[OllamaCaptionRecord]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(asdict(record), ensure_ascii=False) + "\n")


def _write_text_sidecar(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(str(text or ""), encoding="utf-8")


def _write_run_config_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def _build_run_config(
    *,
    model_tag: str,
    ollama_url: str,
    system_prompt: str,
    prompt: str,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    top_k: int,
    repetition_penalty: float,
    max_size: int,
) -> dict[str, Any]:
    return {
        "backend": "ollama",
        "model_name": model_tag,
        "ollama_url": ollama_url,
        "system_prompt": system_prompt,
        "prompt": prompt,
        "generation": {
            "max_new_tokens": int(max_new_tokens),
            "temperature": float(temperature),
            "top_p": float(top_p),
            "top_k": int(top_k),
            "repetition_penalty": float(repetition_penalty),
            "max_size": int(max_size),
        },
        "timestamp": datetime.now().isoformat(timespec="seconds"),
    }


def _expand_ollama_runs_compat(
    pipeline_plan,
    *,
    widget_seed: int,
    widget_temperature: float,
    widget_top_p: float,
    widget_top_k: int,
    widget_max_new_tokens: int,
    widget_max_size: int,
):
    """Expand Planner runs while tolerating future key-name choices.

    The current Planner may not know about an Ollama caption family yet. Future
    Planner revisions can choose one of these keys without requiring this node to
    change immediately.
    """
    keys = ["ollama", "ollama_caption", "caption_ollama", "ollama_vlm"]
    last_error: Exception | None = None
    for key in keys:
        try:
            run_plan = expand_captionforge_runs(
                pipeline_plan,
                model_key=key,
                widget_captions_per_image=1,
                widget_seed=widget_seed,
                widget_temperature=widget_temperature,
                widget_top_p=widget_top_p,
                widget_top_k=widget_top_k,
                widget_max_new_tokens=widget_max_new_tokens,
                widget_max_size=widget_max_size,
                widget_trigger_word="",
                widget_output_dir="",
                widget_input_path="",
                widget_recursive=True,
                widget_filename_glob="*",
            )
            if run_plan:
                return run_plan, key
        except Exception as exc:
            last_error = exc
            continue

    if pipeline_plan and last_error is not None:
        print(f"[JLC CaptionForge Ollama Caption] Could not expand Planner runs for Ollama: {last_error}")
    return [], "ollama"


class JLC_CaptionForgeOllamaCaption:
    """Canonical Ollama-backed image Caption node for CaptionForge."""

    @classmethod
    def INPUT_TYPES(cls):
        caption_type_default = (
            "LoRA Literal"
            if "LoRA Literal" in CAPTION_TYPE_CHOICES
            else CAPTION_TYPE_CHOICES[0]
        )
        prompt_default = "female_character_lora"

        return {
            "required": {
                "model": (
                    OLLAMA_MODEL_CHOICES,
                    {
                        "default": DEFAULT_OLLAMA_CAPTION_MODEL,
                        "tooltip": (
                            "Ollama vision-language model tag. Choices are loaded from "
                            "config/captionforge_ollama_models.json -> caption_models. "
                            "Use custom for any locally installed or pullable Ollama model tag."
                        ),
                    },
                ),
                "custom_model_tag": (
                    "STRING",
                    {
                        "default": "",
                        "multiline": False,
                        "tooltip": "Used only when model is custom. Example: gemma4:12b or qwen3.6:35B-A3B.",
                    },
                ),
                "ollama_url": (
                    "STRING",
                    {
                        "default": DEFAULT_OLLAMA_URL,
                        "multiline": False,
                        "tooltip": "Local Ollama server URL. Default is http://127.0.0.1:11434.",
                    },
                ),
                "keep_loaded": (
                    "BOOLEAN",
                    {
                        "default": True,
                        "tooltip": (
                            "Ask Ollama to keep the model warm after generation. This is passed as keep_alive. "
                            "Ollama ultimately owns model residency."
                        ),
                    },
                ),

                "caption_template_mode": (
                    "BOOLEAN",
                    {
                        "default": True,
                        "tooltip": (
                            "Use the structured CaptionForge template path: caption_type, caption_length, "
                            "and optional Template Options from the template_options pin. If custom_prompt_mode "
                            "is also enabled, custom_prompt_mode takes precedence."
                        ),
                    },
                ),
                "caption_type": (
                    CAPTION_TYPE_CHOICES,
                    {
                        "default": caption_type_default,
                        "tooltip": "Caption template style used when caption_template_mode is active.",
                    },
                ),
                "caption_length": (
                    CAPTION_LENGTH_CHOICES,
                    {
                        "default": "any",
                        "tooltip": "Target caption length used when caption_template_mode is active.",
                    },
                ),

                "custom_prompt_mode": (
                    "BOOLEAN",
                    {
                        "default": False,
                        "tooltip": (
                            "Use custom_prompt when non-empty, otherwise use prompt_preset. "
                            "This overrides caption_template_mode when both toggles are enabled."
                        ),
                    },
                ),
                "prompt_preset": (
                    list(OLLAMA_PROMPT_PRESETS.keys()),
                    {
                        "default": prompt_default,
                        "tooltip": "Built-in prompt preset used only in custom_prompt_mode when custom_prompt is blank. Default is female_character_lora.",
                    },
                ),
                "system_prompt": (
                    "STRING",
                    {
                        "default": DEFAULT_SYSTEM_PROMPT,
                        "multiline": True,
                        "tooltip": (
                            "Ollama system prompt. Kept next to custom_prompt because both control "
                            "the instruction envelope. Prepopulated with the default female-character LoRA "
                            "captioning system prompt. Pipeline Planner does not currently override this."
                        ),
                    },
                ),
                "custom_prompt": (
                    "STRING",
                    {
                        "default": DEFAULT_OLLAMA_PROMPT,
                        "multiline": True,
                        "tooltip": (
                            "Custom prompt used only when custom_prompt_mode is enabled. The widget is prepopulated "
                            "with the default LoRA-oriented prompt; if cleared, prompt_preset is used instead."
                        ),
                    },
                ),

                "max_new_tokens": (
                    "INT",
                    {
                        "default": 1800,
                        "min": 16,
                        "max": 8192,
                        "step": 8,
                        "tooltip": (
                            "Standalone token budget. For this Ollama-backed node, CaptionForge max_new_tokens "
                            "is sent to Ollama as num_predict. When a Pipeline Planner is connected, this is "
                            "overridden by the Planner's shared max_new_tokens."
                        ),
                    },
                ),
                "temperature": (
                    "FLOAT",
                    {
                        "default": 0.18,
                        "min": 0.0,
                        "max": 2.0,
                        "step": 0.01,
                        "tooltip": (
                            "Standalone sampling temperature. When a Pipeline Planner is connected, "
                            "this is overridden by the Planner temperature schedule."
                        ),
                    },
                ),
                "top_p": (
                    "FLOAT",
                    {
                        "default": 0.92,
                        "min": 0.0,
                        "max": 1.0,
                        "step": 0.01,
                        "tooltip": (
                            "Standalone top-p sampling value. When a Pipeline Planner is connected, "
                            "this is overridden by the Planner top-p schedule."
                        ),
                    },
                ),
                "top_k": (
                    "INT",
                    {
                        "default": 60,
                        "min": 0,
                        "max": 500,
                        "step": 1,
                        "tooltip": (
                            "Standalone top-k sampling limit. When a Pipeline Planner is connected, "
                            "this is overridden by the Planner top-k schedule."
                        ),
                    },
                ),
                "repetition_penalty": (
                    "FLOAT",
                    {
                        "default": 1.03,
                        "min": 1.0,
                        "max": 2.0,
                        "step": 0.01,
                        "tooltip": (
                            "Ollama repeat_penalty. Kept with the core captioning parameters. "
                            "This is not currently overridden by the Pipeline Planner."
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
                            "Maximum longest-side image size for standalone captioning. The image is resized "
                            "in memory only. Pipeline Planner overrides this in planned runs."
                        ),
                    },
                ),
                "request_timeout_seconds": (
                    "INT",
                    {
                        "default": 900,
                        "min": 10,
                        "max": 7200,
                        "step": 10,
                        "advanced": True,
                        "tooltip": "HTTP timeout for Ollama API calls, including large model pulls/generation.",
                    },
                ),

                "forbidden_phrases": (
                    "STRING",
                    {
                        "default": "",
                        "multiline": True,
                        "advanced": True,
                        "tooltip": "Optional cleanup filter: remove lines/captions containing any listed phrase, one per line.",
                    },
                ),
                "replace_pairs": (
                    "STRING",
                    {
                        "default": "",
                        "multiline": True,
                        "advanced": True,
                        "tooltip": "Optional cleanup replacements, one per line: old=>new.",
                    },
                ),
                "download_probe_only": (
                    "BOOLEAN",
                    {
                        "default": False,
                        "tooltip": (
                            "At the very bottom by design. Check Ollama availability, local model installation, "
                            "and best-effort remote tag existence, then return a status message without captioning."
                        ),
                    },
                ),
            },
            "optional": {
                "image": (
                    "IMAGE",
                    {
                        "tooltip": (
                            "Image or batch of images to caption. The image is passed through unchanged "
                            "for clean node-to-node pipeline chaining."
                        ),
                    },
                ),
                "pipeline_plan": (
                    "CAPTIONFORGE_PIPELINE_PLAN",
                    {
                        "tooltip": (
                            "Connect the CaptionForge Pipeline Planner output here. When connected, this "
                            "node switches into Pass A evidence mode: Planner image routing, per-run seeds, "
                            "sampling schedules, shared output paths, and internal JSONL evidence append."
                        ),
                    },
                ),
                "template_options": (
                    "CAPTIONFORGE_EXTRA_OPTIONS",
                    {
                        "tooltip": (
                            "Connect the CaptionForge Template Options node here. Works in standalone and "
                            "Pipeline modes. This is the only source for template modifiers and name input."
                        ),
                    },
                ),
                "seed": (
                    "INT",
                    {
                        "forceInput": True,
                        "tooltip": "Optional standalone seed input. Ignored when a Pipeline Planner supplies a seed schedule.",
                    },
                ),
            },
        }

    RETURN_TYPES = ("IMAGE", "CAPTIONFORGE_PIPELINE_PLAN", "STRING", "STRING")
    RETURN_NAMES = ("image_out", "pipeline_plan_out", "caption", "resolved_prompt")
    FUNCTION = "caption"
    CATEGORY = "Captioning/CaptionForge/Caption Nodes"

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        return float("NaN")

    def caption(
        self,
        model,
        custom_model_tag,
        ollama_url,
        keep_loaded,
        caption_template_mode,
        caption_type,
        caption_length,
        custom_prompt_mode,
        prompt_preset,
        system_prompt,
        custom_prompt,
        max_new_tokens,
        temperature,
        top_p,
        top_k,
        repetition_penalty,
        max_size,
        request_timeout_seconds,
        forbidden_phrases,
        replace_pairs,
        download_probe_only,
        image=None,
        pipeline_plan=None,
        template_options=None,
        seed=None,
    ):
        model_tag = _resolve_model_tag(model, custom_model_tag)
        ollama_url = _normalize_ollama_url(ollama_url)
        timeout = float(request_timeout_seconds)

        run_plan_connected = bool(pipeline_plan)
        use_caption_template = _use_template_mode(caption_template_mode, custom_prompt_mode)
        effective_extra_options, effective_person_name, _ = _resolve_template_options(template_options)

        if use_caption_template:
            prompt = build_caption_prompt(
                caption_type=caption_type,
                caption_length=caption_length,
                extra_options=effective_extra_options,
                name_input=effective_person_name,
                dialect="qwen",
            )
        else:
            prompt = _resolve_prompt_preset(prompt_preset, custom_prompt)

        resolved_prompt = _format_resolved_prompt(system_prompt, prompt)

        _evict_python_models_before_ollama_if_needed("JLC CaptionForge Ollama Caption")

        if download_probe_only:
            result = _probe_ollama_model(ollama_url, model_tag, timeout=min(timeout, 60.0))
            return (image, pipeline_plan, result, resolved_prompt)

        _ensure_ollama_model(ollama_url, model_tag, timeout=timeout)
        _persist_caption_model_if_possible(model_tag)

        effective_seed = -1 if seed is None else int(seed)

        run_plan, planner_key = _expand_ollama_runs_compat(
            pipeline_plan,
            widget_seed=effective_seed,
            widget_temperature=float(temperature),
            widget_top_p=float(top_p),
            widget_top_k=int(top_k),
            widget_max_new_tokens=int(max_new_tokens),
            widget_max_size=int(max_size),
        )

        if run_plan_connected and not run_plan:
            status = "[CaptionForge] Ollama Caption disabled by Pipeline Planner or Planner has no Ollama caption count yet."
            print(status)
            return (image, pipeline_plan, status, resolved_prompt)

        if not run_plan:
            # Defensive fallback for standalone mode if expand_captionforge_runs ever changes behavior.
            class _StandaloneRun:
                pass

            standalone_run = _StandaloneRun()
            standalone_run.seed = effective_seed
            standalone_run.temperature = float(temperature)
            standalone_run.top_p = float(top_p)
            standalone_run.top_k = int(top_k)
            standalone_run.max_new_tokens = int(max_new_tokens)
            standalone_run.max_size = int(max_size)
            standalone_run.trigger_word = ""
            standalone_run.output_dir = ""
            standalone_run.input_path = ""
            standalone_run.recursive = True
            standalone_run.filename_glob = "*"
            standalone_run.ensemble_run_index = 0
            run_plan = [standalone_run]

        first_run = run_plan[0]

        direct_images = [(f"comfy_image_{i:04d}", pil) for i, pil in enumerate(_tensor_to_pil(image))]
        file_images: list[tuple[str, Path]] = []
        if getattr(first_run, "input_path", ""):
            file_images = _iter_input_path_images(first_run.input_path, first_run.recursive, first_run.filename_glob)

        if direct_images and file_images:
            print(
                "[CaptionForge] IMAGE input and Pipeline Planner input_path are both active; "
                "captioning both sources."
            )

        if not direct_images and not file_images:
            raise RuntimeError(
                "No image input found. Connect an IMAGE input or provide input_path in the CaptionForge Pipeline Planner."
            )

        output_dir: Path | None = None
        jsonl_path: Path | None = None
        if run_plan_connected:
            planned_caption_jsonl_path = _planned_caption_jsonl_path(pipeline_plan)
            output_dir = (
                planned_caption_jsonl_path.parent
                if planned_caption_jsonl_path is not None
                else Path(first_run.output_dir or (Path(folder_paths.get_output_directory()) / "CaptionForge"))
            )
            output_dir.mkdir(parents=True, exist_ok=True)
            jsonl_path = planned_caption_jsonl_path or (output_dir / DEFAULT_JSONL_FILENAME)

            _write_run_config_json(
                output_dir / f"jlc_captionforge_ollama_caption_run_config_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                _build_run_config(
                    model_tag=model_tag,
                    ollama_url=ollama_url,
                    system_prompt=system_prompt,
                    prompt=prompt,
                    max_new_tokens=int(first_run.max_new_tokens),
                    temperature=float(first_run.temperature),
                    top_p=float(first_run.top_p),
                    top_k=int(first_run.top_k),
                    repetition_penalty=float(repetition_penalty),
                    max_size=int(first_run.max_size),
                ),
            )

        all_records: list[OllamaCaptionRecord] = []
        forbidden = _parse_forbidden_lines(forbidden_phrases)
        replacements = _parse_replace_pairs(replace_pairs)

        def process_one(source_name: str, pil: Image.Image):
            for run in run_plan:
                t0 = time.perf_counter()
                raw_caption = _ollama_generate_caption(
                    ollama_url=ollama_url,
                    model_tag=model_tag,
                    system_prompt=system_prompt,
                    prompt=prompt,
                    pil_image=pil,
                    max_new_tokens=int(run.max_new_tokens),
                    temperature=float(run.temperature),
                    top_p=float(run.top_p),
                    top_k=int(run.top_k),
                    repetition_penalty=float(repetition_penalty),
                    seed=int(run.seed),
                    max_size=int(run.max_size),
                    keep_loaded=bool(keep_loaded),
                    timeout=timeout,
                )
                dt = time.perf_counter() - t0
                print(f"[JLC CaptionForge Ollama Caption] Generation time run {run.ensemble_run_index}: {dt:.2f}s")

                final_caption, status = _clean_caption(
                    raw_caption,
                    trigger_word=getattr(run, "trigger_word", ""),
                    forbidden_phrases=forbidden,
                    replacement_rules=replacements,
                )

                record = OllamaCaptionRecord(
                    image=source_name,
                    caption=final_caption,
                    raw_caption=raw_caption,
                    model_name=model_tag,
                    model_path=model_tag,
                    prompt=prompt,
                    system_prompt=system_prompt,
                    seed=int(run.seed),
                    temperature=float(run.temperature),
                    top_p=float(run.top_p),
                    top_k=int(run.top_k),
                    repetition_penalty=float(repetition_penalty),
                    max_new_tokens=int(run.max_new_tokens),
                    max_size=int(run.max_size),
                    timestamp=datetime.now().isoformat(timespec="seconds"),
                    captionforge_pass="A",
                    model_family="ollama",
                    ensemble_run_index=int(run.ensemble_run_index),
                    image_key=source_name,
                    backend="ollama",
                    status=status,
                )
                all_records.append(record)

                if run_plan_connected and jsonl_path is not None and output_dir is not None:
                    _append_jsonl_records(jsonl_path, [record])
                    _write_text_sidecar(
                        _run_txt_path(output_dir, source_name, len(run_plan), int(run.ensemble_run_index)),
                        record.caption,
                    )
                print(
                    f"[JLC CaptionForge Ollama Caption] Captioned {source_name} "
                    f"run {int(run.ensemble_run_index) + 1}/{len(run_plan)} via Planner key '{planner_key}'"
                )

        for source_name, pil in direct_images:
            process_one(source_name, pil)

        for source_name, path in file_images:
            process_one(source_name, _open_image(path))

        # Ollama model residency is requested through keep_alive on generation calls;
        # Ollama's server/runtime policy ultimately decides residency.
        caption_string = "\n\n".join(r.caption for r in all_records if r.status == "ok" and r.caption)
        return (image, pipeline_plan, caption_string, resolved_prompt)


NODE_CLASS_MAPPINGS = {
    "JLC_CaptionForgeOllamaCaption": JLC_CaptionForgeOllamaCaption,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "JLC_CaptionForgeOllamaCaption": "\u2003JLC CaptionForge Ollama Caption",
}
