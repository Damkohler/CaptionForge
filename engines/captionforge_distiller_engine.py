"""
CaptionForge Distiller Engine

- CaptionForge
  - This engine is part of **CaptionForge**, a model-agnostic captioning
    framework for ComfyUI developed by **J. L. Córdova**.

  - Repository:
    https://github.com/Damkohler/CaptionForge

- CaptionForge focuses on practical dataset-captioning infrastructure for
  LoRA dataset preparation, using multi-engine caption generation, JSONL
  audit trails, claim extraction and refinement, text-LLM distillation,
  image-aware VLM validation, and consensus-oriented caption improvement
  to produce grounded, auditable training captions.

- Engine Purpose
    - The **CaptionForge Distiller Engine** is the Pass B text-LLM
      pollster/copywriter stage of the CaptionForge pipeline.

    - It consumes Pass A raw caption JSONL records from one or more caption
      witness engines, groups records by image, and asks a text LLM to organize
      the captions as imperfect witness ballots.

    - The engine emits:
            • accepted visual claims
            • plausible singleton candidates
            • rejected or unresolved conflicting claims
            • a human-readable claim vote summary
            • a rich narrative caption draft
            • a dense taggy/comma-style caption draft
            • audit metadata and optional raw LLM responses

- CaptionForge Pipeline Role
    - This engine participates in **Pass B_DISTILL**.

    - Pipeline position:

            Pass A raw caption witness JSONL
              -> Pass B_DISTILL text-LLM distillation
              -> Pass C_VLM_VALIDATED image-aware validation
              -> Pass D final TXT/JSONL export

    - Pass B is intentionally recall-oriented. It preserves useful candidate
      details for later image-aware validation instead of aggressively pruning
      the caption down to a short summary.

- Execution Model
    - Primary backend: local Ollama text LLM.

    - Additional supported modes include manual JSON and prompt-only workflows
      for debugging, inspection, or controlled testing.

    - Input records are grouped by `image_key`, and each image group produces
      one distillation record.

    - Trigger words and user caption anchors are treated as training metadata
      and caption guidance, not as facts inferred by the text LLM.

- Design Philosophy
    - The distiller acts first as a ballot pollster, then as a copywriter.

    - Repeated agreement across captions and model families should be promoted.

    - Plausible one-off details should be preserved as singleton candidates when
      they are not contradicted, because the VLM validator will inspect the
      image later.

    - Contradictory, weak, tied, vague, or mutually exclusive details should not
      be promoted as accepted evidence.

    - Draft captions should be rich enough for LoRA dataset preparation while
      remaining auditable through their source claims and metadata.

- Development Status
    - CaptionForge v0.1.0 experimental developer-preview infrastructure.
    - Prompt contracts, audit fields, and parser behavior may evolve before a
      stable CaptionForge release.

- Attribution & License
  - Concept and implementation by **J. L. Córdova**
    with development assistance from **ChatGPT (OpenAI)**.

  - Designed for use with:
    https://github.com/comfyanonymous/ComfyUI

  - Copyright (c) 2026 J. L. Córdova

  - Released under the **MIT License**.
"""

from __future__ import annotations

from ..captionforge_version import CAPTIONFORGE_VERSION

ENGINE_NAME = "captionforge_distiller_engine"
ENGINE_VERSION = CAPTIONFORGE_VERSION
CAPTIONFORGE_PASS = "B_DISTILL"
DEFAULT_OLLAMA_BASE_URL = "http://127.0.0.1:11434"

MANIFEST = {
    "name": "CaptionForge Distiller Engine",
    "version": CAPTIONFORGE_VERSION,
    "author": "J. L. Córdova",
    "description": (
        "CaptionForge Pass B text-LLM distiller. Consumes Pass A caption JSONL "
        "records, groups captions by image, treats them as imperfect witness "
        "ballots, emits accepted claims, singleton candidates, rejected or "
        "unresolved conflicts, and produces rich narrative plus dense taggy "
        "caption drafts for later image-aware VLM validation."
    ),
}


import argparse
import dataclasses
import hashlib
import json
import re
import time
import urllib.error
import urllib.request
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Optional


ENGINE_NAME = "captionforge_distiller_engine"
ENGINE_VERSION = "0.2.0"
CAPTIONFORGE_PASS = "B_DISTILL"
DEFAULT_OLLAMA_BASE_URL = "http://127.0.0.1:11434"

MANIFEST = {
    "name": "CaptionForge Distiller Engine",
    "version": ENGINE_VERSION,
    "author": "J. L. Córdova",
    "description": (
        "CLI-first CaptionForge Pass B pollster/copywriter engine. Consumes Pass A "
        "caption JSONL records, groups captions by image, asks a text LLM to "
        "vote/organize visual claims, and emits accepted evidence, singleton "
        "candidates, rejected conflicts, and rich/taggy draft captions."
    ),
}

DEFAULT_DISTILLER_INSTRUCTIONS = (
    "You are CaptionForge Pass B: a caption ballot pollster and rich-caption copywriter. "
    "Your job is not to summarize, prune for brevity, or compress the source captions. "
    "All source captions describe the same single image. Treat them as imperfect witness "
    "ballots. First, identify concrete visual claims stated by the witnesses in natural "
    "language. Do not use a fixed taxonomy or prepopulated semantic framework; create only "
    "the claims that naturally appear in the source captions. Vote on those claims by source "
    "caption count and by distinct model families. Strong repeated agreement should win over "
    "minority contradictions. If one caption says blue eyes and most others say purple eyes, "
    "purple eyes wins and blue eyes is rejected or unresolved. If two incompatible claims have "
    "weak or tied support, do not promote either as accepted evidence. Keep useful one-off "
    "details as singleton candidates when they are plausible and not contradicted, because a "
    "later VLM will inspect the image and decide whether they are visible. Examples of useful "
    "singletons include jewelry, makeup, seams, buttons, fabric texture, accessories, fingernails, "
    "small props, and subtle pose details. After the ballot step, write a rich copywriter draft "
    "from accepted claims plus plausible singleton candidates. The draft is not final; the VLM "
    "validator will ground it against the image. Prefer rich LoRA-training detail density over "
    "shortness. Include trigger word and user caption anchor exactly when provided as training "
    "metadata/guidance. Do not invent details that have no source support. Do not mention source "
    "captions, ballots, models, uncertainty, or this process in the draft captions."
)

# -----------------------------------------------------------------------------
# Dataclasses
# -----------------------------------------------------------------------------


@dataclass
class DistillerConfig:
    """Runtime settings for one distillation batch."""

    llm_backend: str = "ollama"  # ollama | manual_json | prompt_only
    llm_model: str = ""
    instructions: str = DEFAULT_DISTILLER_INSTRUCTIONS

    strategy: str = "pollster_then_copywriter"  # fixed best path; legacy values are normalized to this
    staged_threshold: int = 12
    max_caption_chars_for_llm: int = 1000

    ollama_base_url: str = DEFAULT_OLLAMA_BASE_URL
    ollama_timeout_sec: int = 600
    ollama_keep_alive: str = "5m"
    ollama_num_predict: int = 2400
    ollama_temperature: float = 0.24
    ollama_top_p: float = 0.90
    ollama_top_k: int = 60
    ollama_seed: int = -1

    preserve_raw_response: bool = False

    # Optional user-supplied caption guidance.
    # trigger_word is metadata/training-token text and should be preserved exactly.
    # user_caption_anchor is user guidance such as style/identity text that can be
    # mixed into the over-complete captions before later VLM pruning.
    trigger_word: str = ""
    user_caption_anchor: str = ""


@dataclass
class BatchConfig:
    input_jsonl: str
    output_jsonl: str = ""
    readable_jsonl: str = ""
    readable_json: str = ""
    prompt_jsonl: str = ""

    dry_run: bool = False
    append_output: bool = False
    skip_existing: bool = False
    no_readable_sidecars: bool = False
    write_prompt_jsonl: bool = False

    limit_images: int = 0
    image_key_filter: str = ""
    include_error_pass_a_records: bool = False
    manual_json_path: str = ""


@dataclass
class SourceCaptionRef:
    source_record_index: int
    image: str
    image_key: str
    model_family: str
    model_name: str
    ensemble_run_index: int
    status: str
    caption: str
    raw_caption: str
    timestamp: str


@dataclass
class ImageDistillRecord:
    captionforge_pass: str
    engine: str
    engine_version: str
    image_key: str
    image: str
    status: str

    trigger_word: str
    user_caption_anchor: str

    source_caption_count: int
    source_caption_records: list[SourceCaptionRef]

    params: dict[str, Any] = field(default_factory=dict)

    accepted_claims: list[dict[str, Any]] = field(default_factory=list)
    singleton_candidates: list[dict[str, Any]] = field(default_factory=list)
    rejected_or_unresolved_claims: list[dict[str, Any]] = field(default_factory=list)
    claim_vote_summary: list[str] = field(default_factory=list)
    rich_caption_draft: str = ""
    taggy_caption_draft: str = ""
    distill_parse_result: dict[str, Any] = field(default_factory=dict)
    raw_distill_response: str = ""
    warnings: list[str] = field(default_factory=list)
    metrics: dict[str, Any] = field(default_factory=dict)
    timestamp: str = ""
    error: str = ""


# -----------------------------------------------------------------------------
# Generic helpers
# -----------------------------------------------------------------------------


def now_iso() -> str:
    return datetime.now().isoformat(timespec="seconds")



def normalize_whitespace(value: Any) -> str:
    return re.sub(r"\s+", " ", str(value or "").replace("\n", " ")).strip()



def sha1_text(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()



def json_safe(value: Any) -> Any:
    if dataclasses.is_dataclass(value):
        return {k: json_safe(v) for k, v in dataclasses.asdict(value).items()}
    if isinstance(value, dict):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, list):
        return [json_safe(v) for v in value]
    if isinstance(value, tuple):
        return [json_safe(v) for v in value]
    if isinstance(value, set):
        return sorted(json_safe(v) for v in value)
    if isinstance(value, Path):
        return str(value)
    return value



def write_jsonl_record(path: Path, record: Any, dry_run: bool = False) -> None:
    if dry_run:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8", newline="\n") as f:
        f.write(json.dumps(json_safe(record), ensure_ascii=False) + "\n")
        f.flush()



def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    p = Path(path)
    if p.is_dir():
        raise IsADirectoryError(f"JSONL path points to a folder, not a file: {p}")
    if not p.exists():
        raise FileNotFoundError(f"JSONL file does not exist: {p}")

    records: list[dict[str, Any]] = []
    with p.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSONL at {p}:{line_no}: {exc}") from exc
            if not isinstance(obj, dict):
                raise ValueError(f"JSONL record at {p}:{line_no} is not an object")
            records.append(obj)
    return records



def default_output_path(input_jsonl: str | Path) -> Path:
    p = Path(input_jsonl)
    return p.with_name(f"{p.stem or 'captions'}_distilled.jsonl")



def default_readable_jsonl_path(output_jsonl: Path) -> Path:
    return output_jsonl.with_name(f"{output_jsonl.stem}_readable.jsonl")



def default_readable_json_path(output_jsonl: Path) -> Path:
    return output_jsonl.with_name(f"{output_jsonl.stem}_readable.json")



def default_prompt_jsonl_path(output_jsonl: Path) -> Path:
    return output_jsonl.with_name(f"{output_jsonl.stem}_prompts.jsonl")


# -----------------------------------------------------------------------------
# Readable sidecars
# -----------------------------------------------------------------------------


def readable_record(record: ImageDistillRecord) -> dict[str, Any]:
    """Small human-readable sidecar record: evidence buckets plus draft captions."""
    return {
        "image_key": record.image_key,
        "trigger_word": record.trigger_word,
        "user_caption_anchor": record.user_caption_anchor,
        "accepted_claims": record.accepted_claims,
        "singleton_candidates": record.singleton_candidates,
        "rejected_or_unresolved_claims": record.rejected_or_unresolved_claims,
        "claim_vote_summary": record.claim_vote_summary,
        "rich_caption_draft": record.rich_caption_draft,
        "taggy_caption_draft": record.taggy_caption_draft,
    }



def write_readable_jsonl_record(path: Path, record: ImageDistillRecord, dry_run: bool = False) -> None:
    if dry_run:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8", newline="\n") as f:
        f.write(json.dumps(readable_record(record), ensure_ascii=False) + "\n")
        f.flush()



def write_readable_json(path: Path, records: list[ImageDistillRecord], dry_run: bool = False) -> None:
    if dry_run:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps([readable_record(r) for r in records], ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


# -----------------------------------------------------------------------------
# Pass A grouping
# -----------------------------------------------------------------------------


def pass_a_record_is_usable(record: dict[str, Any], include_errors: bool = False) -> bool:
    pass_name = str(record.get("captionforge_pass") or "").upper()
    if pass_name not in {"", "A"}:
        return False

    status = str(record.get("status") or "ok").lower()
    if status != "ok" and not include_errors:
        return False

    caption = normalize_whitespace(record.get("caption") or record.get("raw_caption") or "")
    return bool(caption) or include_errors



def get_image_key(record: dict[str, Any]) -> str:
    return normalize_whitespace(
        record.get("image_key") or record.get("image") or record.get("source") or ""
    )



def group_pass_a_records(
    records: list[dict[str, Any]], *, include_errors: bool = False
) -> dict[str, list[dict[str, Any]]]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for idx, rec in enumerate(records):
        if not pass_a_record_is_usable(rec, include_errors=include_errors):
            continue
        key = get_image_key(rec)
        if not key:
            continue
        enriched = dict(rec)
        enriched["_source_record_index"] = idx
        groups[key].append(enriched)
    return dict(groups)



def make_source_ref(record: dict[str, Any]) -> SourceCaptionRef:
    return SourceCaptionRef(
        source_record_index=int(record.get("_source_record_index", 0)),
        image=str(record.get("image") or ""),
        image_key=str(record.get("image_key") or get_image_key(record)),
        model_family=str(record.get("model_family") or ""),
        model_name=str(record.get("model_name") or ""),
        ensemble_run_index=int(record.get("ensemble_run_index") or 0),
        status=str(record.get("status") or "ok"),
        caption=str(record.get("caption") or ""),
        raw_caption=str(record.get("raw_caption") or ""),
        timestamp=str(record.get("timestamp") or ""),
    )



# -----------------------------------------------------------------------------
# User caption guidance
# -----------------------------------------------------------------------------


def _first_text_from_record(record: dict[str, Any], keys: tuple[str, ...]) -> str:
    """Return first non-empty text value from top-level or common nested params."""
    for key in keys:
        value = normalize_whitespace(record.get(key) or "")
        if value:
            return value

    for parent_key in ("params", "run_plan", "captionforge_plan", "pipeline_plan"):
        parent = record.get(parent_key)
        if not isinstance(parent, dict):
            continue
        for key in keys:
            value = normalize_whitespace(parent.get(key) or "")
            if value:
                return value
    return ""


def resolve_trigger_word(group: list[dict[str, Any]], config: DistillerConfig) -> str:
    """Prefer CLI/node trigger override, then trigger metadata from Pass A records."""
    configured = normalize_whitespace(getattr(config, "trigger_word", "") or "")
    if configured:
        return configured

    keys = (
        "trigger_word",
        "trigger",
        "caption_trigger",
        "lora_trigger",
        "training_trigger",
    )
    seen: list[str] = []
    for record in group:
        value = _first_text_from_record(record, keys)
        if value and value not in seen:
            seen.append(value)
    return ", ".join(seen)


def resolve_user_caption_anchor(group: list[dict[str, Any]], config: DistillerConfig) -> str:
    """Prefer CLI/node anchor override, then anchor metadata from Pass A records."""
    configured = normalize_whitespace(getattr(config, "user_caption_anchor", "") or "")
    if configured:
        return configured

    keys = (
        "user_caption_anchor",
        "caption_anchor",
        "style_anchor",
        "user_anchor",
        "caption_prefix",
        "semantic_anchor",
    )
    seen: list[str] = []
    for record in group:
        value = _first_text_from_record(record, keys)
        if value and value not in seen:
            seen.append(value)
    return " ".join(seen)

# -----------------------------------------------------------------------------
# Prompt construction
# -----------------------------------------------------------------------------


def compact_caption(caption: str, max_chars: int) -> tuple[str, bool, int]:
    text = normalize_whitespace(caption)
    original_len = len(text)
    if max_chars <= 0 or original_len <= max_chars:
        return text, False, original_len

    window = text[:max_chars]
    boundary = max(window.rfind(". "), window.rfind("; "), window.rfind(", "))
    if boundary >= int(max_chars * 0.55):
        out = window[: boundary + 1].rstrip(" ,;.")
    else:
        out = window.rsplit(" ", 1)[0].rstrip(" ,;.") if " " in window else window.rstrip()
    return (out or window).rstrip() + " …", True, original_len



def normalize_strategy(value: str) -> str:
    """CaptionForge v0.2 uses one best Pass B strategy.

    Legacy values from older nodes are accepted but normalized to the fixed
    pollster/copywriter path so users are not asked to choose between prompt
    strategies that are not equally good.
    """
    return "pollster_then_copywriter"



def build_prompt(
    image_key: str,
    source_refs: list[SourceCaptionRef],
    config: DistillerConfig,
    stage_name: str,
) -> dict[str, Any]:
    captions: list[dict[str, Any]] = []
    compacted_count = 0
    original_chars = 0
    prompt_caption_chars = 0

    for i, src in enumerate(source_refs):
        cap, compacted, n0 = compact_caption(
            src.caption or src.raw_caption,
            config.max_caption_chars_for_llm,
        )
        compacted_count += int(compacted)
        original_chars += n0
        prompt_caption_chars += len(cap)
        captions.append(
            {
                "local_source_index": i,
                "source_record_index": src.source_record_index,
                "model_family": src.model_family,
                "model_name": src.model_name,
                "ensemble_run_index": src.ensemble_run_index,
                "caption": cap,
            }
        )

    expected = {
        "accepted_claims": [
            {
                "claim": "natural-language visual claim promoted by the ballot",
                "support_count": 2,
                "supporting_model_families": ["joy", "qwen"],
                "supporting_source_indices": [0, 3],
                "contradicted_by": [],
                "confidence": "high|medium|low",
                "reason": "brief vote/evidence reason"
            }
        ],
        "singleton_candidates": [
            {
                "claim": "plausible but only weakly supported detail for VLM inspection",
                "support_count": 1,
                "supporting_model_families": ["joy"],
                "supporting_source_indices": [1],
                "reason": "why this should be preserved for visual validation"
            }
        ],
        "rejected_or_unresolved_claims": [
            {
                "claim": "claim not promoted",
                "reason": "minority contradiction, unresolved tie, vague wording, or unsupported conflict"
            }
        ],
        "claim_vote_summary": [
            "brief human-readable notes about the strongest wins and dropped conflicts"
        ],
        "rich_caption_draft": "rich LoRA-training draft written from accepted claims plus plausible singleton candidates",
        "taggy_caption_draft": "dense comma-separated draft with the same visual content as compact training fragments"
    }

    trigger_word = normalize_whitespace(getattr(config, "trigger_word", "") or "")
    user_caption_anchor = normalize_whitespace(getattr(config, "user_caption_anchor", "") or "")
    user_guidance = {
        "trigger_word": trigger_word,
        "user_caption_anchor": user_caption_anchor,
    }

    prompt = (
        f"{config.instructions}\n\n"

        "Return strict JSON only. Do not use markdown. Do not write prose outside JSON.\n\n"

        "USER-PROVIDED GUIDANCE:\n"
        "- trigger_word is metadata/training-token text supplied by the pipeline or user. "
        "If present, preserve it exactly at the beginning of rich_caption_draft and taggy_caption_draft. "
        "Do not rewrite, split, translate, lowercase, expand, or explain it.\n"
        "- user_caption_anchor is optional user-provided style/identity guidance. "
        "If present, include its useful descriptive wording in the draft captions unless it is clearly contradicted by the source captions. "
        "Treat it as intentional training guidance; the VLM validator will still ground it against the image.\n"
        f"User guidance JSON:\n{json.dumps(user_guidance, ensure_ascii=False, indent=2)}\n\n"

        "BALLOT RULES:\n"
        "- All source captions describe the same single image.\n"
        "- Extract concrete visual claims from the captions in natural language.\n"
        "- Do not use a fixed taxonomy. Do not invent fields such as eye_color or hair_style unless the claim itself naturally appears as text.\n"
        "- Count support by total captions and by distinct model_family values.\n"
        "- Repeated agreement across distinct model families is strongest.\n"
        "- Repeated agreement within one model family is useful but weaker than cross-family agreement.\n"
        "- Strong majority wins over minority contradiction.\n"
        "- Weak one-off details should become singleton_candidates if plausible and not contradicted.\n"
        "- Tied, contradictory, vague, or mutually exclusive low-support details should go to rejected_or_unresolved_claims.\n"
        "- Keep small but useful LoRA details alive as singleton_candidates: jewelry, makeup, nails, fabric texture, seams, trim, buttons, gems, accessories, pose details, and background details.\n\n"

        "COPYWRITER RULES:\n"
        "- rich_caption_draft should be rich, clean, persuasive LoRA-training prose.\n"
        "- taggy_caption_draft should be dense comma-separated caption text with nearly the same visual content.\n"
        "- Do not summarize aggressively. Do not compress detailed outfit/jewelry/makeup/material claims into generic phrases.\n"
        "- Do not mention ballots, sources, captions, models, uncertainty, support counts, or rejected claims in the draft captions.\n"
        "- The VLM validator will inspect the image next, so it is better to preserve plausible visual detail than to under-caption.\n\n"

        f"Expected JSON shape:\n{json.dumps(expected, ensure_ascii=False, indent=2)}\n\n"
        f"Stage: {stage_name}\n"
        f"Image key: {image_key}\n"
        f"Source captions:\n{json.dumps(captions, ensure_ascii=False, indent=2)}\n"
    )

    return {
        "captionforge_pass": "B_DISTILL_PROMPT",
        "engine": ENGINE_NAME,
        "engine_version": ENGINE_VERSION,
        "image_key": image_key,
        "stage_name": stage_name,
        "prompt": prompt,
        "prompt_sha1": sha1_text(prompt),
        "source_caption_count": len(source_refs),
        "user_guidance": user_guidance,
        "caption_compaction": {
            "max_caption_chars_for_llm": config.max_caption_chars_for_llm,
            "compacted_caption_count": compacted_count,
            "original_caption_chars_total": original_chars,
            "prompt_caption_chars_total": prompt_caption_chars,
        },
        "timestamp": now_iso(),
    }


# -----------------------------------------------------------------------------
# Ollama backend
# -----------------------------------------------------------------------------


_MODEL_CHECK_CACHE: set[tuple[str, str]] = set()



def normalize_ollama_base_url(url: str) -> str:
    return str(url or DEFAULT_OLLAMA_BASE_URL).strip().rstrip("/")



def ollama_json(
    base_url: str,
    endpoint: str,
    payload: Optional[dict[str, Any]] = None,
    timeout_sec: int = 600,
) -> Any:
    url = normalize_ollama_base_url(base_url) + endpoint
    data = json.dumps(payload).encode("utf-8") if payload is not None else None
    req = urllib.request.Request(
        url=url,
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST" if payload is not None else "GET",
    )
    try:
        with urllib.request.urlopen(req, timeout=int(timeout_sec)) as resp:
            text = resp.read().decode("utf-8", errors="replace")
    except urllib.error.URLError as exc:
        raise RuntimeError("Could not reach Ollama. Make sure Ollama is running.") from exc
    return json.loads(text) if text.strip() else None



def ensure_ollama_model(config: DistillerConfig) -> None:
    model = normalize_whitespace(config.llm_model)
    if not model:
        raise ValueError("--llm-model is required for Ollama backend")

    cache_key = (normalize_ollama_base_url(config.ollama_base_url), model)
    if cache_key in _MODEL_CHECK_CACHE:
        return

    tags = ollama_json(
        config.ollama_base_url,
        "/api/tags",
        timeout_sec=min(60, config.ollama_timeout_sec),
    )
    names = {str(m.get("name") or "") for m in tags.get("models", [])} if isinstance(tags, dict) else set()
    ok = model in names or (":" not in model and any(n.startswith(model + ":") for n in names))
    if not ok:
        raise RuntimeError(f"Ollama model is not installed: {model}\nInstall with: ollama pull {model}")
    _MODEL_CHECK_CACHE.add(cache_key)



def call_ollama(prompt: str, config: DistillerConfig) -> str:
    ensure_ollama_model(config)

    payload: dict[str, Any] = {
        "model": config.llm_model,
        "prompt": prompt,
        "stream": False,
        "format": "json",
        "options": {
            "temperature": float(config.ollama_temperature),
            "num_predict": int(config.ollama_num_predict),
            "top_p": float(config.ollama_top_p),
            "top_k": int(config.ollama_top_k),
        },
    }
    if int(getattr(config, "ollama_seed", -1)) >= 0:
        payload["options"]["seed"] = int(config.ollama_seed)
    if config.ollama_keep_alive:
        payload["keep_alive"] = config.ollama_keep_alive

    print(
        f"[CaptionForge Distiller] Ollama model={config.llm_model} "
        f"num_predict={config.ollama_num_predict} "
        f"temp={config.ollama_temperature:g} top_p={config.ollama_top_p:g} top_k={config.ollama_top_k}",
        flush=True,
    )

    obj = ollama_json(
        config.ollama_base_url,
        "/api/generate",
        payload=payload,
        timeout_sec=config.ollama_timeout_sec,
    )
    if not isinstance(obj, dict):
        raise RuntimeError("Ollama returned a non-object response")
    if obj.get("error"):
        raise RuntimeError(f"Ollama generation failed: {obj.get('error')}")

    response = normalize_whitespace(obj.get("response") or "")
    if not response:
        raise RuntimeError("Ollama returned an empty response")
    return response


# -----------------------------------------------------------------------------
# Response parsing
# -----------------------------------------------------------------------------


def strip_json_fence(text: str) -> str:
    text = str(text or "").strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.IGNORECASE)
        text = re.sub(r"\s*```$", "", text)
    return text.strip()



def first_balanced_json_object(text: str) -> str:
    start = text.find("{")
    if start < 0:
        raise ValueError("No JSON object found in response")

    depth = 0
    in_string = False
    escape = False
    for idx in range(start, len(text)):
        ch = text[idx]
        if in_string:
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == '"':
                in_string = False
            continue
        if ch == '"':
            in_string = True
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return text[start : idx + 1]
    return text[start:]



def parse_json_object(text: str) -> dict[str, Any]:
    text = strip_json_fence(text)
    if not text:
        raise ValueError("Empty response")

    attempts = [text]
    try:
        balanced = first_balanced_json_object(text)
        if balanced != text:
            attempts.append(balanced)
    except Exception:
        pass

    last_error: Optional[Exception] = None
    for candidate in attempts:
        try:
            obj = json.loads(candidate)
            if not isinstance(obj, dict):
                raise ValueError("Response JSON must be an object")
            return obj
        except Exception as exc:
            last_error = exc
    raise ValueError(f"Invalid JSON response: {last_error}")



def normalize_caption_value(value: Any, *, joiner: str) -> str:
    if isinstance(value, list):
        return joiner.join(normalize_whitespace(x).strip(" ,.;") for x in value if normalize_whitespace(x))
    if isinstance(value, dict):
        for key in ("caption", "text", "value", "claim"):
            if key in value:
                return normalize_caption_value(value[key], joiner=joiner)
    return normalize_whitespace(value)


def normalize_string_list(value: Any, *, max_items: int = 80) -> list[str]:
    if value is None:
        return []
    if isinstance(value, list):
        out: list[str] = []
        for item in value:
            if isinstance(item, dict):
                text = normalize_whitespace(item.get("claim") or item.get("text") or item.get("value") or item.get("reason") or "")
            else:
                text = normalize_whitespace(item)
            if text:
                out.append(text)
            if len(out) >= max_items:
                break
        return out
    text = normalize_whitespace(value)
    return [text] if text else []


def normalize_supporting_sources(value: Any) -> list[int]:
    out: list[int] = []
    if not isinstance(value, list):
        value = [value] if value not in (None, "") else []
    for item in value:
        try:
            out.append(int(item))
        except Exception:
            continue
    # preserve order, remove duplicates
    seen = set()
    clean = []
    for item in out:
        if item not in seen:
            seen.add(item)
            clean.append(item)
    return clean


def normalize_claim_list(value: Any) -> list[dict[str, Any]]:
    if value is None:
        return []
    items = value if isinstance(value, list) else [value]
    out: list[dict[str, Any]] = []
    for item in items:
        if isinstance(item, dict):
            claim = normalize_whitespace(item.get("claim") or item.get("text") or item.get("value") or "")
            if not claim:
                continue
            families_raw = (
                item.get("supporting_model_families")
                or item.get("supporting_models")
                or item.get("models")
                or []
            )
            contradicted_raw = item.get("contradicted_by") or item.get("contradictions") or []
            try:
                support_count = int(item.get("support_count") or item.get("votes") or 0)
            except Exception:
                support_count = 0
            out.append({
                "claim": claim,
                "support_count": max(0, support_count),
                "supporting_model_families": normalize_string_list(families_raw, max_items=20),
                "supporting_source_indices": normalize_supporting_sources(
                    item.get("supporting_source_indices") or item.get("source_indices") or []
                ),
                "contradicted_by": normalize_string_list(contradicted_raw, max_items=20),
                "confidence": normalize_whitespace(item.get("confidence") or ""),
                "reason": normalize_whitespace(item.get("reason") or item.get("rationale") or ""),
            })
        else:
            claim = normalize_whitespace(item)
            if claim:
                out.append({
                    "claim": claim,
                    "support_count": 0,
                    "supporting_model_families": [],
                    "supporting_source_indices": [],
                    "contradicted_by": [],
                    "confidence": "",
                    "reason": "",
                })
    return out


def normalize_response(obj: dict[str, Any]) -> dict[str, Any]:
    accepted_claims = normalize_claim_list(
        obj.get("accepted_claims")
        or obj.get("winning_claims")
        or obj.get("promoted_claims")
        or []
    )
    singleton_candidates = normalize_claim_list(
        obj.get("singleton_candidates")
        or obj.get("plausible_singletons")
        or obj.get("low_support_candidates")
        or []
    )
    rejected_or_unresolved_claims = normalize_claim_list(
        obj.get("rejected_or_unresolved_claims")
        or obj.get("rejected_claims")
        or obj.get("unresolved_claims")
        or obj.get("conflicts")
        or []
    )
    claim_vote_summary = normalize_string_list(
        obj.get("claim_vote_summary")
        or obj.get("vote_summary")
        or obj.get("ballot_summary")
        or [],
        max_items=40,
    )

    rich_caption_draft = normalize_caption_value(
        obj.get("rich_caption_draft")
        or obj.get("copywriter_caption")
        or obj.get("draft_caption")
        or obj.get("distilled_caption_narrative")
        or obj.get("narrative")
        or "",
        joiner=" ",
    )
    taggy_caption_draft = normalize_caption_value(
        obj.get("taggy_caption_draft")
        or obj.get("dense_caption_draft")
        or obj.get("comma_caption_draft")
        or obj.get("distilled_caption_comma")
        or obj.get("comma")
        or "",
        joiner=", ",
    )

    if not accepted_claims and not singleton_candidates and not rich_caption_draft:
        raise ValueError("Response missing accepted_claims/singleton_candidates/rich_caption_draft")
    if not rich_caption_draft:
        # Fallback for models that followed the evidence contract but omitted prose.
        seed_claims = [c["claim"] for c in accepted_claims[:40]] + [c["claim"] for c in singleton_candidates[:20]]
        rich_caption_draft = normalize_whitespace(". ".join(seed_claims))
    if not taggy_caption_draft:
        seed_claims = [c["claim"] for c in accepted_claims[:60]] + [c["claim"] for c in singleton_candidates[:30]]
        taggy_caption_draft = normalize_caption_value(seed_claims, joiner=", ")

    return {
        "accepted_claims": accepted_claims,
        "singleton_candidates": singleton_candidates,
        "rejected_or_unresolved_claims": rejected_or_unresolved_claims,
        "claim_vote_summary": claim_vote_summary,
        "rich_caption_draft": rich_caption_draft,
        "taggy_caption_draft": taggy_caption_draft,
    }


# -----------------------------------------------------------------------------
# Manual backend
# -----------------------------------------------------------------------------


def load_manual_json(path: str) -> dict[str, Any]:
    if not path:
        return {}
    p = Path(path)
    if p.is_dir():
        raise IsADirectoryError(f"manual JSON path points to a folder: {p}")
    if not p.exists():
        raise FileNotFoundError(f"manual JSON file does not exist: {p}")

    if p.suffix.lower() == ".jsonl":
        out: dict[str, Any] = {}
        for rec in read_jsonl(p):
            out[normalize_whitespace(rec.get("image_key") or "__single__")] = rec
        return out

    obj = json.loads(p.read_text(encoding="utf-8"))
    if isinstance(obj, dict):
        if (
            "accepted_claims" in obj
            or "rich_caption_draft" in obj
            or "taggy_caption_draft" in obj
            or "distilled_caption_narrative" in obj
            or "image_key" in obj
        ):
            return {normalize_whitespace(obj.get("image_key") or "__single__"): obj}
        return obj
    raise ValueError("manual JSON must be an object or JSONL file")


# -----------------------------------------------------------------------------
# Distillation execution
# -----------------------------------------------------------------------------


def run_one(
    image_key: str,
    source_refs: list[SourceCaptionRef],
    config: DistillerConfig,
    stage_name: str,
    manual: Optional[dict[str, Any]] = None,
    prompt_sink_path: Optional[Path] = None,
    dry_run: bool = False,
) -> tuple[dict[str, Any], dict[str, Any], str]:
    bundle = build_prompt(image_key, source_refs, config, stage_name)
    if prompt_sink_path is not None:
        write_jsonl_record(prompt_sink_path, bundle, dry_run=dry_run)

    raw = ""
    t0 = time.perf_counter()
    try:
        if config.llm_backend == "prompt_only":
            raise RuntimeError("prompt_only backend wrote prompt bundle but did not run an LLM")
        if config.llm_backend == "manual_json":
            if manual is None:
                raise ValueError("manual_json backend selected but no manual response exists for image")
            obj = manual
        elif config.llm_backend == "ollama":
            raw = call_ollama(bundle["prompt"], config)
            obj = parse_json_object(raw)
        else:
            raise ValueError(f"Unsupported backend: {config.llm_backend}")

        captions = normalize_response(obj)
        parse = {
            "backend": config.llm_backend,
            "model": config.llm_model,
            "status": "ok",
            "prompt_sha1": bundle["prompt_sha1"],
            "prompt_chars": len(bundle["prompt"]),
            "raw_response_chars": len(raw),
            "elapsed_sec": round(time.perf_counter() - t0, 3),
        }
        return captions, parse, raw
    except Exception as exc:
        parse = {
            "backend": config.llm_backend,
            "model": config.llm_model,
            "status": "error",
            "prompt_sha1": bundle["prompt_sha1"],
            "prompt_chars": len(bundle["prompt"]),
            "raw_response_chars": len(raw),
            "elapsed_sec": round(time.perf_counter() - t0, 3),
            "error_class": type(exc).__name__,
            "error": str(exc),
        }
        raise RuntimeError(json.dumps(parse, ensure_ascii=False)) from exc



def synth_source(image_key: str, image: str, family: str, text: str, idx: int) -> SourceCaptionRef:
    return SourceCaptionRef(
        source_record_index=idx,
        image=image,
        image_key=image_key,
        model_family=f"distiller_{family}",
        model_name=f"distiller_{family}",
        ensemble_run_index=0,
        status="ok",
        caption=text,
        raw_caption=text,
        timestamp=now_iso(),
    )



def run_pipeline(
    image_key: str,
    refs: list[SourceCaptionRef],
    config: DistillerConfig,
    manual_by_key: dict[str, Any],
    prompt_sink_path: Optional[Path] = None,
    dry_run: bool = False,
) -> tuple[dict[str, Any], dict[str, Any], str]:
    """Run the fixed CaptionForge v0.2 Pass B path.

    Earlier staged summarization could compress evidence twice. The current
    engine uses one global pollster/copywriter prompt so all witness captions
    compete in a single ballot.
    """
    manual = manual_by_key.get(image_key) or (
        manual_by_key.get("__single__") if len(manual_by_key) == 1 else None
    )
    return run_one(
        image_key,
        refs,
        config,
        "global_pollster_then_copywriter",
        manual,
        prompt_sink_path=prompt_sink_path,
        dry_run=dry_run,
    )



def params_record(config: DistillerConfig) -> dict[str, Any]:
    return {
        "engine_name": ENGINE_NAME,
        "engine_version": ENGINE_VERSION,
        "llm_backend": config.llm_backend,
        "llm_model": config.llm_model,
        "strategy": normalize_strategy(config.strategy),
        "contract": "pollster_then_copywriter_v0.2",
        "staged_threshold": config.staged_threshold,
        "max_caption_chars_for_llm": config.max_caption_chars_for_llm,
        "ollama_base_url": config.ollama_base_url,
        "ollama_timeout_sec": config.ollama_timeout_sec,
        "ollama_keep_alive": config.ollama_keep_alive,
        "ollama_num_predict": config.ollama_num_predict,
        "ollama_temperature": config.ollama_temperature,
        "ollama_top_p": config.ollama_top_p,
        "ollama_top_k": config.ollama_top_k,
        "preserve_raw_response": config.preserve_raw_response,
        "ollama_seed": int(getattr(config, "ollama_seed", -1)),
        "trigger_word": config.trigger_word,
        "user_caption_anchor": config.user_caption_anchor,
    }



def build_record(
    image_key: str,
    group: list[dict[str, Any]],
    config: DistillerConfig,
    manual_by_key: dict[str, Any],
    prompt_sink_path: Optional[Path] = None,
    dry_run: bool = False,
) -> ImageDistillRecord:
    refs = [make_source_ref(r) for r in group]
    image = refs[0].image if refs else image_key
    warnings: list[str] = []

    trigger_word = resolve_trigger_word(group, config)
    user_caption_anchor = resolve_user_caption_anchor(group, config)
    local_config = dataclasses.replace(
        config,
        trigger_word=trigger_word,
        user_caption_anchor=user_caption_anchor,
    )

    try:
        captions, parse, raw = run_pipeline(
            image_key,
            refs,
            local_config,
            manual_by_key,
            prompt_sink_path=prompt_sink_path,
            dry_run=dry_run,
        )
        status = "ok"
        error = ""
    except Exception as exc:
        captions = {"accepted_claims": [], "singleton_candidates": [], "rejected_or_unresolved_claims": [], "claim_vote_summary": [], "rich_caption_draft": "", "taggy_caption_draft": ""}
        raw = ""
        status = "error"
        error = str(exc)
        try:
            parse = json.loads(str(exc))
        except Exception:
            parse = {"status": "error", "error_class": type(exc).__name__, "error": str(exc)}

    return ImageDistillRecord(
        captionforge_pass=CAPTIONFORGE_PASS,
        engine=ENGINE_NAME,
        engine_version=ENGINE_VERSION,
        image_key=image_key,
        image=image,
        status=status,
        trigger_word=trigger_word,
        user_caption_anchor=user_caption_anchor,
        source_caption_count=len(refs),
        source_caption_records=refs,
        params=params_record(local_config),
        accepted_claims=captions["accepted_claims"],
        singleton_candidates=captions["singleton_candidates"],
        rejected_or_unresolved_claims=captions["rejected_or_unresolved_claims"],
        claim_vote_summary=captions["claim_vote_summary"],
        rich_caption_draft=captions["rich_caption_draft"],
        taggy_caption_draft=captions["taggy_caption_draft"],
        distill_parse_result=parse,
        raw_distill_response=raw if local_config.preserve_raw_response else "",
        warnings=warnings,
        metrics={
            "source_caption_count": len(refs),
            "accepted_claim_count": len(captions["accepted_claims"]),
            "singleton_candidate_count": len(captions["singleton_candidates"]),
            "rejected_or_unresolved_claim_count": len(captions["rejected_or_unresolved_claims"]),
            "rich_caption_draft_chars": len(captions["rich_caption_draft"]),
            "taggy_caption_draft_chars": len(captions["taggy_caption_draft"]),
            "raw_response_chars": len(raw),
        },
        timestamp=now_iso(),
        error=error,
    )


# -----------------------------------------------------------------------------
# Batch processing
# -----------------------------------------------------------------------------


def successful_keys(path: Path) -> set[str]:
    if not path.exists():
        return set()
    return {
        normalize_whitespace(r.get("image_key"))
        for r in read_jsonl(path)
        if r.get("status") == "ok" and normalize_whitespace(r.get("image_key"))
    }



def process_batch(batch: BatchConfig, config: DistillerConfig) -> int:
    groups = group_pass_a_records(
        read_jsonl(batch.input_jsonl),
        include_errors=batch.include_error_pass_a_records,
    )
    keys = sorted(groups)
    if batch.image_key_filter:
        keys = [k for k in keys if k == batch.image_key_filter]
    if batch.limit_images > 0:
        keys = keys[: batch.limit_images]

    out_path = Path(batch.output_jsonl) if batch.output_jsonl else default_output_path(batch.input_jsonl)
    readable_jsonl_path = Path(batch.readable_jsonl) if batch.readable_jsonl else default_readable_jsonl_path(out_path)
    readable_json_path = Path(batch.readable_json) if batch.readable_json else default_readable_json_path(out_path)
    prompt_jsonl_path = Path(batch.prompt_jsonl) if batch.prompt_jsonl else default_prompt_jsonl_path(out_path)

    readable_records: list[ImageDistillRecord] = []

    if not batch.append_output and not batch.skip_existing and not batch.dry_run:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text("", encoding="utf-8")
        if not batch.no_readable_sidecars:
            readable_jsonl_path.write_text("", encoding="utf-8")
            readable_json_path.write_text("[]\n", encoding="utf-8")
        if batch.write_prompt_jsonl:
            prompt_jsonl_path.write_text("", encoding="utf-8")

    done = successful_keys(out_path) if batch.skip_existing else set()
    manual = load_manual_json(batch.manual_json_path) if batch.manual_json_path else {}

    ok = failed = skipped = 0
    total = len(keys)
    print(f"[CaptionForge Distiller] Found {total} image group(s).", flush=True)

    for i, key in enumerate(keys, start=1):
        if key in done:
            skipped += 1
            print(f"[{i}/{total}] SKIP {key}", flush=True)
            continue

        print(f"[{i}/{total}] DISTILL {key}", flush=True)
        rec = build_record(
            key,
            groups[key],
            config,
            manual,
            prompt_sink_path=prompt_jsonl_path if batch.write_prompt_jsonl else None,
            dry_run=batch.dry_run,
        )
        ok += int(rec.status == "ok")
        failed += int(rec.status != "ok")
        if rec.status != "ok":
            print(f"    ERROR {rec.error}", flush=True)

        write_jsonl_record(out_path, rec, dry_run=batch.dry_run)

        if not batch.no_readable_sidecars:
            write_readable_jsonl_record(readable_jsonl_path, rec, dry_run=batch.dry_run)
            readable_records.append(rec)
            write_readable_json(readable_json_path, readable_records, dry_run=batch.dry_run)

    readable_msg = ""
    if not batch.no_readable_sidecars:
        readable_msg = f" readable_jsonl={readable_jsonl_path} readable_json={readable_json_path}"
    prompt_msg = f" prompt_jsonl={prompt_jsonl_path}" if batch.write_prompt_jsonl else ""
    print(
        f"[CaptionForge Distiller] Done. ok={ok} failed={failed} skipped={skipped} "
        f"output={out_path}{readable_msg}{prompt_msg}",
        flush=True,
    )
    return 0 if failed == 0 else 1


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="CaptionForge Distiller Engine v0.2.0")

    p.add_argument("--input-jsonl", required=True, help="Pass A captions JSONL input.")
    p.add_argument("--output-jsonl", default="", help="Full output JSONL path. Default: <input>_distilled.jsonl")
    p.add_argument("--readable-jsonl", default="", help="Human-readable JSONL sidecar path. Default: <output_stem>_readable.jsonl")
    p.add_argument("--readable-json", default="", help="Human-readable JSON sidecar path. Default: <output_stem>_readable.json")
    p.add_argument("--no-readable-sidecars", action="store_true", help="Disable human-readable sidecar files.")
    p.add_argument("--write-prompt-jsonl", action="store_true", help="Write prompt bundles to a JSONL sidecar for debugging.")
    p.add_argument("--prompt-jsonl", default="", help="Prompt JSONL sidecar path. Default: <output_stem>_prompts.jsonl")

    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--append-output", action="store_true")
    p.add_argument("--skip-existing", action="store_true")
    p.add_argument("--limit-images", type=int, default=0)
    p.add_argument("--image-key-filter", default="")
    p.add_argument("--include-error-pass-a-records", action="store_true")

    p.add_argument("--llm-backend", default="ollama", choices=["ollama", "manual_json", "prompt_only"])
    p.add_argument("--llm-model", default="", help="Text LLM model, e.g. llama3.1:8b")
    p.add_argument("--strategy", default="pollster_then_copywriter", choices=["pollster_then_copywriter", "single_pass", "by_model_then_global"])
    p.add_argument("--staged-threshold", type=int, default=12)
    p.add_argument("--max-caption-chars-for-llm", type=int, default=1536)
    p.add_argument("--num-predict", type=int, default=2400)
    p.add_argument("--temperature", type=float, default=0.24)
    p.add_argument("--top-p", type=float, default=0.90)
    p.add_argument("--top-k", type=int, default=60)
    p.add_argument("--seed", type=int, default=-1, help="Optional Ollama seed. -1 omits seed.")
    p.add_argument("--preserve-raw-response", action="store_true")
    p.add_argument("--instructions-file", default="", help="Optional file containing user-editable distiller instructions.")
    p.add_argument("--trigger-word", default="", help="Optional trigger token/string to preserve exactly at the beginning of both distilled captions. Overrides trigger metadata from Pass A records.")
    p.add_argument("--user-caption-anchor", default="", help="Optional user style/identity/caption anchor to mix into both over-complete captions before VLM pruning. Overrides anchor metadata from Pass A records.")
    p.add_argument("--manual-json-path", default="", help="Manual JSON/JSONL responses for manual_json backend.")

    p.add_argument("--ollama-base-url", default=DEFAULT_OLLAMA_BASE_URL)
    p.add_argument("--ollama-timeout-sec", type=int, default=600)
    p.add_argument("--ollama-keep-alive", default="5m")

    # Backward-compatible aliases from v0.9.x distill-only drafts.
    p.add_argument("--distill-llm-backend", dest="llm_backend_alias", default=None, choices=["ollama", "manual_json", "prompt_only"], help=argparse.SUPPRESS)
    p.add_argument("--distill-llm-model", dest="llm_model_alias", default=None, help=argparse.SUPPRESS)
    p.add_argument("--distill-strategy", dest="strategy_alias", default=None, choices=["pollster_then_copywriter", "single_pass", "by_model_then_global"], help=argparse.SUPPRESS)
    p.add_argument("--distill-staged-threshold", dest="staged_threshold_alias", type=int, default=None, help=argparse.SUPPRESS)
    p.add_argument("--distill-max-caption-chars-for-llm", dest="max_caption_chars_alias", type=int, default=None, help=argparse.SUPPRESS)
    p.add_argument("--distill-ollama-num-predict", dest="num_predict_alias", type=int, default=None, help=argparse.SUPPRESS)
    p.add_argument("--distill-ollama-temperature", dest="temperature_alias", type=float, default=None, help=argparse.SUPPRESS)
    p.add_argument("--preserve-raw-distill-response", dest="preserve_raw_response_alias", action="store_true", help=argparse.SUPPRESS)
    p.add_argument("--distill-prompt-file", dest="instructions_file_alias", default=None, help=argparse.SUPPRESS)
    p.add_argument("--mode", default="caption_distill_only", help=argparse.SUPPRESS)
    p.add_argument("--distill-trigger-word", dest="trigger_word_alias", default=None, help=argparse.SUPPRESS)
    p.add_argument("--distill-user-caption-anchor", dest="user_caption_anchor_alias", default=None, help=argparse.SUPPRESS)

    return p



def main(argv: Optional[list[str]] = None) -> int:
    args = build_parser().parse_args(argv)

    llm_backend = args.llm_backend_alias or args.llm_backend
    llm_model = args.llm_model_alias if args.llm_model_alias is not None else args.llm_model
    strategy = args.strategy_alias or args.strategy
    staged_threshold = args.staged_threshold_alias if args.staged_threshold_alias is not None else args.staged_threshold
    max_caption_chars = args.max_caption_chars_alias if args.max_caption_chars_alias is not None else args.max_caption_chars_for_llm
    num_predict = args.num_predict_alias if args.num_predict_alias is not None else args.num_predict
    temperature = args.temperature_alias if args.temperature_alias is not None else args.temperature
    preserve_raw = bool(args.preserve_raw_response or args.preserve_raw_response_alias)
    instructions_file = args.instructions_file_alias or args.instructions_file
    trigger_word = args.trigger_word_alias if getattr(args, "trigger_word_alias", None) is not None else args.trigger_word
    user_caption_anchor = (
        args.user_caption_anchor_alias
        if getattr(args, "user_caption_anchor_alias", None) is not None
        else args.user_caption_anchor
    )

    instructions = DEFAULT_DISTILLER_INSTRUCTIONS
    if instructions_file:
        instructions = Path(instructions_file).read_text(encoding="utf-8")

    config = DistillerConfig(
        llm_backend=llm_backend,
        llm_model=llm_model,
        instructions=instructions,
        strategy=strategy,
        staged_threshold=max(0, int(staged_threshold)),
        max_caption_chars_for_llm=max(0, int(max_caption_chars)),
        ollama_base_url=args.ollama_base_url,
        ollama_timeout_sec=max(1, int(args.ollama_timeout_sec)),
        ollama_keep_alive=args.ollama_keep_alive,
        ollama_num_predict=int(num_predict),
        ollama_temperature=float(temperature),
        ollama_top_p=float(args.top_p),
        ollama_top_k=int(args.top_k),
        ollama_seed=int(args.seed),
        preserve_raw_response=preserve_raw,
        trigger_word=trigger_word,
        user_caption_anchor=user_caption_anchor,
    )

    batch = BatchConfig(
        input_jsonl=args.input_jsonl,
        output_jsonl=args.output_jsonl,
        readable_jsonl=args.readable_jsonl,
        readable_json=args.readable_json,
        prompt_jsonl=args.prompt_jsonl,
        dry_run=bool(args.dry_run),
        append_output=bool(args.append_output),
        skip_existing=bool(args.skip_existing),
        no_readable_sidecars=bool(args.no_readable_sidecars),
        write_prompt_jsonl=bool(args.write_prompt_jsonl),
        limit_images=max(0, int(args.limit_images)),
        image_key_filter=args.image_key_filter,
        include_error_pass_a_records=bool(args.include_error_pass_a_records),
        manual_json_path=args.manual_json_path,
    )

    return process_batch(batch, config)


if __name__ == "__main__":
    raise SystemExit(main())



# =============================================================================
# CaptionForge Experimental Reversed Pipeline: Pass C Text Reconciler
# =============================================================================

RECONCILER_PASS = "C_RECONCILED"

DEFAULT_RECONCILER_INSTRUCTIONS = (
    "You are CaptionForge Pass C: a caption reconciliation and copywriting engine. "
    "You receive several cleaned captions for the same image. Each cleaned caption was already checked against the image by a VLM. "
    "Your job is to merge them into one coherent LoRA-training caption. You are not an image captioner. "
    "You are not a summarizer. Preserve all compatible concrete visual details. Do not shorten aggressively. "
    "Do not replace specific details with generic phrases. Remove duplicate wording. Resolve contradictions. "
    "If two details conflict, prefer the detail repeated across more cleaned captions and across more model families. "
    "If a contradiction cannot be resolved, omit the conflicting detail. Never allow self-contradictions such as two different eye colors, "
    "hair colors, outfits, poses, or backgrounds. Do not add details absent from the cleaned captions. "
    "Write one clear, fluent, self-consistent caption."
)


def _cf_cleaned_record_is_usable(record: dict[str, Any]) -> bool:
    pass_name = str(record.get("captionforge_pass") or "").upper()
    if pass_name not in {"B_VLM_CLEANED", "B_CLEANED", "CLEANED"}:
        return False
    if str(record.get("status") or "ok").lower() not in {"ok", "prompt_only"}:
        return False
    return bool(normalize_whitespace(record.get("cleaned_caption") or record.get("caption") or ""))


def _cf_cleaned_image_key(record: dict[str, Any]) -> str:
    return normalize_whitespace(record.get("image_key") or record.get("image") or "")


def _cf_cleaned_caption(record: dict[str, Any]) -> str:
    return normalize_whitespace(record.get("cleaned_caption") or record.get("caption") or record.get("cleaned_caption_narrative") or "")


def group_cleaned_records(records: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for idx, rec in enumerate(records):
        if not _cf_cleaned_record_is_usable(rec):
            continue
        key = _cf_cleaned_image_key(rec)
        if not key:
            continue
        enriched = dict(rec)
        enriched["_source_record_index"] = idx
        groups[key].append(enriched)
    return dict(groups)


def build_reconciler_prompt(image_key: str, group: list[dict[str, Any]], config: DistillerConfig) -> dict[str, Any]:
    cleaned_captions: list[dict[str, Any]] = []
    for i, rec in enumerate(group):
        caption, compacted, original_len = compact_caption(_cf_cleaned_caption(rec), config.max_caption_chars_for_llm)
        cleaned_captions.append({
            "local_source_index": i,
            "source_record_index": int(rec.get("_source_record_index", 0)),
            "model_family": str(rec.get("model_family") or ""),
            "model_name": str(rec.get("model_name") or ""),
            "ensemble_run_index": int(rec.get("ensemble_run_index") or 0),
            "cleaned_caption": caption,
            "caption_was_compacted_for_prompt": bool(compacted),
            "original_caption_chars": int(original_len),
            "removed_or_rejected_details": rec.get("removed_or_rejected_details") or [],
            "corrected_details": rec.get("corrected_details") or [],
        })

    expected = {
        "validated_caption_narrative": "one coherent rich LoRA-training caption, preserving compatible concrete details without contradictions",
        "validated_caption_comma": "dense comma-separated caption with nearly the same visual content",
        "merge_notes": ["brief internal audit notes about resolved contradictions; not prose for the final caption"],
        "dropped_conflicts": ["details omitted because they conflicted or were unsupported by cleaned-caption consensus"],
    }
    trigger_word = normalize_whitespace(getattr(config, "trigger_word", "") or "")
    user_caption_anchor = normalize_whitespace(getattr(config, "user_caption_anchor", "") or "")
    instruction = normalize_whitespace(config.instructions or DEFAULT_RECONCILER_INSTRUCTIONS)
    if "reconciliation" not in instruction.lower() and "reconcile" not in instruction.lower():
        instruction = DEFAULT_RECONCILER_INSTRUCTIONS

    payload = {
        "image_key": image_key,
        "trigger_word": trigger_word,
        "user_caption_anchor": user_caption_anchor,
        "cleaned_captions": cleaned_captions,
        "required_output": expected,
    }
    prompt = (
        f"{instruction}\n\n"
        "Return strict JSON only. Do not use markdown. Do not write prose outside JSON.\n\n"
        "MERGE RULES:\n"
        "1. All cleaned captions describe the same single image.\n"
        "2. Merge them into one coherent caption with no contradictions.\n"
        "3. Preserve compatible concrete details; do not compress them away.\n"
        "4. Remove duplicate wording.\n"
        "5. Prefer repeated details, cross-model agreement, and more specific compatible wording.\n"
        "6. If a conflict cannot be resolved, omit that conflicting detail rather than choosing randomly.\n"
        "7. Do not add visual details absent from the cleaned captions.\n"
        "8. Do not mention sources, captions, models, uncertainty, validation, or this process in the final caption.\n"
        "9. Keep trigger_word and user_caption_anchor only when provided; place them naturally at the beginning if present.\n"
        "10. The narrative caption may be detailed. No compression for brevity.\n\n"
        f"Expected JSON shape:\n{json.dumps(expected, ensure_ascii=False, indent=2)}\n\n"
        f"Input JSON:\n{json.dumps(payload, ensure_ascii=False, indent=2)}\n"
    )
    return {
        "captionforge_pass": "C_RECONCILER_PROMPT",
        "engine": ENGINE_NAME,
        "engine_version": ENGINE_VERSION,
        "image_key": image_key,
        "prompt": prompt,
        "prompt_sha1": sha1_text(prompt),
        "source_caption_count": len(group),
        "timestamp": now_iso(),
    }


def _cf_parse_reconciler_response(raw: str) -> dict[str, Any]:
    obj = parse_json_object(raw)
    narrative = normalize_caption_value(
        obj.get("validated_caption_narrative")
        or obj.get("final_caption")
        or obj.get("caption")
        or obj.get("rich_caption")
        or "",
        joiner=" ",
    )
    comma = normalize_caption_value(
        obj.get("validated_caption_comma")
        or obj.get("comma_caption")
        or obj.get("taggy_caption")
        or "",
        joiner=", ",
    )
    if not narrative:
        raise ValueError("Reconciler response did not contain validated_caption_narrative/final_caption.")
    if not comma:
        comma = narrative
    return {
        "validated_caption_narrative": narrative,
        "validated_caption_comma": comma,
        "merge_notes": normalize_string_list(obj.get("merge_notes") or obj.get("notes") or [], max_items=40),
        "dropped_conflicts": normalize_string_list(obj.get("dropped_conflicts") or obj.get("removed_conflicts") or [], max_items=40),
    }


def _cf_prefix_guidance(text: str, trigger: str, anchor: str) -> str:
    text = normalize_whitespace(text)
    parts = []
    trig = normalize_whitespace(trigger)
    anch = normalize_whitespace(anchor)
    low = text.lower()
    if trig and not low.startswith(trig.lower()):
        parts.append(trig)
    if anch and anch.lower() not in low:
        parts.append(anch)
    parts.append(text)
    return normalize_whitespace(", ".join(p for p in parts if p))


def build_reconciled_record(image_key: str, group: list[dict[str, Any]], config: DistillerConfig, parsed: dict[str, Any], raw: str, prompt_bundle: dict[str, Any], status: str, error: str = "") -> dict[str, Any]:
    trigger_word = normalize_whitespace(getattr(config, "trigger_word", "") or "")
    user_caption_anchor = normalize_whitespace(getattr(config, "user_caption_anchor", "") or "")
    narrative = _cf_prefix_guidance(parsed.get("validated_caption_narrative", ""), trigger_word, user_caption_anchor)
    comma = normalize_caption_value(parsed.get("validated_caption_comma") or narrative, joiner=", ")
    comma = _cf_prefix_guidance(comma, trigger_word, user_caption_anchor)
    image = str(group[0].get("image") or image_key) if group else image_key
    return {
        "captionforge_pass": RECONCILER_PASS,
        "engine": ENGINE_NAME,
        "engine_version": ENGINE_VERSION,
        "contract": "cleaned_caption_reconciler_v0.1",
        "image_key": image_key,
        "image": image,
        "image_resolved_path": str(group[0].get("image_resolved_path") or "") if group else "",
        "status": status,
        "validated_caption_narrative": narrative,
        "validated_caption_comma": normalize_whitespace(comma),
        "trigger_word": trigger_word,
        "user_caption_anchor": user_caption_anchor,
        "removed_or_rejected_details": parsed.get("dropped_conflicts", []),
        "corrected_details": [],
        "uncertain_details": [],
        "visual_validation_notes": parsed.get("merge_notes", []),
        "validated_claims": [],
        "added_visible_details": [],
        "source_distiller": {
            "captionforge_pass": "B_VLM_CLEANED",
            "source_caption_count": len(group),
            "source_caption_records": group,
        },
        "source_cleaned_records": group,
        "params": {
            "llm_backend": config.llm_backend,
            "llm_model": config.llm_model,
            "strategy": "cleaned_caption_reconciler",
            "ollama_num_predict": config.ollama_num_predict,
            "ollama_temperature": config.ollama_temperature,
            "ollama_top_p": config.ollama_top_p,
            "ollama_top_k": config.ollama_top_k,
            "ollama_seed": int(getattr(config, "ollama_seed", -1)),
        },
        "distill_parse_result": {"status": status, "prompt_sha1": prompt_bundle.get("prompt_sha1", "")},
        "raw_distill_response": raw if config.preserve_raw_response else "",
        "warnings": [],
        "metrics": {
            "source_cleaned_caption_count": len(group),
            "validated_caption_narrative_chars": len(narrative),
            "validated_caption_comma_chars": len(comma),
            "raw_response_chars": len(raw or ""),
        },
        "timestamp": now_iso(),
        "error": error,
    }


def readable_reconciled_record(record: dict[str, Any]) -> dict[str, Any]:
    return {
        "image_key": record.get("image_key", ""),
        "trigger_word": record.get("trigger_word", ""),
        "user_caption_anchor": record.get("user_caption_anchor", ""),
        "validated_caption_narrative": record.get("validated_caption_narrative", ""),
        "validated_caption_comma": record.get("validated_caption_comma", ""),
        "visual_validation_notes": record.get("visual_validation_notes", []),
        "removed_or_rejected_details": record.get("removed_or_rejected_details", []),
    }


def process_reconciler_batch(batch: BatchConfig, config: DistillerConfig) -> int:
    """Experimental reversed-pipeline Pass C: merge B_VLM_CLEANED records into final captions."""
    groups = group_cleaned_records(read_jsonl(batch.input_jsonl))
    keys = sorted(groups)
    if batch.image_key_filter:
        keys = [k for k in keys if k == batch.image_key_filter]
    if batch.limit_images > 0:
        keys = keys[: batch.limit_images]

    out_path = Path(batch.output_jsonl) if batch.output_jsonl else default_output_path(batch.input_jsonl)
    readable_jsonl_path = Path(batch.readable_jsonl) if batch.readable_jsonl else default_readable_jsonl_path(out_path)
    readable_json_path = Path(batch.readable_json) if batch.readable_json else default_readable_json_path(out_path)
    prompt_jsonl_path = Path(batch.prompt_jsonl) if batch.prompt_jsonl else default_prompt_jsonl_path(out_path)

    if not batch.append_output and not batch.skip_existing and not batch.dry_run:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text("", encoding="utf-8")
        if not batch.no_readable_sidecars:
            readable_jsonl_path.write_text("", encoding="utf-8")
            readable_json_path.write_text("[]\n", encoding="utf-8")
        if batch.write_prompt_jsonl:
            prompt_jsonl_path.write_text("", encoding="utf-8")

    ok = failed = skipped = 0
    readable_records: list[dict[str, Any]] = []
    total = len(keys)
    print(f"[CaptionForge Reconciler] Found {total} cleaned image group(s).", flush=True)

    for i, key in enumerate(keys, start=1):
        print(f"[{i}/{total}] RECONCILE {key}", flush=True)
        group = groups[key]
        prompt_bundle = build_reconciler_prompt(key, group, config)
        if batch.write_prompt_jsonl:
            write_jsonl_record(prompt_jsonl_path, prompt_bundle, dry_run=batch.dry_run)
        try:
            if config.llm_backend == "prompt_only":
                raw = ""
                joined = " ".join(_cf_cleaned_caption(r) for r in group if _cf_cleaned_caption(r))
                parsed = {"validated_caption_narrative": joined, "validated_caption_comma": joined, "merge_notes": ["prompt_only backend; captions concatenated without LLM reconciliation."], "dropped_conflicts": []}
            else:
                raw = call_ollama(prompt_bundle["prompt"], config)
                parsed = _cf_parse_reconciler_response(raw)
            rec = build_reconciled_record(key, group, config, parsed, raw, prompt_bundle, "ok")
            ok += 1
        except Exception as exc:
            parsed = {"validated_caption_narrative": "", "validated_caption_comma": "", "merge_notes": [], "dropped_conflicts": []}
            rec = build_reconciled_record(key, group, config, parsed, "", prompt_bundle, "error", error=str(exc))
            failed += 1
            print(f"    ERROR {exc}", flush=True)

        write_jsonl_record(out_path, rec, dry_run=batch.dry_run)
        if not batch.no_readable_sidecars:
            rr = readable_reconciled_record(rec)
            write_jsonl_record(readable_jsonl_path, rr, dry_run=batch.dry_run)
            readable_records.append(rr)
            if not batch.dry_run:
                readable_json_path.write_text(json.dumps(readable_records, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"[CaptionForge Reconciler] Done. ok={ok} failed={failed} skipped={skipped} output={out_path}", flush=True)
    return 0 if failed == 0 else 1
