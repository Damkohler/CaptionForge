#!/usr/bin/env python
"""
CaptionForge Claim Extraction Engine

- CaptionForge
  - This module is part of **CaptionForge**, a model-agnostic captioning
    framework for ComfyUI developed by **J. L. Córdova**.

  - Repository
    https://github.com/Damkohler/CaptionForge

  - CaptionForge focuses on practical dataset-captioning infrastructure for:
        • LoRA dataset preparation
        • multi-engine caption generation
        • JSONL audit trails
        • claim extraction and refinement
        • consensus-oriented caption improvement

- Module Purpose
    - The **CaptionForge Claim Extraction Engine** implements Pass B of the
      CaptionForge caption-refinement pipeline.

    - Pass B consumes Pass A caption audit records and decomposes caption text
      into structured, atomic visual claims that can later be compared,
      normalized, scored, audited, and recombined into cleaner final captions.

    - Instead of treating a caption as a final answer, this engine treats each
      caption as a source of visual evidence.

    - The engine:
            • Reads shared Pass A caption JSONL records
            • Groups captions by image key
            • Preserves source-caption references
            • Extracts atomic visual claims
            • Normalizes equivalent or near-equivalent claims
            • Aggregates claim support across sources
            • Detects simple mutually exclusive attribute conflicts
            • Emits Pass B JSONL records for downstream refinement
            • Exports future-LLM prompt bundles for backend development

- Current Backend Strategy
    - v0.4.0 is a deterministic, LLM-ready scaffold.

    - Supported backends:
            • heuristic
                Deterministic rule-based claim extraction for immediate local
                testing and reproducible Pass B output.

            • manual_json
                Development backend for parsing precomputed JSON/JSONL claim
                responses using the same response shape expected from a future
                live text-only LLM backend.

    - The engine is intentionally structured so a future backend such as Ollama,
      Transformers, llama.cpp, or another local text model provider can be wired
      into the same prompt and response contract.

- Claim Model
    - Extracted claims preserve both original and normalized forms.

    - Each atomic claim tracks:
            • original claim text
            • normalized claim text
            • claim type
            • specificity
            • confidence
            • source record index
            • model family
            • model name
            • ensemble run index
            • phrase index

    - Normalized claims aggregate support across matching atomic claims and
      preserve examples, source references, model families, support counts,
      confidence labels, and conflict-relevant categories.

- Audit and Reproducibility
    - Pass B records preserve the evidence chain from final normalized claims
      back to the originating Pass A captions.

    - v0.4.0 adds parser-audit fields for LLM-readiness:
            • llm_parse_result
            • parser_warnings
            • rejected_claims
            • raw_llm_response

    - These fields allow future live LLM integrations to be tested without
      losing visibility into malformed responses, rejected claims, or prompt
      contract failures.

- Design Philosophy
    - CaptionForge treats claim extraction as an intermediate reasoning layer,
      not as a final captioning step.

    - The purpose of Pass B is to make caption evidence inspectable.

    - By decomposing captions into atomic claims, later passes can compare
      agreement across engines, detect contradictions, normalize synonyms, and
      produce captions that are closer to human-quality descriptive summaries.

- ⚠️ Development Status
    - This is Pass B v0.4.0 infrastructure.
    - The default backend is deterministic and heuristic.
    - Live LLM claim extraction is not yet integrated.
    - The prompt and response schema are intentionally present so future LLM
      backends can be added without changing the downstream Pass B record shape.

- Attribution & License
  - Concept and implementation by **J. L. Córdova**
    with development assistance from **ChatGPT (OpenAI)**.

  - Copyright (c) 2026 J. L. Córdova

  - Released under the **MIT License**.
"""

from __future__ import annotations

MANIFEST = {
    "name": "CaptionForge Claim Extraction Engine",
    "version": (0, 4, 0),
    "author": "J. L. Córdova",
    "description": (
        "CaptionForge Pass B text-only claim extraction engine for converting Pass A "
        "caption JSONL records into auditable atomic visual claims. Provides deterministic "
        "heuristic extraction, manual JSON parsing for future LLM response testing, source "
        "caption preservation, claim normalization, support aggregation, simple conflict "
        "detection, parser-audit fields, and future-LLM prompt bundle export. Designed as "
        "an LLM-ready intermediate reasoning layer for model-agnostic, consensus-oriented "
        "caption refinement workflows."
    ),
}

import argparse
import hashlib
import json
import re
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, field, is_dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable, Optional


# -------------------------------------------------------------------------
# Dataclasses
# -------------------------------------------------------------------------

@dataclass
class ClaimExtractionConfig:
    engine_name: str = "CaptionForge Claim Engine"
    engine_version: str = "0.4.0"

    # Extraction behavior.
    min_claim_chars: int = 3
    max_claim_chars: int = 120
    max_claims_per_caption: int = 80
    include_low_confidence: bool = True
    dedupe_within_caption: bool = True
    normalize_synonyms: bool = True
    detect_conflicts: bool = True

    # LLM-ready backend contract.
    # v0.3 implements "heuristic" and "manual_json".
    text_llm_backend: str = "heuristic"
    text_llm_model: str = ""
    text_llm_prompt: str = (
        "Extract atomic visible visual claims from the provided image captions. "
        "Return strict JSON only. Preserve source references. Do not add facts "
        "that are not supported by the captions."
    )
    prompt_schema_version: str = "captionforge-pass-b-prompt-v0.4"
    response_schema_version: str = "captionforge-pass-b-claims-v0.4"
    preserve_raw_llm_response: bool = False


@dataclass
class BatchClaimConfig:
    input_jsonl: str
    output_jsonl: str = ""
    write_jsonl: bool = True
    dry_run: bool = False
    limit_images: int = 0
    include_error_pass_a_records: bool = False
    overwrite: bool = True

    # v0.3 LLM-ready diagnostics / manual backend.
    manual_json_path: str = ""
    write_llm_prompt_jsonl: bool = False
    llm_prompt_jsonl: str = ""

    # v0.4 focused debugging / parser audit.
    image_key_filter: str = ""


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
class AtomicClaim:
    claim_id: str

    # v0.2+ rich claim preservation.
    original_claim: str
    normalized_claim: str
    claim_type: str
    specificity: str

    confidence: float
    source_record_index: int
    model_family: str
    model_name: str
    ensemble_run_index: int
    phrase_index: int

    # Backward-compatible aliases.
    text: str = ""
    normalized: str = ""
    category: str = ""


@dataclass
class NormalizedClaim:
    claim_id: str
    normalized_claim: str
    representative_original_claim: str
    original_claims: list[str]
    claim_type: str
    specificity: str
    support_count: int
    model_families: list[str]
    model_names: list[str]
    source_refs: list[dict[str, Any]]
    examples: list[str]
    confidence: float
    confidence_label: str

    # Backward-compatible aliases.
    normalized: str = ""
    category: str = ""


@dataclass
class ConflictRecord:
    conflict_type: str
    category: str
    values: list[str]
    support_counts: dict[str, int]
    note: str


@dataclass
class LLMParseResult:
    backend: str
    status: str
    prompt_sha1: str = ""
    prompt_schema_version: str = ""
    response_schema_version: str = ""
    parsed_claim_count: int = 0
    rejected_claim_count: int = 0
    parser_warnings: list[str] = field(default_factory=list)
    raw_response_sha1: str = ""
    error: str = ""


@dataclass
class RejectedClaim:
    source_index: int
    reason: str
    item: Any


@dataclass
class ImageClaimRecord:
    captionforge_pass: str
    image_key: str
    image: str
    status: str
    source_caption_count: int
    source_caption_records: list[SourceCaptionRef]
    atomic_claims: list[AtomicClaim]
    normalized_claims: list[NormalizedClaim]
    conflicts: list[ConflictRecord]
    uncertainty_flags: list[str]
    params: dict[str, Any]
    timestamp: str
    llm_parse_result: Optional[LLMParseResult] = None
    parser_warnings: list[str] = field(default_factory=list)
    rejected_claims: list[RejectedClaim] = field(default_factory=list)
    raw_llm_response: str = ""


@dataclass
class BatchClaimResult:
    records: list[ImageClaimRecord] = field(default_factory=list)
    skipped: int = 0
    failed: int = 0

    @property
    def processed(self) -> int:
        return len([r for r in self.records if r.status == "ok"])

    @property
    def jsonl_text(self) -> str:
        return "\n".join(json.dumps(record_to_json(r), ensure_ascii=False) for r in self.records)


# -------------------------------------------------------------------------
# Constants and normalization vocabulary
# -------------------------------------------------------------------------

BOILERPLATE_PREFIXES = [
    "the image depicts",
    "the image shows",
    "this image depicts",
    "this image shows",
    "the photo shows",
    "this photo shows",
    "a photo of",
    "a photograph of",
    "an image of",
    "in this image",
    "in this digital illustration",
    "digital illustration of",
]

STYLE_WORDS = {
    "digital illustration", "illustration", "render", "3d render", "quasi-3d render",
    "anime style", "doll-like", "stylized", "cartoon", "painting", "photograph", "photo",
    "studio lighting", "soft lighting", "dramatic lighting", "moody lighting",
    "glossy", "vinyl", "plastic", "dollora",
}

COLOR_SYNONYMS = {
    "blond": "blonde",
    "light blond": "blonde",
    "light blonde": "blonde",
    "golden": "blonde",
    "golden blond": "blonde",
    "golden blonde": "blonde",
    "platinum": "platinum blonde",
    "brunette": "brown",
    "auburn": "red",
    "ginger": "red",
    "reddish": "red",
    "grey": "gray",
    "silver": "gray",
}

HAIR_COLORS = [
    "platinum blonde", "blonde", "black", "brown", "red", "gray", "white", "pink",
    "blue", "green", "purple", "orange", "dark", "light",
]
EYE_COLORS = ["blue", "green", "brown", "hazel", "gray", "black", "red", "pink", "purple"]
BACKGROUND_COLORS = ["white", "gray", "black", "blue", "green", "red", "pink", "purple", "yellow", "orange", "brown"]

CLOTHING_HINTS = [
    "shirt", "t-shirt", "top", "tank top", "blouse", "dress", "jacket", "coat",
    "skirt", "pants", "jeans", "shorts", "bodysuit", "sweater", "hoodie", "boots",
    "shoes", "gloves", "hat", "scarf", "outfit", "clothing",
]

POSE_HINTS = [
    "standing", "sitting", "kneeling", "crouching", "leaning", "looking", "facing",
    "turned", "pose", "posed", "arms", "hands", "hand", "shoulder", "profile",
    "three-quarter", "over one shoulder",
]

SHOT_HINTS = [
    "close-up", "extreme close-up", "medium shot", "upper body", "portrait",
    "full body", "cowboy shot", "above knee", "waist-up", "headshot",
]

MUTUALLY_EXCLUSIVE_CATEGORIES = {"hair_color", "eye_color", "background_color"}


# -------------------------------------------------------------------------
# Serialization and file helpers
# -------------------------------------------------------------------------

def iso_timestamp() -> str:
    return datetime.now().isoformat(timespec="seconds")


def json_safe(value: Any) -> Any:
    if is_dataclass(value):
        return {k: json_safe(v) for k, v in asdict(value).items()}
    if isinstance(value, set):
        return sorted(value)
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {k: json_safe(v) for k, v in value.items()}
    if isinstance(value, list):
        return [json_safe(v) for v in value]
    if isinstance(value, tuple):
        return [json_safe(v) for v in value]
    return value


def record_to_json(record: Any) -> dict[str, Any]:
    return json_safe(record)


def safe_mkdir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def stable_claim_id(*parts: str) -> str:
    text = "::".join(str(p) for p in parts)
    return hashlib.sha1(text.encode("utf-8")).hexdigest()[:16]


def sha1_text(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    p = Path(path)
    if p.is_dir():
        raise IsADirectoryError(
            f"input_jsonl points to a folder, not a JSONL file: {p}\n"
            "Select the Pass A captions JSONL file, for example captions.jsonl."
        )
    if not p.exists():
        raise FileNotFoundError(f"JSONL file does not exist: {p}")

    records: list[dict[str, Any]] = []
    with p.open("r", encoding="utf-8") as f:
        for line_number, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSONL at {p}:{line_number}: {exc}") from exc
            if isinstance(obj, dict):
                records.append(obj)
            else:
                raise ValueError(f"JSONL record at {p}:{line_number} is not an object.")
    return records


def write_jsonl(path: str | Path, records: Iterable[Any], overwrite: bool = True, dry_run: bool = False) -> None:
    if dry_run:
        return
    p = Path(path)
    safe_mkdir(p.parent)
    mode = "w" if overwrite else "a"
    with p.open(mode, encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record_to_json(record), ensure_ascii=False) + "\n")


def default_output_path(input_jsonl: str | Path) -> Path:
    p = Path(input_jsonl)
    if p.is_dir():
        raise IsADirectoryError(
            f"input_jsonl points to a folder, not a JSONL file: {p}\n"
            "Select the Pass A captions JSONL file, for example captions.jsonl."
        )
    stem = p.stem or "captions"
    return p.with_name(f"{stem}_claims_pass_b.jsonl")


def default_prompt_output_path(input_jsonl: str | Path) -> Path:
    p = Path(input_jsonl)
    if p.is_dir():
        raise IsADirectoryError(
            f"input_jsonl points to a folder, not a JSONL file: {p}\n"
            "Select the Pass A captions JSONL file, for example captions.jsonl."
        )
    stem = p.stem or "captions"
    return p.with_name(f"{stem}_pass_b_llm_prompts.jsonl")


# -------------------------------------------------------------------------
# Pass A grouping
# -------------------------------------------------------------------------

def pass_a_record_is_usable(record: dict[str, Any], include_errors: bool = False) -> bool:
    if str(record.get("captionforge_pass") or "").upper() not in {"", "A"}:
        return False
    status = str(record.get("status") or "ok").lower()
    if status != "ok" and not include_errors:
        return False
    caption = str(record.get("caption") or record.get("raw_caption") or "").strip()
    if not caption and status == "ok":
        return False
    return True


def get_image_key(record: dict[str, Any]) -> str:
    return str(record.get("image_key") or record.get("image") or record.get("source") or "").strip()


def group_pass_a_records(records: list[dict[str, Any]], include_errors: bool = False) -> dict[str, list[dict[str, Any]]]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for index, record in enumerate(records):
        if not pass_a_record_is_usable(record, include_errors=include_errors):
            continue
        key = get_image_key(record)
        if not key:
            continue
        enriched = dict(record)
        enriched["_source_record_index"] = index
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


# -------------------------------------------------------------------------
# LLM prompt contract
# -------------------------------------------------------------------------

def build_llm_prompt_bundle(
    image_key: str,
    source_refs: list[SourceCaptionRef],
    config: ClaimExtractionConfig,
) -> dict[str, Any]:
    captions = [
        {
            "source_record_index": src.source_record_index,
            "model_family": src.model_family,
            "model_name": src.model_name,
            "ensemble_run_index": src.ensemble_run_index,
            "caption": src.caption or src.raw_caption,
        }
        for src in source_refs
    ]

    instructions = config.text_llm_prompt.strip() or ClaimExtractionConfig().text_llm_prompt
    response_contract = {
        "claims": [
            {
                "claim_type": "hair_color | eye_color | clothing | pose_action | expression | style_medium | background_color | background_object | composition | visible_text | general",
                "original_claim": "short atomic visible claim copied or minimally rewritten from captions",
                "normalized_claim": "canonical comparison form; preserve useful specificity when uncertain",
                "specificity": "generic | specific | specific_normalized",
                "certainty": "visible | inferred_from_caption | uncertain",
                "source_record_indexes": [0]
            }
        ]
    }

    prompt_text = (
        f"{instructions}\n\n"
        "Return strict JSON only, with no markdown fences and no explanatory prose.\n"
        "Do not add any visual fact that is not supported by the captions.\n"
        "Keep claims atomic: one visible attribute/action/style/background fact per claim.\n\n"
        f"Expected JSON response shape:\n{json.dumps(response_contract, ensure_ascii=False, indent=2)}\n\n"
        f"Image key: {image_key}\n"
        f"Source captions:\n{json.dumps(captions, ensure_ascii=False, indent=2)}\n"
    )

    return {
        "captionforge_pass": "B_prompt",
        "prompt_schema_version": config.prompt_schema_version,
        "image_key": image_key,
        "prompt": prompt_text,
        "prompt_sha1": sha1_text(prompt_text),
        "source_captions": captions,
        "timestamp": iso_timestamp(),
    }


def load_manual_claims(path: str | Path) -> dict[str, Any]:
    if not str(path).strip():
        return {}

    p = Path(path)
    if p.is_dir():
        raise IsADirectoryError(f"manual_json_path points to a folder, not a JSON/JSONL file: {p}")
    if not p.exists():
        raise FileNotFoundError(f"manual_json_path does not exist: {p}")

    by_key: dict[str, Any] = {}

    if p.suffix.lower() == ".jsonl":
        with p.open("r", encoding="utf-8") as f:
            for line_number, line in enumerate(f, start=1):
                line = line.strip()
                if not line:
                    continue
                obj = json.loads(line)
                if not isinstance(obj, dict):
                    raise ValueError(f"manual_json record at {p}:{line_number} is not an object.")
                key = str(obj.get("image_key") or "").strip()
                if not key:
                    key = "__single__" if "__single__" not in by_key else f"__single__:{line_number}"
                by_key[key] = obj
        return by_key

    obj = json.loads(p.read_text(encoding="utf-8"))
    if isinstance(obj, dict):
        if "image_key" in obj:
            by_key[str(obj.get("image_key") or "__single__")] = obj
        elif "claims" in obj:
            by_key["__single__"] = obj
        else:
            # Treat as mapping: image_key -> response object
            for key, value in obj.items():
                by_key[str(key)] = value
        return by_key

    if isinstance(obj, list):
        for idx, item in enumerate(obj):
            if not isinstance(item, dict):
                raise ValueError(f"manual_json list item {idx} is not an object.")
            key = str(item.get("image_key") or f"__single__:{idx}").strip()
            by_key[key] = item
        return by_key

    raise ValueError("manual_json_path must contain a JSON object/list or JSONL objects.")


def extract_json_object(text: str) -> dict[str, Any]:
    text = text.strip()
    if not text:
        raise ValueError("Empty LLM/manual JSON response.")
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.IGNORECASE)
        text = re.sub(r"\s*```$", "", text)
    obj = json.loads(text)
    if not isinstance(obj, dict):
        raise ValueError("LLM/manual JSON response must be a JSON object.")
    return obj


def normalize_manual_response(value: Any) -> dict[str, Any]:
    if isinstance(value, str):
        return extract_json_object(value)
    if isinstance(value, dict):
        if "claims" in value:
            return value
        # Accept direct mapping with response nested under "response".
        if "response" in value:
            return normalize_manual_response(value["response"])
    raise ValueError("Manual response must be an object with a 'claims' list or a JSON string.")


def manual_claims_for_image(image_key: str, manual_by_key: dict[str, Any]) -> Optional[dict[str, Any]]:
    if not manual_by_key:
        return None
    if image_key in manual_by_key:
        return normalize_manual_response(manual_by_key[image_key])
    if "__single__" in manual_by_key and len(manual_by_key) == 1:
        return normalize_manual_response(manual_by_key["__single__"])
    return None


# -------------------------------------------------------------------------
# Text splitting / normalization
# -------------------------------------------------------------------------

def normalize_whitespace(text: str) -> str:
    text = text.replace("\n", " ")
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"\s+([,.;:!?])", r"\1", text)
    return text.strip()


def strip_boilerplate(text: str) -> str:
    s = normalize_whitespace(text)
    lowered = s.lower()
    for prefix in BOILERPLATE_PREFIXES:
        if lowered.startswith(prefix):
            return s[len(prefix):].lstrip(" ,:-")
    return s


def split_caption_to_phrases(caption: str) -> list[str]:
    caption = strip_boilerplate(caption)
    raw_parts = re.split(r"[,;•\n]+|(?<=[.!?])\s+", caption)

    phrases: list[str] = []
    for raw in raw_parts:
        p = normalize_whitespace(raw).strip(" .;:,")
        if not p:
            continue
        subparts = re.split(r"\s+(?:and|while|with)\s+", p, flags=re.IGNORECASE)
        if len(subparts) > 1:
            for sub in subparts:
                sub = normalize_whitespace(sub).strip(" .;:,")
                if len(sub) >= 3:
                    phrases.append(sub)
        else:
            phrases.append(p)
    return phrases


def canonical_color(value: str) -> str:
    v = value.lower().strip()
    v = re.sub(r"[-_]+", " ", v)
    v = re.sub(r"\s+", " ", v)
    return COLOR_SYNONYMS.get(v, v)


def clean_claim_text(text: str) -> str:
    text = strip_boilerplate(text)
    text = text.strip().strip(" .;:,")
    text = re.sub(r"^(?:and|with|while|a|an|the)\s+", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\b(a|an)\s+(a|an|the)\b", r"\2", text, flags=re.IGNORECASE)
    text = normalize_whitespace(text)
    return text


def singularize_light(text: str) -> str:
    replacements = {
        "eyes": "eye",
        "hands": "hand",
        "arms": "arm",
        "legs": "leg",
        "buns": "bun",
        "gloves": "glove",
        "boots": "boot",
        "shoes": "shoe",
    }
    words = text.split()
    return " ".join(replacements.get(w, w) for w in words)


def normalize_claim_text(text: str, config: ClaimExtractionConfig) -> tuple[str, str, str]:
    """
    Return (normalized_text, claim_type, specificity).
    """
    raw = clean_claim_text(text)
    lower = raw.lower()
    lower = lower.replace("t shirt", "t-shirt")
    lower = re.sub(r"\s+", " ", lower).strip()

    hair_match = re.search(
        r"\b((?:light|dark|golden|platinum|blond|blonde|black|brown|brunette|red|reddish|auburn|ginger|gray|grey|white|pink|blue|green|purple|orange)(?:\s+blond|\s+blonde)?)\s+hair\b",
        lower,
    )
    if hair_match:
        color_raw = hair_match.group(1)
        color = canonical_color(color_raw)
        specificity = "specific_normalized" if color != color_raw else "specific"
        return f"{color} hair", "hair_color", specificity

    eye_match = re.search(r"\b(blue|green|brown|hazel|gray|grey|black|red|pink|purple)\s+eyes?\b", lower)
    if eye_match:
        color_raw = eye_match.group(1)
        color = canonical_color(color_raw)
        specificity = "specific_normalized" if color != color_raw else "specific"
        return f"{color} eyes", "eye_color", specificity

    bg_match = re.search(
        r"\b(white|gray|grey|black|blue|green|red|pink|purple|yellow|orange|brown)\s+(?:background|backdrop|wall)\b",
        lower,
    )
    if bg_match:
        color_raw = bg_match.group(1)
        color = canonical_color(color_raw)
        specificity = "specific_normalized" if color != color_raw else "specific"
        return f"{color} background", "background_color", specificity

    for hint in CLOTHING_HINTS:
        if re.search(rf"\b{re.escape(hint)}\b", lower):
            return singularize_light(lower), "clothing", "specific"

    for hint in POSE_HINTS:
        if re.search(rf"\b{re.escape(hint)}\b", lower):
            return lower, "pose_action", "specific"

    for hint in SHOT_HINTS:
        if re.search(rf"\b{re.escape(hint)}\b", lower):
            return lower, "composition", "specific"

    for hint in STYLE_WORDS:
        if hint in lower:
            return lower, "style_medium", "specific"

    if '"' in raw or " text " in f" {lower} " or "reads" in lower or "says" in lower:
        return lower, "visible_text", "specific"

    if any(w in lower for w in ["poster", "guitar", "graffiti", "wall", "floor", "room", "background", "backdrop"]):
        return lower, "background_object", "specific"

    if any(w in lower for w in ["hair", "hairstyle", "bun", "braid", "ponytail", "bangs"]):
        return lower, "hair_style", "specific"

    if any(w in lower for w in ["expression", "smile", "smiling", "serious", "neutral", "half-lidded", "brows", "mouth"]):
        return lower, "expression", "specific"

    return lower, "general", "generic"


def phrase_confidence(phrase: str, normalized: str, claim_type: str) -> float:
    lower = phrase.lower()
    confidence = 0.65

    if claim_type not in {"general"}:
        confidence += 0.15
    if any(token in lower for token in ["maybe", "possibly", "appears to", "seems to", "likely"]):
        confidence -= 0.25
    if len(normalized.split()) <= 2:
        confidence -= 0.05
    if '"' in phrase:
        confidence += 0.05

    return max(0.05, min(0.95, round(confidence, 2)))


def confidence_label(confidence: float, support_count: int, model_families: list[str]) -> str:
    if support_count >= 2 and len(set(model_families)) >= 2:
        return "cross_model_supported"
    if support_count >= 2:
        return "multi_source_supported"
    if confidence >= 0.75:
        return "single_source_high"
    return "single_source"


def should_keep_claim(text: str, normalized: str, config: ClaimExtractionConfig) -> bool:
    if len(normalized) < config.min_claim_chars:
        return False
    if len(normalized) > config.max_claim_chars:
        return False
    if normalized in {"image", "photo", "illustration", "the scene", "scene"}:
        return False
    if re.fullmatch(r"\W+", normalized):
        return False
    return True


# -------------------------------------------------------------------------
# Extraction / aggregation
# -------------------------------------------------------------------------

def make_atomic_claim(
    *,
    image_key: str,
    original_claim: str,
    normalized_claim: str,
    claim_type: str,
    specificity: str,
    confidence: float,
    source: SourceCaptionRef,
    phrase_index: int,
) -> AtomicClaim:
    claim_id = stable_claim_id(image_key, str(source.source_record_index), normalized_claim, original_claim, str(phrase_index))
    return AtomicClaim(
        claim_id=claim_id,
        original_claim=original_claim,
        normalized_claim=normalized_claim,
        claim_type=claim_type,
        specificity=specificity,
        confidence=confidence,
        source_record_index=source.source_record_index,
        model_family=source.model_family,
        model_name=source.model_name,
        ensemble_run_index=source.ensemble_run_index,
        phrase_index=phrase_index,
        text=original_claim,
        normalized=normalized_claim,
        category=claim_type,
    )


def extract_claims_from_caption(caption: str, source: SourceCaptionRef, config: ClaimExtractionConfig) -> list[AtomicClaim]:
    phrases = split_caption_to_phrases(caption)
    claims: list[AtomicClaim] = []
    seen_norms: set[str] = set()

    for phrase_index, phrase in enumerate(phrases):
        original = clean_claim_text(phrase)
        if not original:
            continue

        normalized, claim_type, specificity = normalize_claim_text(original, config)
        if not should_keep_claim(original, normalized, config):
            continue

        if config.dedupe_within_caption and normalized in seen_norms:
            continue
        seen_norms.add(normalized)

        confidence = phrase_confidence(original, normalized, claim_type)
        if not config.include_low_confidence and confidence < 0.5:
            continue

        claims.append(
            make_atomic_claim(
                image_key=source.image_key,
                original_claim=original,
                normalized_claim=normalized,
                claim_type=claim_type,
                specificity=specificity,
                confidence=confidence,
                source=source,
                phrase_index=phrase_index,
            )
        )

        if len(claims) >= config.max_claims_per_caption:
            break

    return claims


def claims_from_manual_response(
    image_key: str,
    response: dict[str, Any],
    source_refs: list[SourceCaptionRef],
    config: ClaimExtractionConfig,
) -> tuple[list[AtomicClaim], LLMParseResult, list[RejectedClaim], str]:
    """
    Parse a manual/future-LLM JSON response into AtomicClaim objects.

    v0.4 intentionally avoids crashing on per-claim mistakes. Bad claim items are
    collected in rejected_claims and parser_warnings so the prompt/schema can be
    improved before a live LLM backend is wired in.
    """
    claims_obj = response.get("claims")
    if not isinstance(claims_obj, list):
        raise ValueError("Manual/LLM response must contain a 'claims' list.")

    raw_response = response.get("raw_llm_response")
    if raw_response is None:
        raw_response = json.dumps(response, ensure_ascii=False, sort_keys=True)
    else:
        raw_response = str(raw_response)

    source_by_index = {src.source_record_index: src for src in source_refs}
    default_source = source_refs[0] if source_refs else SourceCaptionRef(0, image_key, image_key, "", "", 0, "ok", "", "", "")

    claims: list[AtomicClaim] = []
    rejected: list[RejectedClaim] = []
    warnings: list[str] = []

    for idx, item in enumerate(claims_obj):
        if not isinstance(item, dict):
            rejected.append(RejectedClaim(idx, "claim item is not an object", item))
            continue

        original = clean_claim_text(str(item.get("original_claim") or item.get("claim") or item.get("text") or ""))
        if not original:
            rejected.append(RejectedClaim(idx, "missing original_claim/claim/text", item))
            continue

        if len(original) < config.min_claim_chars:
            rejected.append(RejectedClaim(idx, f"claim shorter than min_claim_chars={config.min_claim_chars}", item))
            continue
        if len(original) > config.max_claim_chars:
            rejected.append(RejectedClaim(idx, f"claim longer than max_claim_chars={config.max_claim_chars}", item))
            continue

        normalized_raw = str(item.get("normalized_claim") or item.get("normalized") or "").strip()
        claim_type_raw = str(item.get("claim_type") or item.get("category") or "").strip()
        specificity = str(item.get("specificity") or "specific").strip() or "specific"

        if normalized_raw:
            normalized = clean_claim_text(normalized_raw).lower()
            if not normalized:
                normalized, claim_type, norm_specificity = normalize_claim_text(original, config)
                if not item.get("specificity"):
                    specificity = norm_specificity
            else:
                claim_type = claim_type_raw or normalize_claim_text(original, config)[1]
        else:
            normalized, claim_type, norm_specificity = normalize_claim_text(original, config)
            if not item.get("specificity"):
                specificity = norm_specificity

        if not claim_type:
            claim_type = "general"

        # Accept source_record_indexes, source_record_index, or no source.
        src_indexes = item.get("source_record_indexes")
        if src_indexes is None:
            src_indexes = item.get("source_record_index")
        if src_indexes is None:
            src_indexes = [default_source.source_record_index]
            warnings.append(f"claims[{idx}] missing source_record_indexes; defaulted to {default_source.source_record_index}.")
        if not isinstance(src_indexes, list):
            src_indexes = [src_indexes]

        confidence = item.get("confidence")
        try:
            confidence_f = float(confidence)
        except Exception:
            confidence_f = 0.80
        if confidence_f < 0.0 or confidence_f > 1.0:
            warnings.append(f"claims[{idx}] confidence {confidence_f!r} outside 0..1; clamped.")
            confidence_f = max(0.0, min(1.0, confidence_f))

        made_any = False
        for local_phrase_index, src_index_value in enumerate(src_indexes):
            try:
                src_index = int(src_index_value)
            except Exception:
                warnings.append(f"claims[{idx}] has non-integer source index {src_index_value!r}; defaulted.")
                src_index = default_source.source_record_index

            source = source_by_index.get(src_index)
            if source is None:
                warnings.append(f"claims[{idx}] source index {src_index!r} not found in Pass A records; defaulted.")
                source = default_source

            claims.append(
                make_atomic_claim(
                    image_key=image_key,
                    original_claim=original,
                    normalized_claim=normalized,
                    claim_type=claim_type,
                    specificity=specificity,
                    confidence=confidence_f,
                    source=source,
                    phrase_index=idx * 1000 + local_phrase_index,
                )
            )
            made_any = True

        if not made_any:
            rejected.append(RejectedClaim(idx, "no usable source_record_indexes", item))

    status = "ok" if not rejected else "ok_with_rejected_claims"
    parse_result = LLMParseResult(
        backend="manual_json",
        status=status,
        response_schema_version=config.response_schema_version,
        parsed_claim_count=len(claims),
        rejected_claim_count=len(rejected),
        parser_warnings=warnings,
        raw_response_sha1=sha1_text(raw_response) if raw_response else "",
    )
    return claims, parse_result, rejected, raw_response

def aggregate_claims(image_key: str, claims: list[AtomicClaim]) -> list[NormalizedClaim]:
    buckets: dict[str, list[AtomicClaim]] = defaultdict(list)
    for claim in claims:
        buckets[claim.normalized_claim].append(claim)

    normalized_claims: list[NormalizedClaim] = []

    for normalized, items in sorted(buckets.items(), key=lambda kv: (-len(kv[1]), kv[0])):
        category_counts = Counter(item.claim_type for item in items)
        claim_type = category_counts.most_common(1)[0][0]

        source_refs = [
            {
                "source_record_index": item.source_record_index,
                "model_family": item.model_family,
                "model_name": item.model_name,
                "ensemble_run_index": item.ensemble_run_index,
                "phrase_index": item.phrase_index,
                "claim_id": item.claim_id,
            }
            for item in items
        ]

        original_claims: list[str] = []
        for item in items:
            if item.original_claim not in original_claims:
                original_claims.append(item.original_claim)

        examples = original_claims[:3]
        model_families = sorted({item.model_family for item in items if item.model_family})
        model_names = sorted({item.model_name for item in items if item.model_name})
        avg_confidence = round(sum(item.confidence for item in items) / max(1, len(items)), 3)
        specificity_counts = Counter(item.specificity for item in items)
        specificity = specificity_counts.most_common(1)[0][0]

        normalized_claims.append(
            NormalizedClaim(
                claim_id=stable_claim_id(image_key, normalized),
                normalized_claim=normalized,
                representative_original_claim=original_claims[0] if original_claims else normalized,
                original_claims=original_claims,
                claim_type=claim_type,
                specificity=specificity,
                support_count=len(items),
                model_families=model_families,
                model_names=model_names,
                source_refs=source_refs,
                examples=examples,
                confidence=avg_confidence,
                confidence_label=confidence_label(avg_confidence, len(items), model_families),
                normalized=normalized,
                category=claim_type,
            )
        )

    return normalized_claims


def extract_value_for_conflict(normalized: str, category: str) -> str:
    if category == "hair_color" and normalized.endswith(" hair"):
        return normalized[:-5].strip()
    if category == "eye_color" and normalized.endswith(" eyes"):
        return normalized[:-5].strip()
    if category == "background_color" and normalized.endswith(" background"):
        return normalized[:-11].strip()
    return normalized


def detect_conflicts(normalized_claims: list[NormalizedClaim]) -> list[ConflictRecord]:
    conflicts: list[ConflictRecord] = []

    by_category: dict[str, list[NormalizedClaim]] = defaultdict(list)
    for claim in normalized_claims:
        if claim.claim_type in MUTUALLY_EXCLUSIVE_CATEGORIES:
            by_category[claim.claim_type].append(claim)

    for category, items in by_category.items():
        value_counts: Counter[str] = Counter()
        for item in items:
            value = extract_value_for_conflict(item.normalized_claim, category)
            value_counts[value] += item.support_count

        if len(value_counts) > 1:
            conflicts.append(
                ConflictRecord(
                    conflict_type="mutually_exclusive_attribute",
                    category=category,
                    values=sorted(value_counts.keys()),
                    support_counts=dict(sorted(value_counts.items())),
                    note=f"Multiple {category} values were extracted from the captions for this image.",
                )
            )

    return conflicts


def make_uncertainty_flags(source_refs: list[SourceCaptionRef], normalized_claims: list[NormalizedClaim], conflicts: list[ConflictRecord]) -> list[str]:
    flags: list[str] = []

    if len(source_refs) < 2:
        flags.append("single_caption_source")
    if conflicts:
        flags.append("conflicts_detected")

    supported_by_multiple_models = [
        c for c in normalized_claims
        if len(set(c.model_families)) >= 2 or c.support_count >= 2
    ]
    if not supported_by_multiple_models and normalized_claims:
        flags.append("no_cross_model_claim_support")

    low_support = [c for c in normalized_claims if c.support_count == 1]
    if normalized_claims and len(low_support) / len(normalized_claims) > 0.75:
        flags.append("mostly_single_source_claims")

    return flags


def build_image_claim_record(
    image_key: str,
    pass_a_records: list[dict[str, Any]],
    config: ClaimExtractionConfig,
    *,
    manual_response: Optional[dict[str, Any]] = None,
) -> ImageClaimRecord:
    source_refs = [make_source_ref(r) for r in pass_a_records]
    prompt_bundle = build_llm_prompt_bundle(image_key, source_refs, config)
    prompt_sha1 = str(prompt_bundle.get("prompt_sha1") or "")

    llm_parse_result: LLMParseResult
    rejected_claims: list[RejectedClaim] = []
    raw_llm_response = ""

    if config.text_llm_backend == "manual_json":
        if manual_response is None:
            all_claims = []
            llm_parse_result = LLMParseResult(
                backend="manual_json",
                status="missing_manual_response",
                prompt_sha1=prompt_sha1,
                prompt_schema_version=config.prompt_schema_version,
                response_schema_version=config.response_schema_version,
                parsed_claim_count=0,
                error=f"No manual JSON response found for image_key={image_key!r}.",
            )
        else:
            all_claims, llm_parse_result, rejected_claims, raw_llm_response = claims_from_manual_response(image_key, manual_response, source_refs, config)
            llm_parse_result.prompt_sha1 = prompt_sha1
            llm_parse_result.prompt_schema_version = config.prompt_schema_version
    else:
        all_claims: list[AtomicClaim] = []
        for source in source_refs:
            caption = source.caption or source.raw_caption
            all_claims.extend(extract_claims_from_caption(caption, source, config))
        llm_parse_result = LLMParseResult(
            backend="heuristic",
            status="ok",
            prompt_sha1=prompt_sha1,
            prompt_schema_version=config.prompt_schema_version,
            response_schema_version=config.response_schema_version,
            parsed_claim_count=len(all_claims),
        )

    normalized_claims = aggregate_claims(image_key, all_claims)
    conflicts = detect_conflicts(normalized_claims) if config.detect_conflicts else []
    uncertainty_flags = make_uncertainty_flags(source_refs, normalized_claims, conflicts)

    if llm_parse_result.status not in {"ok"}:
        uncertainty_flags.append(llm_parse_result.status)
    if llm_parse_result.parser_warnings:
        uncertainty_flags.append("parser_warnings_present")

    image = source_refs[0].image if source_refs else image_key
    status = "ok"

    return ImageClaimRecord(
        captionforge_pass="B",
        image_key=image_key,
        image=image,
        status=status,
        source_caption_count=len(source_refs),
        source_caption_records=source_refs,
        atomic_claims=all_claims,
        normalized_claims=normalized_claims,
        conflicts=conflicts,
        uncertainty_flags=uncertainty_flags,
        params={
            "engine": config.engine_name,
            "engine_version": config.engine_version,
            "text_llm_backend": config.text_llm_backend,
            "text_llm_model": config.text_llm_model,
            "text_llm_prompt": config.text_llm_prompt,
            "prompt_schema_version": config.prompt_schema_version,
            "response_schema_version": config.response_schema_version,
            "prompt_sha1": prompt_sha1,
            "config": record_to_json(config),
        },
        timestamp=iso_timestamp(),
        llm_parse_result=llm_parse_result,
        parser_warnings=list(llm_parse_result.parser_warnings),
        rejected_claims=rejected_claims,
        raw_llm_response=raw_llm_response if config.preserve_raw_llm_response else "",
    )


def extract_claims_batch(
    batch: BatchClaimConfig,
    config: Optional[ClaimExtractionConfig] = None,
) -> BatchClaimResult:
    config = config or ClaimExtractionConfig()
    config.text_llm_backend = (config.text_llm_backend or "heuristic").strip().lower()

    if config.text_llm_backend not in {"heuristic", "manual_json"}:
        raise ValueError(
            f"Unsupported text_llm_backend={config.text_llm_backend!r}. "
            "v0.4 supports: heuristic, manual_json."
        )

    result = BatchClaimResult()

    input_records = read_jsonl(batch.input_jsonl)
    groups = group_pass_a_records(input_records, include_errors=batch.include_error_pass_a_records)

    items = list(groups.items())

    image_key_filter = (batch.image_key_filter or "").strip()
    if image_key_filter:
        needle = image_key_filter.lower()
        exact = [(k, v) for k, v in items if k == image_key_filter]
        items = exact if exact else [(k, v) for k, v in items if needle in k.lower()]

    if batch.limit_images and batch.limit_images > 0:
        items = items[: int(batch.limit_images)]

    manual_by_key = load_manual_claims(batch.manual_json_path) if config.text_llm_backend == "manual_json" else {}

    prompt_bundles: list[dict[str, Any]] = []
    for image_key, grouped_records in items:
        source_refs = [make_source_ref(r) for r in grouped_records]
        prompt_bundles.append(build_llm_prompt_bundle(image_key, source_refs, config))

    if batch.write_llm_prompt_jsonl:
        prompt_path = Path(batch.llm_prompt_jsonl) if batch.llm_prompt_jsonl.strip() else default_prompt_output_path(batch.input_jsonl)
        write_jsonl(prompt_path, prompt_bundles, overwrite=True, dry_run=batch.dry_run)

    for image_key, grouped_records in items:
        try:
            manual_response = manual_claims_for_image(image_key, manual_by_key) if config.text_llm_backend == "manual_json" else None
            result.records.append(build_image_claim_record(image_key, grouped_records, config, manual_response=manual_response))
        except Exception as exc:
            result.failed += 1
            result.records.append(
                ImageClaimRecord(
                    captionforge_pass="B",
                    image_key=image_key,
                    image=str(grouped_records[0].get("image") or image_key) if grouped_records else image_key,
                    status="error",
                    source_caption_count=len(grouped_records),
                    source_caption_records=[],
                    atomic_claims=[],
                    normalized_claims=[],
                    conflicts=[],
                    uncertainty_flags=[f"error: {exc}"],
                    params={
                        "engine": config.engine_name,
                        "engine_version": config.engine_version,
                        "text_llm_backend": config.text_llm_backend,
                        "config": record_to_json(config),
                    },
                    timestamp=iso_timestamp(),
                    llm_parse_result=LLMParseResult(
                        backend=config.text_llm_backend,
                        status="error",
                        error=str(exc),
                    ),
                )
            )

    if batch.write_jsonl:
        out = Path(batch.output_jsonl) if batch.output_jsonl.strip() else default_output_path(batch.input_jsonl)
        write_jsonl(out, result.records, overwrite=batch.overwrite, dry_run=batch.dry_run)

    return result


# -------------------------------------------------------------------------
# CLI
# -------------------------------------------------------------------------

def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="CaptionForge Pass B claim extraction from Pass A caption JSONL.")
    parser.add_argument("--input-jsonl", required=True, help="Path to Pass A shared captions JSONL.")
    parser.add_argument("--output-jsonl", default="", help="Path to Pass B claims JSONL. Defaults beside input.")
    parser.add_argument("--limit-images", type=int, default=0, help="Process only the first N grouped images. 0 = no limit.")
    parser.add_argument("--dry-run", action="store_true", help="Run extraction without writing JSONL.")
    parser.add_argument("--append", action="store_true", help="Append to output JSONL instead of overwriting.")
    parser.add_argument("--include-error-pass-a-records", action="store_true", help="Include Pass A records with status != ok.")
    parser.add_argument("--no-conflicts", action="store_true", help="Disable simple conflict detection.")
    parser.add_argument("--no-low-confidence", action="store_true", help="Drop lower-confidence heuristic claims.")
    parser.add_argument("--text-llm-backend", default="heuristic", choices=["heuristic", "manual_json"])
    parser.add_argument("--manual-json-path", default="", help="JSON/JSONL response file for manual_json backend.")
    parser.add_argument("--image-key-filter", default="", help="Process only image_keys that exactly match or contain this string.")
    parser.add_argument("--preserve-raw-llm-response", action="store_true", help="Store raw manual/LLM response text in each Pass B record.")
    parser.add_argument("--write-llm-prompt-jsonl", action="store_true", help="Export future-LLM prompt bundles.")
    parser.add_argument("--llm-prompt-jsonl", default="", help="Path for exported future-LLM prompt bundles.")
    return parser


def main(argv: Optional[list[str]] = None) -> int:
    args = build_arg_parser().parse_args(argv)

    config = ClaimExtractionConfig(
        include_low_confidence=not args.no_low_confidence,
        detect_conflicts=not args.no_conflicts,
        text_llm_backend=args.text_llm_backend,
        preserve_raw_llm_response=bool(args.preserve_raw_llm_response),
    )

    output_jsonl = args.output_jsonl or str(default_output_path(args.input_jsonl))

    batch = BatchClaimConfig(
        input_jsonl=args.input_jsonl,
        output_jsonl=output_jsonl,
        write_jsonl=True,
        dry_run=bool(args.dry_run),
        limit_images=int(args.limit_images),
        include_error_pass_a_records=bool(args.include_error_pass_a_records),
        overwrite=not bool(args.append),
        manual_json_path=args.manual_json_path,
        write_llm_prompt_jsonl=bool(args.write_llm_prompt_jsonl),
        llm_prompt_jsonl=args.llm_prompt_jsonl,
        image_key_filter=args.image_key_filter,
    )

    result = extract_claims_batch(batch, config)

    print(
        f"[CaptionForge Claim Engine] processed={result.processed}, "
        f"failed={result.failed}, output={output_jsonl}"
    )

    if args.dry_run:
        print(result.jsonl_text)

    return 0 if result.failed == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
