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
    - v0.6.5 is a deterministic, LLM-ready scaffold with profile-loaded semantic taxonomy, normalization rules, and claim-routing rules.

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

    - v0.5.0 added parser-audit fields for LLM-readiness:
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
    - This is Pass B v0.6.5 infrastructure.
    - The default backend is deterministic and heuristic.
    - Live Ollama LLM claim extraction is integrated; heuristic/manual backends remain available.
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
    "version": (0, 6, 5),
    "author": "J. L. Córdova",
    "description": (
        "CaptionForge Pass B text-only claim extraction engine for converting Pass A "
        "caption JSONL records into auditable atomic visual claims. Provides deterministic "
        "heuristic extraction, manual JSON parsing for future LLM response testing, source "
        "caption preservation, claim normalization, support aggregation, simple conflict "
        "detection, parser-audit fields, and Ollama backend support, incremental JSONL writes, profile-loaded semantic taxonomy, normalization rules, and claim-routing rules, future-LLM prompt bundle export. Designed as "
        "an LLM-ready intermediate reasoning layer for model-agnostic, consensus-oriented "
        "caption refinement workflows."
    ),
}

import argparse
import hashlib
import json
import re
import urllib.error
import urllib.request
import time
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
    engine_version: str = "0.6.5"

    # Extraction behavior.
    min_claim_chars: int = 3
    max_claim_chars: int = 180
    max_claims_per_caption: int = 80
    # v0.6.5: LLM output pressure guard. This caps the requested number of
    # claims per image in the prompt, and the parser also truncates overlong
    # valid responses defensively. 0 disables the LLM-specific cap.
    max_llm_claims_per_image: int = 40
    include_low_confidence: bool = True
    dedupe_within_caption: bool = True
    normalize_synonyms: bool = True
    detect_conflicts: bool = True

    # LLM-ready backend contract.
    # v0.5 implements "heuristic", "manual_json", and "ollama".
    text_llm_backend: str = "heuristic"
    text_llm_model: str = ""
    text_llm_prompt: str = (
        "Extract clean, atomic, visible visual claims from the provided image captions. "
        "Treat the captions as imperfect evidence: preserve the visual meaning, but correct "
        "obvious grammar errors and typos in the claim text. Return strict JSON only. "
        "Preserve source references. Do not add facts that are not supported by the captions."
    )
    prompt_schema_version: str = "captionforge-pass-b-prompt-v0.6"
    response_schema_version: str = "captionforge-pass-b-claims-v0.5"
    preserve_raw_llm_response: bool = False

    # Ollama backend behavior.
    ollama_base_url: str = "http://127.0.0.1:11434"
    auto_pull_ollama_model: bool = False
    ollama_timeout_sec: int = 600
    ollama_keep_alive: str = "5m"
    # v0.6.5: keep the proven 2400-token common-path default, but make
    # it configurable for pathological reruns without editing source code.
    ollama_num_predict: int = 2400
    ollama_temperature: float = 0.0

    # v0.6 semantic profile behavior.
    # The Pass B output schema remains stable; the profile supplies deterministic
    # taxonomy constants, aliases, hint vocabulary, and conflict categories.
    semantic_profile_name: str = "image_v1_minimum"
    semantic_profile_version: str = "1.0.0"
    semantic_profile_json: str = ""


@dataclass
class BatchClaimConfig:
    input_jsonl: str
    output_jsonl: str = ""
    write_jsonl: bool = True
    dry_run: bool = False
    limit_images: int = 0
    include_error_pass_a_records: bool = False
    overwrite: bool = True

    # v0.6.4 resumable-batch behavior. Resume mode preserves prior ok
    # records in the output JSONL and reruns only missing, error, or explicitly
    # flagged image keys.
    resume: bool = False
    rerun_errors: bool = True
    rerun_image_keys: list[str] = field(default_factory=list)

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
    error_class: str = ""
    prompt_chars: int = 0
    raw_response_chars: int = 0
    elapsed_sec: float = 0.0
    normalized_claim_count: int = 0
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
    metrics: dict[str, Any] = field(default_factory=dict)


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
# Semantic profile loading and generic rule interpretation
# -------------------------------------------------------------------------

MINIMUM_SEMANTIC_PROFILE_NAME = "image_v1_minimum"
MINIMUM_SEMANTIC_PROFILE_VERSION = "1.0.0"

# v0.6.3 design note:
# The engine now keeps only a deliberately tiny built-in fallback profile.
# Domain semantics (female characters, landscapes, products, etc.) should live
# in installed or user-supplied semantic profile JSON files. The Pass B output
# schema is intentionally unchanged; profiles change taxonomy/vocabulary/rules,
# not the shape of emitted JSONL records.
MINIMUM_IMAGE_SEMANTIC_PROFILE: dict[str, Any] = {'profile_name': 'image_v1_minimum', 'profile_version': '1.0.0', 'description': 'Minimum viable built-in CaptionForge fallback profile. It exists only to keep the engine functional when an installed profile is missing or corrupt.', 'claim_types': ['visible_text', 'style_medium', 'composition', 'general'], 'claim_type_aliases': {'text': 'visible_text', 'style': 'style_medium', 'lighting': 'style_medium'}, 'mutually_exclusive_categories': [], 'claim_type_override_rules': [{'claim_type': 'visible_text', 'mode': 'visible_text_cue', 'text_scope': 'combined'}, {'claim_type': 'style_medium', 'mode': 'hints', 'profile_key': 'style_words', 'match': 'contains', 'text_scope': 'normalized_then_combined'}], 'normalization_rules': [{'claim_type': 'visible_text', 'mode': 'visible_text_cue'}, {'claim_type': 'style_medium', 'mode': 'hints', 'profile_key': 'style_words', 'match': 'contains'}], 'boilerplate_prefixes': ['the image depicts', 'the image shows', 'this image depicts', 'this image shows', 'a photo of', 'a photograph of', 'an image of', 'in this image'], 'style_words': ['photo', 'photograph', 'illustration', 'render', 'painting', 'drawing', 'style', 'lighting', 'studio lighting'], 'color_synonyms': {'grey': 'gray'}, 'singularize_map': {}}

# Backward-compatible alias. Older tests/imports may look for this symbol; it is
# no longer the runtime default and should not be treated as the female profile.
DEFAULT_SEMANTIC_PROFILE_NAME = MINIMUM_SEMANTIC_PROFILE_NAME
DEFAULT_SEMANTIC_PROFILE_VERSION = MINIMUM_SEMANTIC_PROFILE_VERSION
DEFAULT_FEMALE_CHARACTER_SEMANTIC_PROFILE = MINIMUM_IMAGE_SEMANTIC_PROFILE

ALLOWED_PROFILE_RULE_MODES = {"visible_text_cue", "hints", "contains", "color_noun"}


def deep_copy_jsonable(value: Any) -> Any:
    return json.loads(json.dumps(value, ensure_ascii=False))


def semantic_profile_sha1(profile: dict[str, Any]) -> str:
    return sha1_text(json.dumps(profile, ensure_ascii=False, sort_keys=True))


def validate_semantic_profile(profile: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(profile, dict):
        raise ValueError("Semantic profile must be a JSON object.")

    profile = deep_copy_jsonable(profile)

    required = [
        "profile_name",
        "profile_version",
        "claim_types",
        "claim_type_aliases",
        "mutually_exclusive_categories",
    ]
    missing = [key for key in required if key not in profile]
    if missing:
        raise ValueError(f"Semantic profile missing required keys: {missing}")

    if not isinstance(profile["claim_types"], list) or not profile["claim_types"]:
        raise ValueError("Semantic profile 'claim_types' must be a non-empty list.")
    if not all(isinstance(x, str) and x.strip() for x in profile["claim_types"]):
        raise ValueError("Semantic profile 'claim_types' must contain only non-empty strings.")

    canonical = {str(x).strip().lower() for x in profile["claim_types"]}
    profile["claim_types"] = sorted(canonical)

    if "general" not in canonical:
        profile["claim_types"].append("general")
        canonical.add("general")

    if not isinstance(profile["claim_type_aliases"], dict):
        raise ValueError("Semantic profile 'claim_type_aliases' must be an object.")

    aliases: dict[str, str] = {}
    for key, value in profile["claim_type_aliases"].items():
        k = str(key).strip().lower()
        v = str(value).strip().lower()
        if not k or not v:
            raise ValueError("Semantic profile claim_type_aliases cannot contain empty keys/values.")
        if v not in canonical:
            raise ValueError(f"Semantic profile alias {k!r} maps to unknown claim type {v!r}.")
        aliases[k] = v
    profile["claim_type_aliases"] = aliases

    conflicts = profile.get("mutually_exclusive_categories") or []
    if not isinstance(conflicts, list):
        raise ValueError("Semantic profile 'mutually_exclusive_categories' must be a list.")
    bad_conflicts = [str(x) for x in conflicts if str(x).strip().lower() not in canonical]
    if bad_conflicts:
        raise ValueError(f"Semantic profile conflict categories are not claim types: {bad_conflicts}")
    profile["mutually_exclusive_categories"] = [str(x).strip().lower() for x in conflicts]

    for rule_key in ("normalization_rules", "claim_type_override_rules"):
        rules = profile.get(rule_key, [])
        if rules is None:
            rules = []
        if not isinstance(rules, list):
            raise ValueError(f"Semantic profile '{rule_key}' must be a list.")
        clean_rules: list[dict[str, Any]] = []
        for i, rule in enumerate(rules):
            if not isinstance(rule, dict):
                raise ValueError(f"Semantic profile {rule_key}[{i}] must be an object.")
            mode = str(rule.get("mode") or "").strip().lower()
            claim_type = str(rule.get("claim_type") or "").strip().lower()
            if mode not in ALLOWED_PROFILE_RULE_MODES:
                raise ValueError(f"Semantic profile {rule_key}[{i}] has unsupported mode: {mode!r}.")
            if claim_type not in canonical:
                raise ValueError(f"Semantic profile {rule_key}[{i}] maps to unknown claim type: {claim_type!r}.")
            if mode in {"hints", "contains"}:
                profile_key = str(rule.get("profile_key") or "").strip()
                if not profile_key:
                    raise ValueError(f"Semantic profile {rule_key}[{i}] mode={mode!r} requires profile_key.")
                if profile_key not in profile:
                    raise ValueError(f"Semantic profile {rule_key}[{i}] references missing profile_key: {profile_key!r}.")
                if not isinstance(profile.get(profile_key), list):
                    raise ValueError(f"Semantic profile key {profile_key!r} must be a list for rule {rule_key}[{i}].")
            if mode == "color_noun":
                colors_key = str(rule.get("colors_key") or "").strip()
                noun_regex = str(rule.get("noun_regex") or "").strip()
                if not colors_key or colors_key not in profile:
                    raise ValueError(f"Semantic profile {rule_key}[{i}] mode='color_noun' requires valid colors_key.")
                if not isinstance(profile.get(colors_key), list) or not profile.get(colors_key):
                    raise ValueError(f"Semantic profile colors_key {colors_key!r} must be a non-empty list.")
                if not noun_regex:
                    raise ValueError(f"Semantic profile {rule_key}[{i}] mode='color_noun' requires noun_regex.")
                try:
                    re.compile(noun_regex)
                except re.error as exc:
                    raise ValueError(f"Semantic profile {rule_key}[{i}] invalid noun_regex: {exc}") from exc
            clean = dict(rule)
            clean["mode"] = mode
            clean["claim_type"] = claim_type
            clean_rules.append(clean)
        profile[rule_key] = clean_rules

    return profile


def warn_semantic_profile_fallback(reason: str) -> None:
    print(
        "[CaptionForge Claim Engine] WARNING: Semantic profile missing or corrupt; "
        f"defaulting to minimum viable profile '{MINIMUM_SEMANTIC_PROFILE_NAME}'. Reason: {reason}",
        flush=True,
    )


def load_semantic_profile(config: Optional[ClaimExtractionConfig] = None) -> dict[str, Any]:
    profile_path = str(getattr(config, "semantic_profile_json", "") or "").strip() if config else ""

    try:
        if profile_path:
            p = Path(profile_path)
            if p.is_dir():
                raise IsADirectoryError(f"semantic_profile_json points to a folder, not a JSON file: {p}")
            if not p.exists():
                raise FileNotFoundError(f"semantic_profile_json does not exist: {p}")
            profile = json.loads(p.read_text(encoding="utf-8"))
        else:
            raise FileNotFoundError("no semantic_profile_json was supplied")

        profile = validate_semantic_profile(profile)
        fallback_reason = ""
    except Exception as exc:
        fallback_reason = str(exc)
        warn_semantic_profile_fallback(fallback_reason)
        profile = validate_semantic_profile(MINIMUM_IMAGE_SEMANTIC_PROFILE)

    if config is not None:
        config.semantic_profile_name = str(profile.get("profile_name") or MINIMUM_SEMANTIC_PROFILE_NAME)
        config.semantic_profile_version = str(profile.get("profile_version") or MINIMUM_SEMANTIC_PROFILE_VERSION)
        setattr(config, "_captionforge_semantic_profile", profile)
        setattr(config, "_captionforge_semantic_profile_sha1", semantic_profile_sha1(profile))
        setattr(config, "_captionforge_semantic_profile_fallback_reason", fallback_reason)

    return profile


def get_semantic_profile(config: Optional[ClaimExtractionConfig] = None) -> dict[str, Any]:
    if config is not None:
        existing = getattr(config, "_captionforge_semantic_profile", None)
        if isinstance(existing, dict) and existing:
            return existing
    return load_semantic_profile(config)


def profile_list(config: Optional[ClaimExtractionConfig], key: str, fallback: Iterable[str] = ()) -> list[str]:
    profile = get_semantic_profile(config)
    value = profile.get(key, list(fallback))
    if isinstance(value, list):
        return [str(x).strip() for x in value if str(x).strip()]
    if isinstance(value, set):
        return [str(x).strip() for x in sorted(value) if str(x).strip()]
    return [str(value).strip()] if str(value).strip() else []


def profile_set(config: Optional[ClaimExtractionConfig], key: str, fallback: Iterable[str] = ()) -> set[str]:
    return {x.lower() for x in profile_list(config, key, fallback)}


def profile_mapping(config: Optional[ClaimExtractionConfig], key: str, fallback: Optional[dict[str, str]] = None) -> dict[str, str]:
    profile = get_semantic_profile(config)
    value = profile.get(key, fallback or {})
    if not isinstance(value, dict):
        return {}
    return {str(k).strip().lower(): str(v).strip().lower() for k, v in value.items() if str(k).strip()}


# Legacy import compatibility only. Runtime logic should read from the selected
# semantic profile through profile_list/profile_mapping/profile_set.
BOILERPLATE_PREFIXES = MINIMUM_IMAGE_SEMANTIC_PROFILE.get("boilerplate_prefixes", [])
STYLE_WORDS = set(MINIMUM_IMAGE_SEMANTIC_PROFILE.get("style_words", []))
COLOR_SYNONYMS = MINIMUM_IMAGE_SEMANTIC_PROFILE.get("color_synonyms", {})
HAIR_COLORS: list[str] = []
EYE_COLORS: list[str] = []
BACKGROUND_COLORS: list[str] = []
CLOTHING_HINTS: list[str] = []
POSE_HINTS: list[str] = []
SHOT_HINTS: list[str] = []
CANONICAL_CLAIM_TYPES = set(MINIMUM_IMAGE_SEMANTIC_PROFILE.get("claim_types", []))
CLAIM_TYPE_ALIASES = MINIMUM_IMAGE_SEMANTIC_PROFILE.get("claim_type_aliases", {})
MUTUALLY_EXCLUSIVE_CATEGORIES = set(MINIMUM_IMAGE_SEMANTIC_PROFILE.get("mutually_exclusive_categories", []))


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


def append_jsonl_record(path: str | Path, record: Any, dry_run: bool = False) -> None:
    """
    Append one JSONL record immediately and flush it.

    Used for long-running LLM batches so completed image records survive even if
    a later image stalls, times out, or the user aborts.
    """
    if dry_run:
        return

    p = Path(path)
    safe_mkdir(p.parent)

    with p.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record_to_json(record), ensure_ascii=False) + "\n")
        f.flush()            


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
            "local_source_index": idx,
            "source_record_index": src.source_record_index,
            "model_family": src.model_family,
            "model_name": src.model_name,
            "ensemble_run_index": src.ensemble_run_index,
            "caption": src.caption or src.raw_caption,
        }
        for idx, src in enumerate(source_refs)
    ]

    instructions = config.text_llm_prompt.strip() or ClaimExtractionConfig().text_llm_prompt
    claim_type_contract = " | ".join(profile_list(config, "claim_types", CANONICAL_CLAIM_TYPES))

    response_contract = {
        "claims": [
            {
                "claim_type": claim_type_contract,
                "evidence_text": "raw supporting phrase from one source caption; may contain typos",
                "original_claim": "clean grammatical atomic visual claim, e.g. 'the woman has blonde hair'",
                "normalized_claim": "short canonical comparison form with noun, e.g. 'blonde hair'",
                "specificity": "generic | specific | specific_normalized",
                "certainty": "visible | inferred_from_caption | uncertain",
                "source_record_indexes": [13]
            }
        ]
    }

    max_llm_claims = int(getattr(config, "max_llm_claims_per_image", 0) or 0)
    claim_limit_rule = ""
    if max_llm_claims > 0:
        claim_limit_rule = (
            f"13. Return at most {max_llm_claims} claims total.\n"
            "    Prefer the strongest, most visually useful claims. Do not exhaustively decompose every phrase.\n"
            "    If captions are repetitive or malformed, summarize only clear visible facts.\n"
        )

    prompt_text = (
        f"{instructions}\n\n"
        "You are extracting visual claims from caption text only. You do not see the image.\n"
        "Return strict JSON only, with no markdown fences and no explanatory prose.\n\n"

        "Important rules:\n"
        "1. Do not add visual facts that are not supported by the captions.\n"
        "2. Keep claims atomic: one visible attribute/action/style/background/text fact per claim.\n"
        "3. Use the exact source_record_index values provided in Source captions.\n"
        "4. Do not invent local source indexes such as 0, 1, 2 unless those are also the actual source_record_index values.\n"
        "5. You may correct grammar and obvious caption typos in original_claim.\n"
        "6. Keep evidence_text close to the raw caption phrase that supports the claim.\n"
        "7. Keep visible quoted text exactly as written unless the caption itself is clearly malformed.\n"
        "8. Do not merge unrelated facts into one claim.\n"
        "9. Prefer useful dataset-caption claims over tiny fragments such as 'to the right', 'textures', or 'has a slim'.\n"
        "10. normalized_claim must include the attribute noun when useful for comparison.\n"
        "    Good: 'blonde hair', 'green eye makeup', 'black crop top', 'pink electric guitar', 'wooden table', 'blue sky'.\n"
        "    Bad: 'blonde', 'green', 'black', 'guitar', 'table', 'sky'.\n"
        "11. For visible_text claims, normalized_claim should preserve the readable text and include context when appropriate.\n"
        "    Good: 'shirt text reading <quoted text>', 'poster text reading <quoted text>', 'sign text reading <quoted text>'.\n"
        "    Bad: 'text', 'words', 'letters'.\n"
        "12. For pose claims, normalized_claim should preserve the body relationship.\n"
        "    Good: 'hands behind head', 'arms behind back', 'elbows out', 'looking over shoulder'.\n"
        f"{claim_limit_rule}\n"

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
# Ollama backend
# -------------------------------------------------------------------------

def normalize_ollama_base_url(url: str) -> str:
    url = (url or "http://127.0.0.1:11434").strip()
    return url.rstrip("/")


def ollama_request_json(
    *,
    base_url: str,
    endpoint: str,
    payload: Optional[dict[str, Any]] = None,
    timeout_sec: int = 600,
) -> Any:
    """
    Minimal stdlib Ollama JSON helper.

    Uses urllib instead of requests so CaptionForge does not add another
    dependency to the ComfyUI environment.
    """
    base_url = normalize_ollama_base_url(base_url)
    url = f"{base_url}{endpoint}"

    data = None
    headers = {"Content-Type": "application/json"}

    if payload is not None:
        data = json.dumps(payload).encode("utf-8")

    req = urllib.request.Request(
        url=url,
        data=data,
        headers=headers,
        method="POST" if payload is not None else "GET",
    )

    try:
        with urllib.request.urlopen(req, timeout=int(timeout_sec)) as resp:
            text = resp.read().decode("utf-8", errors="replace")
    except urllib.error.URLError as exc:
        raise RuntimeError(
            "Could not reach Ollama.\n"
            f"Base URL: {base_url}\n"
            "Make sure Ollama is installed and running, then test with:\n"
            "  ollama list"
        ) from exc

    if not text.strip():
        return None

    try:
        return json.loads(text)
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"Ollama returned invalid JSON from {endpoint}: {exc}") from exc


def ollama_model_names(base_url: str, timeout_sec: int = 30) -> set[str]:
    obj = ollama_request_json(
        base_url=base_url,
        endpoint="/api/tags",
        payload=None,
        timeout_sec=timeout_sec,
    )

    names: set[str] = set()
    for item in obj.get("models", []) if isinstance(obj, dict) else []:
        name = str(item.get("name") or "").strip()
        if name:
            names.add(name)

    return names


def ollama_model_is_available(
    model_name: str,
    base_url: str,
    timeout_sec: int = 30,
) -> bool:
    model_name = (model_name or "").strip()
    if not model_name:
        return False

    names = ollama_model_names(base_url, timeout_sec=timeout_sec)

    if model_name in names:
        return True

    # Accept untagged family name only if Ollama has a tagged local match.
    # Example: user enters "llama3.1" and local model is "llama3.1:8b".
    if ":" not in model_name:
        prefix = f"{model_name}:"
        return any(name.startswith(prefix) for name in names)

    return False


def ensure_ollama_model_available(config: ClaimExtractionConfig) -> None:
    model_name = (config.text_llm_model or "").strip()
    if not model_name:
        raise ValueError(
            "text_llm_model is required when text_llm_backend='ollama'. "
            "Example: gpt-oss:20b, llama3.1:8b, qwen2.5:7b"
        )

    if ollama_model_is_available(
        model_name=model_name,
        base_url=config.ollama_base_url,
        timeout_sec=min(int(config.ollama_timeout_sec), 60),
    ):
        return

    raise RuntimeError(
        f"Ollama model is not installed: {model_name}\n\n"
        "Install it manually, then rerun CaptionForge:\n"
        f"  ollama pull {model_name}"
    )


def ollama_generate_claim_response(
    prompt_bundle: dict[str, Any],
    config: ClaimExtractionConfig,
) -> str:
    """
    Send one Pass B prompt bundle to Ollama and return raw response text.

    Ollama is asked for JSON mode, but the normal parser still validates the
    returned object. Bad or non-schema output is captured by existing parser
    audit/error handling.
    """
    ensure_ollama_model_available(config)

    model_name = (config.text_llm_model or "").strip()
    prompt = str(prompt_bundle.get("prompt") or "")

    payload: dict[str, Any] = {
        "model": model_name,
        "prompt": prompt,
        "stream": False,
        "format": "json",
        "options": {
            "temperature": float(config.ollama_temperature),
            "num_predict": int(config.ollama_num_predict),
        },
    }

    keep_alive = (config.ollama_keep_alive or "").strip()
    if keep_alive:
        payload["keep_alive"] = keep_alive

    print(
        f"[CaptionForge Claim Engine] Sending prompt to Ollama model: {model_name} "
        f"| num_predict={int(config.ollama_num_predict)} "
        f"| temperature={float(config.ollama_temperature):g}",
        flush=True,
    )
    t0 = time.perf_counter()

    obj = ollama_request_json(
        base_url=config.ollama_base_url,
        endpoint="/api/generate",
        payload=payload,
        timeout_sec=int(config.ollama_timeout_sec),
    )

    elapsed = time.perf_counter() - t0
    print(
        f"[CaptionForge Claim Engine] Ollama response received in {elapsed:.1f}s.",
        flush=True,
    )

    if not isinstance(obj, dict):
        raise RuntimeError("Ollama returned a non-object response.")

    if obj.get("error"):
        raise RuntimeError(f"Ollama generation failed: {obj.get('error')}")

    response_text = str(obj.get("response") or "").strip()
    if not response_text:
        raise RuntimeError("Ollama returned an empty response.")

    return response_text


# -------------------------------------------------------------------------
# Text splitting / normalization
# -------------------------------------------------------------------------

def normalize_whitespace(text: str) -> str:
    text = text.replace("\n", " ")
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"\s+([,.;:!?])", r"\1", text)
    return text.strip()


def strip_boilerplate(text: str, config: Optional[ClaimExtractionConfig] = None) -> str:
    s = normalize_whitespace(text)
    lowered = s.lower()
    for prefix in profile_list(config, "boilerplate_prefixes", BOILERPLATE_PREFIXES):
        if lowered.startswith(prefix):
            return s[len(prefix):].lstrip(" ,:-")
    return s


def split_caption_to_phrases(caption: str, config: Optional[ClaimExtractionConfig] = None) -> list[str]:
    caption = strip_boilerplate(caption, config=config)
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


def canonical_color(value: str, config: Optional[ClaimExtractionConfig] = None) -> str:
    v = value.lower().strip()
    v = re.sub(r"[-_]+", " ", v)
    v = re.sub(r"\s+", " ", v)
    return profile_mapping(config, "color_synonyms", COLOR_SYNONYMS).get(v, v)


def clean_claim_text(text: str, config: Optional[ClaimExtractionConfig] = None) -> str:
    text = strip_boilerplate(text, config=config)
    text = text.strip().strip(" .;:,")
    text = re.sub(r"^(?:and|with|while|a|an|the)\s+", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\b(a|an)\s+(a|an|the)\b", r"\2", text, flags=re.IGNORECASE)
    text = normalize_whitespace(text)
    return text



def singularize_light(text: str, config: Optional[ClaimExtractionConfig] = None) -> str:
    replacements = profile_mapping(config, "singularize_map", {})
    if not replacements:
        return text
    words = text.split()
    return " ".join(replacements.get(w, w) for w in words)


def color_noun_match(text: str, rule: dict[str, Any], config: Optional[ClaimExtractionConfig]) -> Optional[tuple[str, str]]:
    colors_key = str(rule.get("colors_key") or "").strip()
    noun_regex = str(rule.get("noun_regex") or "").strip()
    colors = profile_list(config, colors_key)
    if not colors or not noun_regex:
        return None

    color_pattern = "|".join(re.escape(c.lower()) for c in sorted(colors, key=len, reverse=True) if c)
    if not color_pattern:
        return None

    m = re.search(rf"\b({color_pattern})\s+({noun_regex})\b", text)
    if not m:
        return None
    return m.group(1), m.group(2)


def apply_profile_normalization_rule(
    rule: dict[str, Any],
    lower: str,
    raw: str,
    config: Optional[ClaimExtractionConfig],
) -> Optional[tuple[str, str, str]]:
    mode = str(rule.get("mode") or "").strip().lower()
    claim_type = str(rule.get("claim_type") or "general").strip().lower() or "general"

    if mode == "color_noun":
        match = color_noun_match(lower, rule, config)
        if not match:
            return None
        color_raw, _noun = match
        color = canonical_color(color_raw, config=config)
        template = str(rule.get("normalized_template") or "{color}").strip() or "{color}"
        normalized = template.format(color=color, color_raw=color_raw).strip().lower()
        specificity = "specific_normalized" if color != color_raw else "specific"
        return normalized, claim_type, specificity

    if mode == "visible_text_cue":
        if has_visible_text_cue(f"{lower} {raw.lower()}"):
            return lower, claim_type, "specific"
        return None

    if mode in {"hints", "contains"}:
        key = str(rule.get("profile_key") or "").strip()
        if not key:
            return None
        match_mode = str(rule.get("match") or mode).strip().lower()
        for hint in profile_list(config, key):
            h = hint.lower()
            if not h:
                continue
            if match_mode == "contains":
                matched = h in lower
            else:
                matched = bool(re.search(rf"\b{re.escape(h)}\b", lower))
            if matched:
                normalized = singularize_light(lower, config=config) if bool(rule.get("singularize")) else lower
                return normalized, claim_type, "specific"
        return None

    return None


def normalize_claim_text(text: str, config: ClaimExtractionConfig) -> tuple[str, str, str]:
    """
    Return (normalized_text, claim_type, specificity).

    v0.6.3 keeps detailed semantics in the selected semantic profile. The engine
    applies generic normalization rule modes in deterministic order.
    """
    raw = clean_claim_text(text, config=config)
    lower = raw.lower().replace("t shirt", "t-shirt")
    lower = re.sub(r"\s+", " ", lower).strip()

    for rule in get_semantic_profile(config).get("normalization_rules", []):
        if not isinstance(rule, dict):
            continue
        result = apply_profile_normalization_rule(rule, lower, raw, config)
        if result is not None:
            return result

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
    phrases = split_caption_to_phrases(caption, config=config)
    claims: list[AtomicClaim] = []
    seen_norms: set[str] = set()

    for phrase_index, phrase in enumerate(phrases):
        original = clean_claim_text(phrase, config=config)
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


def choose_llm_claim_text(item: dict[str, Any], config: ClaimExtractionConfig) -> tuple[str, str]:
    """
    Return (original_claim, evidence_text) from an LLM/manual claim item.

    Prefer a clean, atomic original_claim. If the LLM bloats original_claim with
    a whole sentence or caption fragment, use evidence_text when it is shorter
    and usable.
    """
    evidence_text = clean_claim_text(str(item.get("evidence_text") or ""), config=config)

    original_raw = clean_claim_text(str(item.get("original_claim") or ""), config=config)
    claim_raw = clean_claim_text(str(item.get("claim") or ""), config=config)
    text_raw = clean_claim_text(str(item.get("text") or ""), config=config)

    candidates = [original_raw, claim_raw, text_raw, evidence_text]

    cleaned: list[str] = []
    for value in candidates:
        text = clean_claim_text(value, config=config)
        if text and text not in cleaned:
            cleaned.append(text)

    if not cleaned:
        return "", evidence_text

    def is_usable(text: str) -> bool:
        return config.min_claim_chars <= len(text) <= config.max_claim_chars

    # If evidence_text is clean and much shorter than original_claim, prefer it.
    # This prevents storing whole caption clauses as original_claim.
    if evidence_text and is_usable(evidence_text):
        if not original_raw:
            return evidence_text, evidence_text

        original_too_long = len(original_raw) > config.max_claim_chars
        evidence_much_shorter = len(evidence_text) <= max(60, int(len(original_raw) * 0.55))
        original_sentence_like = (
            original_raw.count(".") >= 1
            or len(re.split(r"\s+(?:and|with|while|that|which)\s+", original_raw, flags=re.IGNORECASE)) >= 3
        )

        if original_too_long or evidence_much_shorter or original_sentence_like:
            return evidence_text, evidence_text

    for text in cleaned:
        if is_usable(text):
            return text, evidence_text

    return min(cleaned, key=len), evidence_text



def has_visible_text_cue(text: str) -> bool:
    return bool(
        '"' in text
        or "'" in text
        or re.search(r"\b(?:text|lettering|letters|words|word|logo|label|sign|caption)\b", text)
        or re.search(r"\b(?:reading|reads|says|spells|written)\b", text)
    )



def profile_rule_texts(rule: dict[str, Any], normalized: str, original: str, combined: str) -> list[str]:
    scope = str(rule.get("text_scope") or "combined").strip().lower()
    if scope == "normalized":
        return [normalized]
    if scope == "original":
        return [original]
    if scope == "normalized_then_combined":
        return [normalized, combined]
    if scope == "original_then_combined":
        return [original, combined]
    return [combined]


def profile_rule_matches(
    rule: dict[str, Any],
    normalized: str,
    original: str,
    combined: str,
    original_claim: str,
    config: Optional[ClaimExtractionConfig],
) -> bool:
    """
    Generic interpreter for deterministic semantic-profile claim routing rules.

    The code intentionally knows only rule modes and text scopes, not domain-
    specific taxonomy. Domain concepts such as makeup/accessory/landscape/
    architecture should live in installed or user-supplied semantic profiles.
    """
    mode = str(rule.get("mode") or "").strip().lower()
    texts = [t for t in profile_rule_texts(rule, normalized, original, combined) if t]

    if mode == "visible_text_cue":
        return any(has_visible_text_cue(f"{text} {original_claim}".lower()) for text in texts)

    if mode in {"hints", "contains"}:
        key = str(rule.get("profile_key") or "").strip()
        if not key:
            return False
        match_mode = str(rule.get("match") or mode).strip().lower()
        hints = [h.lower() for h in profile_list(config, key) if h]
        for text in texts:
            for hint in hints:
                if match_mode == "contains":
                    if hint in text:
                        return True
                elif re.search(rf"\b{re.escape(hint)}\b", text):
                    return True
        return False

    if mode == "color_noun":
        return any(color_noun_match(text, rule, config) is not None for text in texts)

    return False


def canonicalize_claim_type(
    value: str,
    normalized_claim: str = "",
    original_claim: str = "",
    config: Optional[ClaimExtractionConfig] = None,
) -> str:
    """
    Convert LLM claim_type output into one canonical CaptionForge category.

    v0.6.3 makes content-derived overrides profile-driven. The engine interprets
    generic rule modes from claim_type_override_rules, while the selected semantic
    profile supplies the domain-specific taxonomy and hint lists.
    """
    canonical_types = profile_set(config, "claim_types", CANONICAL_CLAIM_TYPES)
    aliases = profile_mapping(config, "claim_type_aliases", CLAIM_TYPE_ALIASES)

    raw = (value or "").strip().lower()
    normalized = (normalized_claim or "").lower()
    original = (original_claim or "").lower()
    combined = f"{normalized} {original}".lower()

    # Ordered profile-defined routing rules. Example for female_character_v1:
    # makeup hints intentionally run before eye_color, so "green eye makeup"
    # becomes makeup while "green eyes" remains eye_color.
    rules = get_semantic_profile(config).get("claim_type_override_rules") or []
    if isinstance(rules, list):
        for rule in rules:
            if not isinstance(rule, dict):
                continue
            target = str(rule.get("claim_type") or "").strip().lower()
            if not target or target not in canonical_types:
                continue
            if profile_rule_matches(rule, normalized, original, combined, original_claim, config):
                return target

    parts = [p.strip() for p in re.split(r"[|,/;]+", raw) if p.strip()]

    # Respect the model's category only after deterministic profile rules had a
    # chance to correct obvious domain cues.
    for part in parts:
        if part in canonical_types:
            return part

    for part in parts:
        alias_value = aliases.get(part)
        if alias_value:
            return alias_value

    return "general" if "general" in canonical_types else sorted(canonical_types)[0]


def is_negative_absence_claim(
    *,
    original_claim: str,
    normalized_claim: str,
    evidence_text: str = "",
) -> bool:
    """
    Reject absence-style claims such as:
      - "there is no visible text in the image"
      - "there is no makeup in the image"
      - "the image does not show a phone"
      - "no readable text is visible"
      - normalized_claim "<no quoted text>"

    Pass B currently extracts positive visible claims, not negative inventory
    claims about everything absent from the image.
    """
    orig = (original_claim or "").strip().lower()
    norm = (normalized_claim or "").strip().lower()
    evid = (evidence_text or "").strip().lower()
    combined = f"{orig} || {norm} || {evid}"

    if not combined.strip(" |"):
        return False

    exact_bad = {
        "<no quoted text>",
        "no quoted text",
        "none visible",
        "nothing visible",
        "not visible",
        "not shown",
        "not present",
        "not depicted",
    }

    if orig in exact_bad or norm in exact_bad or evid in exact_bad:
        return True

    absence_patterns = [
        r"\bthere (?:is|are) no\b",
        r"\bthere (?:is|are) not any\b",
        r"\bthere (?:is|are)n['’]?t any\b",
        r"\bno\b.{0,80}\b(?:visible|shown|present|depicted|seen|readable)\b",
        r"\bthe image does not (?:show|contain|include|depict)\b",
        r"\bthe image doesn['’]?t (?:show|contain|include|depict)\b",
        r"\bnot (?:visible|shown|present|depicted|seen|readable)\b",
        r"\bwithout any\b",
        r"\bwithout (?:visible|readable)\b",
        r"\bdoes not appear\b",
        r"\bdo not appear\b",
        r"<no quoted text>",
    ]

    return any(re.search(pattern, combined) for pattern in absence_patterns)


def claims_from_manual_response(
    image_key: str,
    response: dict[str, Any],
    source_refs: list[SourceCaptionRef],
    config: ClaimExtractionConfig,
    *,
    backend: str = "manual_json",
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

    source_by_record_index = {src.source_record_index: src for src in source_refs}
    source_by_local_index = {idx: src for idx, src in enumerate(source_refs)}
    default_source = source_refs[0] if source_refs else SourceCaptionRef(0, image_key, image_key, "", "", 0, "ok", "", "", "")

    claims: list[AtomicClaim] = []
    rejected: list[RejectedClaim] = []
    warnings: list[str] = []

    max_llm_claims = int(getattr(config, "max_llm_claims_per_image", 0) or 0)
    if max_llm_claims > 0 and len(claims_obj) > max_llm_claims:
        warnings.append(
            f"LLM returned {len(claims_obj)} claims; truncated to max_llm_claims_per_image={max_llm_claims}."
        )
        claims_obj = claims_obj[:max_llm_claims]

    for idx, item in enumerate(claims_obj):
        if not isinstance(item, dict):
            rejected.append(RejectedClaim(idx, "claim item is not an object", item))
            continue

        original, evidence_text = choose_llm_claim_text(item, config)

        if not original:
            rejected.append(RejectedClaim(idx, "missing original_claim/claim/text/evidence_text", item))
            continue

        if len(original) < config.min_claim_chars:
            rejected.append(RejectedClaim(idx, f"claim shorter than min_claim_chars={config.min_claim_chars}", item))
            continue
        if len(original) > config.max_claim_chars:
            rejected.append(
                RejectedClaim(
                    idx,
                    f"all claim text candidates longer than max_claim_chars={config.max_claim_chars}",
                    item,
                )
            )
            continue

        normalized_raw = str(item.get("normalized_claim") or item.get("normalized") or "").strip()
        claim_type_raw = str(item.get("claim_type") or item.get("category") or "").strip()
        specificity = str(item.get("specificity") or "specific").strip() or "specific"

        if normalized_raw:
            normalized = clean_claim_text(normalized_raw, config=config).lower()
            if not normalized:
                normalized, inferred_claim_type, norm_specificity = normalize_claim_text(original, config)
                claim_type = claim_type_raw or inferred_claim_type
                if not item.get("specificity"):
                    specificity = norm_specificity
            else:
                inferred_claim_type = normalize_claim_text(original, config)[1]
                claim_type = claim_type_raw or inferred_claim_type
        else:
            normalized, inferred_claim_type, norm_specificity = normalize_claim_text(original, config)
            claim_type = claim_type_raw or inferred_claim_type
            if not item.get("specificity"):
                specificity = norm_specificity

        claim_type = canonicalize_claim_type(claim_type, normalized, original, config=config)

        if is_negative_absence_claim(
            original_claim=original,
            normalized_claim=normalized,
            evidence_text=evidence_text,
        ):
            rejected.append(
                RejectedClaim(
                    idx,
                    "negative/absence claim rejected",
                    item,
                )
            )
            continue

        # Accept source_record_indexes, source_record_index, or no source.
        src_indexes = item.get("source_record_indexes")
        if src_indexes is None:
            src_indexes = item.get("source_record_index")
        if src_indexes is None:
            # v0.6.1: A single safe default source is not treated as a parser
            # warning. The generated AtomicClaim still records the concrete
            # source_record_index, so the output remains auditable without
            # noisy warnings for otherwise valid claims.
            src_indexes = [default_source.source_record_index]
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

            source = source_by_record_index.get(src_index)

            if source is None and src_index in source_by_local_index:
                source = source_by_local_index[src_index]
                warnings.append(
                    f"claims[{idx}] used local source index {src_index!r}; "
                    f"mapped to source_record_index={source.source_record_index}."
                )

            if source is None:
                warnings.append(
                    f"claims[{idx}] source index {src_index!r} not found in Pass A records; defaulted."
                )
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
        backend=backend,
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


def detect_conflicts(normalized_claims: list[NormalizedClaim], config: Optional[ClaimExtractionConfig] = None) -> list[ConflictRecord]:
    conflicts: list[ConflictRecord] = []

    by_category: dict[str, list[NormalizedClaim]] = defaultdict(list)
    for claim in normalized_claims:
        if claim.claim_type in profile_set(config, "mutually_exclusive_categories", MUTUALLY_EXCLUSIVE_CATEGORIES):
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



def classify_pass_b_error(exc: BaseException, raw_response: str = "") -> str:
    """
    Best-effort error classifier for resumable Pass B batches.

    This intentionally avoids raising new errors while classifying failures; the
    class is used for audit fields and retry selection, not control flow.
    """
    message = str(exc or "").lower()
    raw = str(raw_response or "")

    if isinstance(exc, TimeoutError) or "timed out" in message or "timeout" in message:
        return "timed_out"
    if "empty" in message and "response" in message:
        return "empty_response"
    if isinstance(exc, json.JSONDecodeError) or "json" in message:
        stripped = raw.strip()
        if stripped and not stripped.endswith(("}", "]")):
            return "truncated_or_invalid_json"
        if "unterminated" in message or "expecting value" in message or "expecting ',' delimiter" in message:
            return "truncated_or_invalid_json"
        return "invalid_json"
    if "could not reach ollama" in message:
        return "ollama_unreachable"
    if "ollama model is not installed" in message:
        return "ollama_model_missing"
    if "non-object response" in message:
        return "invalid_ollama_response"
    return "error"


def make_record_metrics(
    *,
    prompt_bundle: Optional[dict[str, Any]] = None,
    raw_response: str = "",
    elapsed_sec: float = 0.0,
    llm_parse_result: Optional[LLMParseResult] = None,
    atomic_claim_count: int = 0,
    normalized_claim_count: int = 0,
    rejected_claim_count: int = 0,
) -> dict[str, Any]:
    prompt_text = ""
    if isinstance(prompt_bundle, dict):
        prompt_text = str(prompt_bundle.get("prompt") or "")

    parsed_claim_count = 0
    parser_warning_count = 0
    error_class = ""
    if llm_parse_result is not None:
        parsed_claim_count = int(llm_parse_result.parsed_claim_count or 0)
        parser_warning_count = len(llm_parse_result.parser_warnings or [])
        error_class = str(llm_parse_result.error_class or "")

    return {
        "prompt_chars": len(prompt_text),
        "raw_response_chars": len(str(raw_response or "")),
        "elapsed_sec": round(float(elapsed_sec or 0.0), 3),
        "parsed_claim_count": parsed_claim_count,
        "atomic_claim_count": int(atomic_claim_count or 0),
        "normalized_claim_count": int(normalized_claim_count or 0),
        "rejected_claim_count": int(rejected_claim_count or 0),
        "parser_warning_count": parser_warning_count,
        "error_class": error_class,
    }


def build_image_claim_record(
    image_key: str,
    pass_a_records: list[dict[str, Any]],
    config: ClaimExtractionConfig,
    *,
    manual_response: Optional[dict[str, Any]] = None,
) -> ImageClaimRecord:
    record_t0 = time.perf_counter()
    source_refs = [make_source_ref(r) for r in pass_a_records]
    prompt_bundle = build_llm_prompt_bundle(image_key, source_refs, config)
    prompt_sha1 = str(prompt_bundle.get("prompt_sha1") or "")
    prompt_chars = len(str(prompt_bundle.get("prompt") or ""))

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
                prompt_chars=prompt_chars,
                error_class="missing_manual_response",
                error=f"No manual JSON response found for image_key={image_key!r}.",
            )
        else:
            all_claims, llm_parse_result, rejected_claims, raw_llm_response = claims_from_manual_response(
                image_key,
                manual_response,
                source_refs,
                config,
                backend="manual_json",
            )
            llm_parse_result.prompt_sha1 = prompt_sha1
            llm_parse_result.prompt_schema_version = config.prompt_schema_version

    elif config.text_llm_backend == "ollama":
        raw_llm_response = ollama_generate_claim_response(prompt_bundle, config)
        ollama_response = extract_json_object(raw_llm_response)

        all_claims, llm_parse_result, rejected_claims, raw_llm_response = claims_from_manual_response(
            image_key,
            {
                **ollama_response,
                "raw_llm_response": raw_llm_response,
            },
            source_refs,
            config,
            backend="ollama",
        )
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
    llm_parse_result.prompt_chars = prompt_chars
    llm_parse_result.raw_response_chars = len(raw_llm_response or "")
    llm_parse_result.elapsed_sec = round(time.perf_counter() - record_t0, 3)
    llm_parse_result.normalized_claim_count = len(normalized_claims)
    conflicts = detect_conflicts(normalized_claims, config=config) if config.detect_conflicts else []
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
            "max_llm_claims_per_image": int(config.max_llm_claims_per_image),
            "prompt_schema_version": config.prompt_schema_version,
            "response_schema_version": config.response_schema_version,
            "prompt_sha1": prompt_sha1,
            "semantic_profile_name": config.semantic_profile_name,
            "semantic_profile_version": config.semantic_profile_version,
            "semantic_profile_sha1": str(getattr(config, "_captionforge_semantic_profile_sha1", "")),
            "semantic_profile_fallback_reason": str(getattr(config, "_captionforge_semantic_profile_fallback_reason", "")),
            "config": record_to_json(config),
        },
        timestamp=iso_timestamp(),
        llm_parse_result=llm_parse_result,
        parser_warnings=list(llm_parse_result.parser_warnings),
        rejected_claims=rejected_claims,
        raw_llm_response=raw_llm_response if config.preserve_raw_llm_response else "",
        metrics=make_record_metrics(
            prompt_bundle=prompt_bundle,
            raw_response=raw_llm_response,
            elapsed_sec=llm_parse_result.elapsed_sec,
            llm_parse_result=llm_parse_result,
            atomic_claim_count=len(all_claims),
            normalized_claim_count=len(normalized_claims),
            rejected_claim_count=len(rejected_claims),
        ),
    )



def load_existing_pass_b_records(path: str | Path) -> dict[str, dict[str, Any]]:
    """Load existing Pass B output records keyed by image_key for resume mode."""
    p = Path(path)
    if not p.exists() or p.is_dir():
        return {}

    by_key: dict[str, dict[str, Any]] = {}
    with p.open("r", encoding="utf-8") as f:
        for line_number, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                print(
                    f"[CaptionForge Claim Engine] WARNING: ignoring invalid existing output line "
                    f"during resume: {p}:{line_number}",
                    flush=True,
                )
                continue
            if not isinstance(obj, dict):
                continue
            key = str(obj.get("image_key") or "").strip()
            if key:
                by_key[key] = obj
    return by_key


def existing_record_is_success(record: dict[str, Any]) -> bool:
    if not isinstance(record, dict):
        return False
    if str(record.get("status") or "").lower() != "ok":
        return False
    llm_parse = record.get("llm_parse_result") or {}
    if isinstance(llm_parse, dict):
        status = str(llm_parse.get("status") or "").lower()
        if status and status not in {"ok", "ok_with_rejected_claims"}:
            return False
    return True


def parse_rerun_image_keys(values: Iterable[str]) -> set[str]:
    keys: set[str] = set()
    for value in values or []:
        for part in str(value or "").split(","):
            part = part.strip()
            if part:
                keys.add(part)
    return keys


def extract_claims_batch(
    batch: BatchClaimConfig,
    config: Optional[ClaimExtractionConfig] = None,
) -> BatchClaimResult:
    config = config or ClaimExtractionConfig()
    config.text_llm_backend = (config.text_llm_backend or "heuristic").strip().lower()

    if config.text_llm_backend not in {"heuristic", "manual_json", "ollama"}:
        raise ValueError(
            f"Unsupported text_llm_backend={config.text_llm_backend!r}. "
            "v0.6 supports: heuristic, manual_json, ollama."
        )

    semantic_profile = load_semantic_profile(config)

    print(
        "[CaptionForge Claim Engine] Semantic profile: "
        f"{config.semantic_profile_name} v{config.semantic_profile_version} "
        f"sha1={semantic_profile_sha1(semantic_profile)}",
        flush=True,
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

    out: Optional[Path] = None
    if batch.write_jsonl:
        out = Path(batch.output_jsonl) if batch.output_jsonl.strip() else default_output_path(batch.input_jsonl)

        # Preserve old overwrite semantics, but switch actual batch writing to
        # per-record append so long LLM runs produce visible progress.
        resume_existing: dict[str, dict[str, Any]] = {}
        explicit_rerun_keys = parse_rerun_image_keys(batch.rerun_image_keys)

        if batch.resume and out.exists():
            resume_existing = load_existing_pass_b_records(out)
            kept_existing: list[dict[str, Any]] = []
            filtered_items: list[tuple[str, list[dict[str, Any]]]] = []

            for image_key, grouped_records in items:
                existing = resume_existing.get(image_key)
                force_rerun = image_key in explicit_rerun_keys
                if existing and existing_record_is_success(existing) and not force_rerun:
                    kept_existing.append(existing)
                    result.skipped += 1
                    continue
                if existing and not batch.rerun_errors and not force_rerun:
                    kept_existing.append(existing)
                    result.skipped += 1
                    continue
                filtered_items.append((image_key, grouped_records))

            items = filtered_items

            if not batch.dry_run:
                safe_mkdir(out.parent)
                with out.open("w", encoding="utf-8") as f:
                    for existing in kept_existing:
                        f.write(json.dumps(existing, ensure_ascii=False) + "\n")

            print(
                f"[CaptionForge Claim Engine] Resume mode: preserved={len(kept_existing)}, "
                f"queued_for_processing={len(items)}, output={out}",
                flush=True,
            )

        elif batch.overwrite and not batch.dry_run:
            safe_mkdir(out.parent)
            out.write_text("", encoding="utf-8")

    total_items = len(items)

    for item_index, (image_key, grouped_records) in enumerate(items, start=1):
        print(
            f"[CaptionForge Claim Engine] [{item_index}/{total_items}] Processing image_key: {image_key}",
            flush=True,
        )
        item_t0 = time.perf_counter()

        try:
            manual_response = manual_claims_for_image(image_key, manual_by_key) if config.text_llm_backend == "manual_json" else None
            record = build_image_claim_record(
                image_key,
                grouped_records,
                config,
                manual_response=manual_response,
            )
            result.records.append(record)

        except KeyboardInterrupt:
            print(
                f"[CaptionForge Claim Engine] Interrupted while processing image_key: {image_key}",
                flush=True,
            )
            raise

        except Exception as exc:
            result.failed += 1
            source_refs = [make_source_ref(r) for r in grouped_records]
            prompt_bundle = build_llm_prompt_bundle(image_key, source_refs, config)
            prompt_sha1 = str(prompt_bundle.get("prompt_sha1") or "")
            error_class = classify_pass_b_error(exc)
            error_elapsed = time.perf_counter() - item_t0
            llm_error = LLMParseResult(
                backend=config.text_llm_backend,
                status="error",
                prompt_sha1=prompt_sha1,
                prompt_schema_version=config.prompt_schema_version,
                response_schema_version=config.response_schema_version,
                parsed_claim_count=0,
                rejected_claim_count=0,
                parser_warnings=[],
                error_class=error_class,
                prompt_chars=len(str(prompt_bundle.get("prompt") or "")),
                raw_response_chars=0,
                elapsed_sec=round(error_elapsed, 3),
                normalized_claim_count=0,
                error=str(exc),
            )
            record = ImageClaimRecord(
                captionforge_pass="B",
                image_key=image_key,
                image=str(grouped_records[0].get("image") or image_key) if grouped_records else image_key,
                status="error",
                source_caption_count=len(grouped_records),
                source_caption_records=source_refs,
                atomic_claims=[],
                normalized_claims=[],
                conflicts=[],
                uncertainty_flags=[error_class, f"error: {exc}"],
                params={
                    "engine": config.engine_name,
                    "engine_version": config.engine_version,
                    "text_llm_backend": config.text_llm_backend,
                    "text_llm_model": config.text_llm_model,
                    "prompt_schema_version": config.prompt_schema_version,
                    "response_schema_version": config.response_schema_version,
                    "prompt_sha1": prompt_sha1,
                    "semantic_profile_name": config.semantic_profile_name,
                    "semantic_profile_version": config.semantic_profile_version,
                    "semantic_profile_sha1": str(getattr(config, "_captionforge_semantic_profile_sha1", "")),
                    "semantic_profile_fallback_reason": str(getattr(config, "_captionforge_semantic_profile_fallback_reason", "")),
                    "config": record_to_json(config),
                },
                timestamp=iso_timestamp(),
                llm_parse_result=llm_error,
                metrics=make_record_metrics(
                    prompt_bundle=prompt_bundle,
                    raw_response="",
                    elapsed_sec=error_elapsed,
                    llm_parse_result=llm_error,
                    atomic_claim_count=0,
                    normalized_claim_count=0,
                    rejected_claim_count=0,
                ),
            )
            result.records.append(record)

        item_elapsed = time.perf_counter() - item_t0
        print(
            f"[CaptionForge Claim Engine] [{item_index}/{total_items}] Finished image_key: {image_key} "
            f"in {item_elapsed:.1f}s | status={record.status}",
            flush=True,
        )

        if out is not None:
            append_jsonl_record(out, record, dry_run=batch.dry_run)
            print(
                f"[CaptionForge Claim Engine] [{item_index}/{total_items}] Wrote record to: {out}",
                flush=True,
            )

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
    parser.add_argument("--resume", action="store_true", help="Preserve existing successful Pass B output records and rerun only missing/error/flagged image keys.")
    parser.add_argument("--no-rerun-errors", action="store_true", help="In --resume mode, preserve existing error records instead of rerunning them.")
    parser.add_argument("--rerun-image-key", action="append", default=[], help="Force rerun for an image_key in --resume mode. May be repeated or comma-separated.")
    parser.add_argument("--include-error-pass-a-records", action="store_true", help="Include Pass A records with status != ok.")
    parser.add_argument("--no-conflicts", action="store_true", help="Disable simple conflict detection.")
    parser.add_argument("--no-low-confidence", action="store_true", help="Drop lower-confidence heuristic claims.")
    parser.add_argument("--max-llm-claims-per-image", type=int, default=40, help="Maximum claims requested from/parsing accepted for LLM backends per image. Default 40; use 0 to disable.")
    parser.add_argument("--text-llm-backend", default="heuristic", choices=["heuristic", "manual_json", "ollama"])
    parser.add_argument("--text-llm-model", default="", help="Text LLM model identifier. For Ollama, e.g. gpt-oss:20b or llama3.1:8b.")
    parser.add_argument("--ollama-base-url", default="http://127.0.0.1:11434", help="Ollama server URL.")
    parser.add_argument("--ollama-timeout-sec", type=int, default=600, help="Ollama request timeout in seconds.")
    parser.add_argument("--ollama-keep-alive", default="5m", help="Ollama keep_alive value, e.g. 5m, 30m, or 0.")
    parser.add_argument("--ollama-num-predict", type=int, default=2400, help="Ollama num_predict token cap. Default 2400; try 3000 for pathological reruns.")
    parser.add_argument("--ollama-temperature", type=float, default=0.0, help="Ollama sampling temperature. Default 0.0 for deterministic claim extraction.")
    parser.add_argument("--manual-json-path", default="", help="JSON/JSONL response file for manual_json backend.")
    parser.add_argument("--semantic-profile-json", default="", help="Optional deterministic semantic profile JSON. Defaults to female_character_v1.")
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
        max_llm_claims_per_image=max(0, int(args.max_llm_claims_per_image)),
        text_llm_backend=args.text_llm_backend,
        text_llm_model=args.text_llm_model,
        preserve_raw_llm_response=bool(args.preserve_raw_llm_response),
        ollama_base_url=args.ollama_base_url,
        ollama_timeout_sec=int(args.ollama_timeout_sec),
        ollama_keep_alive=args.ollama_keep_alive,
        ollama_num_predict=int(args.ollama_num_predict),
        ollama_temperature=float(args.ollama_temperature),
        semantic_profile_json=args.semantic_profile_json,
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
        resume=bool(args.resume),
        rerun_errors=not bool(args.no_rerun_errors),
        rerun_image_keys=list(args.rerun_image_key or []),
        manual_json_path=args.manual_json_path,
        write_llm_prompt_jsonl=bool(args.write_llm_prompt_jsonl),
        llm_prompt_jsonl=args.llm_prompt_jsonl,
        image_key_filter=args.image_key_filter,
    )

    result = extract_claims_batch(batch, config)

    print(
        f"[CaptionForge Claim Engine] processed={result.processed}, "
        f"skipped={result.skipped}, failed={result.failed}, output={output_jsonl}"
    )

    if args.dry_run:
        print(result.jsonl_text)

    return 0 if result.failed == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
