#!/usr/bin/env python
"""
CaptionForge VLM Validator Engine

- CaptionForge
  - This module is part of CaptionForge, a model-agnostic captioning and
    caption-refinement framework for ComfyUI developed by J. L. Córdova.

- Module Purpose
  - CLI-first image-aware validation/pruning/correction pass for the updated
    CaptionForge pipeline.
  - Consumes CaptionForge Distiller records from captionforge_distiller_engine.py
    Pass B_DISTILL.
  - Sends each distilled caption pair plus the source image to a VLM.
  - Emits clean, validated final-caption candidates without using semantic
    profiles, deterministic claim extraction, normalized claims, conflict
    tables, rescue rules, or truth-table reconstruction.

- Updated Pipeline Position
      Pass A raw caption ensemble JSONL
        -> Pass B_DISTILL text LLM distiller JSONL
        -> Pass C_VLM_VALIDATED image-aware validation/pruning/correction JSONL
        -> final CaptionForge node / final-caption export

- Design Notes
  - The distiller is intentionally recall-heavy and over-complete.
  - This engine is intentionally precision-oriented: it removes unsupported
    details, corrects visually wrong details, and keeps trigger/anchor fields
    separate and auditable.
  - Trigger words and user caption anchors are user-provided metadata/guidance,
    not VLM-inferred visible facts.
  - The trigger word is prepended exactly to both final captions when present.

- Attribution & License
  - Concept and implementation by J. L. Córdova with development assistance from
    ChatGPT (OpenAI).
  - Copyright (c) 2026 J. L. Córdova
  - Released under the MIT License.
"""

from __future__ import annotations

MANIFEST = {
    "name": "CaptionForge VLM Validator Engine",
    "version": (0, 2, 0),
    "author": "J. L. Córdova",
    "description": (
        "CLI-first CaptionForge VLM validation/pruning/correction engine. "
        "Consumes B_DISTILL pollster records, validates accepted and singleton "
        "claims against the source image, and writes rich validated final "
        "caption candidates without deterministic semantic profiles."
    ),
}

import argparse
import base64
import hashlib
import json
import mimetypes
import re
import time
import urllib.error
import urllib.request
from dataclasses import asdict, dataclass, field, is_dataclass
from datetime import datetime
from pathlib import Path, PureWindowsPath
from typing import Any, Iterable, Optional


# -----------------------------------------------------------------------------
# Dataclasses
# -----------------------------------------------------------------------------


@dataclass
class VLMValidatorConfig:
    engine_name: str = "CaptionForge VLM Validator Engine"
    engine_version: str = "0.2.0"

    vlm_backend: str = "ollama"  # ollama, manual_json, prompt_only
    vlm_model: str = ""

    # Internal fixed validator prompt. The node may still pass a widget prompt,
    # but CaptionForge's intended path is one strong grounding/copywriting contract.
    vlm_validator_prompt: str = (
        "You are CaptionForge Pass C: an image-grounded rich-caption validator and copywriter. "
        "You see the actual image and a Pass B pollster record. Pass B contains accepted claims, "
        "plausible singleton candidates, rejected or unresolved conflicts, and rich/taggy draft captions. "
        "Your job is not to summarize. Your job is to ground the evidence against the image, keep all "
        "supported useful details, correct visibly wrong details, visually confirm useful singletons when "
        "possible, add clearly visible missing details when they improve LoRA training value, and write a "
        "rich final caption. Use no deterministic semantic taxonomy. Judge natural-language claims only. "
        "Reject details only when they are visibly false, contradicted, not visible enough, or inappropriate "
        "as a training caption claim. Preserve accurate outfit construction, materials, jewelry, accessories, "
        "makeup, hair, eye details, body pose, hand placement, crop, lighting, background, and visible texture. "
        "Do not over-prune. Do not compress detailed jewelry, makeup, fabric, pose, or material evidence into "
        "generic phrases. Treat trigger words as metadata and do not prepend them yourself; the engine will "
        "prepend trigger and anchor after parsing. Treat the user caption anchor as training guidance: preserve "
        "it when compatible with the image and reject it only when clearly contradicted. The final narrative "
        "caption should be rich LoRA-training prose. The final comma caption should be a dense taggy caption "
        "with nearly the same visual content."
    )


    prompt_schema_version: str = "captionforge-vlm-validator-prompt-v0.2.0"
    response_schema_version: str = "captionforge-vlm-validator-response-v0.2.0"

    trigger_word: str = ""
    user_caption_anchor: str = ""

    preserve_raw_vlm_response: bool = False
    max_input_caption_chars: int = 2400
    max_output_caption_chars: int = 1800
    max_audit_note_chars: int = 260
    allow_missing_images: bool = False

    ollama_base_url: str = "http://127.0.0.1:11434"
    ollama_timeout_sec: int = 900
    ollama_keep_alive: str = "10m"
    ollama_num_predict: int = 2200
    ollama_temperature: float = 0.0
    ollama_top_p: float = 0.92
    ollama_top_k: int = 80
    ollama_seed: int = -1
    ollama_format_mode: str = "schema"  # json or schema


@dataclass
class BatchVLMValidatorConfig:
    input_jsonl: str
    image_root: str = ""
    output_jsonl: str = ""
    readable_sidecar_dir: str = ""
    write_jsonl: bool = True
    write_readable_sidecars: bool = True
    dry_run: bool = False
    limit_images: int = 0
    overwrite: bool = True
    resume: bool = False
    rerun_errors: bool = True
    rerun_image_keys: list[str] = field(default_factory=list)
    image_key_filter: str = ""

    manual_json_path: str = ""
    write_prompt_jsonl: bool = False
    prompt_jsonl: str = ""


@dataclass
class VLMValidatorParseResult:
    backend: str
    status: str
    prompt_sha1: str = ""
    prompt_schema_version: str = ""
    response_schema_version: str = ""
    raw_response_sha1: str = ""
    parser_warnings: list[str] = field(default_factory=list)
    error_class: str = ""
    error: str = ""
    prompt_chars: int = 0
    raw_response_chars: int = 0
    elapsed_sec: float = 0.0


@dataclass
class ImageVLMValidatedRecord:
    captionforge_pass: str
    image_key: str
    image: str
    image_resolved_path: str
    status: str

    validated_caption_narrative: str
    validated_caption_comma: str
    trigger_word: str
    user_caption_anchor: str

    removed_or_rejected_details: list[str]
    corrected_details: list[dict[str, Any]]
    uncertain_details: list[str]
    visual_validation_notes: list[str]
    validated_claims: list[str]
    added_visible_details: list[str]

    source_distiller: dict[str, Any]
    params: dict[str, Any]
    vlm_parse_result: VLMValidatorParseResult
    parser_warnings: list[str]
    raw_vlm_response: str
    metrics: dict[str, Any]
    timestamp: str


@dataclass
class BatchVLMValidatorResult:
    records: list[ImageVLMValidatedRecord] = field(default_factory=list)
    skipped: int = 0
    failed: int = 0

    @property
    def processed(self) -> int:
        return len([r for r in self.records if r.status in {"ok", "prompt_only"}])

    @property
    def jsonl_text(self) -> str:
        return "\n".join(json.dumps(record_to_json(r), ensure_ascii=False) for r in self.records)


class VLMResponseParseError(ValueError):
    def __init__(self, message: str, raw_response: str = "") -> None:
        super().__init__(message)
        self.raw_response = str(raw_response or "")


# -----------------------------------------------------------------------------
# Generic helpers
# -----------------------------------------------------------------------------


def iso_timestamp() -> str:
    return datetime.now().isoformat(timespec="seconds")


def json_safe(value: Any) -> Any:
    if is_dataclass(value):
        return {k: json_safe(v) for k, v in asdict(value).items()}
    if isinstance(value, set):
        return sorted(value)
    if isinstance(value, tuple):
        return [json_safe(v) for v in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, list):
        return [json_safe(v) for v in value]
    return value


def record_to_json(record: Any) -> dict[str, Any]:
    return json_safe(record)


def sha1_text(text: str) -> str:
    return hashlib.sha1(str(text or "").encode("utf-8")).hexdigest()


def sha1_file(path: str | Path, chunk_size: int = 1024 * 1024) -> str:
    h = hashlib.sha1()
    with Path(path).open("rb") as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def normalize_whitespace(text: Any) -> str:
    text = str(text or "").replace("\n", " ")
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def truncate_text(text: Any, max_chars: int) -> str:
    s = normalize_whitespace(text)
    if max_chars <= 0 or len(s) <= max_chars:
        return s
    cut = s[:max_chars].rstrip()
    boundary = max(cut.rfind(" "), cut.rfind("."), cut.rfind(","), cut.rfind(";"))
    if boundary >= max(80, int(max_chars * 0.65)):
        cut = cut[:boundary].rstrip()
    return cut.rstrip(" ,;:-") + "…"


def safe_mkdir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    p = Path(path)
    if p.is_dir():
        raise IsADirectoryError(f"JSONL path points to a folder, not a file: {p}")
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
            if not isinstance(obj, dict):
                raise ValueError(f"JSONL record at {p}:{line_number} is not an object.")
            records.append(obj)
    return records


def append_jsonl_record(path: str | Path, record: Any, dry_run: bool = False) -> None:
    if dry_run:
        return
    p = Path(path)
    safe_mkdir(p.parent)
    with p.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record_to_json(record), ensure_ascii=False) + "\n")
        f.flush()


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
    return p.with_name(f"{p.stem or 'captionforge'}_vlm_validated.jsonl")


def default_prompt_output_path(input_jsonl: str | Path) -> Path:
    p = Path(input_jsonl)
    return p.with_name(f"{p.stem or 'captionforge'}_vlm_validator_prompts.jsonl")


def default_sidecar_dir(output_jsonl: str | Path) -> Path:
    p = Path(output_jsonl)
    return p.with_name(f"{p.stem}_readable")


# -----------------------------------------------------------------------------
# Input field compatibility
# -----------------------------------------------------------------------------


def first_nonempty(record: dict[str, Any], keys: Iterable[str]) -> str:
    for key in keys:
        value = normalize_whitespace(record.get(key))
        if value:
            return value
    return ""


def get_distilled_narrative(record: dict[str, Any]) -> str:
    return normalize_whitespace(record.get("distilled_caption_narrative"))


def get_distilled_comma(record: dict[str, Any]) -> str:
    return normalize_whitespace(record.get("distilled_caption_comma"))


def get_rich_caption_draft(record: dict[str, Any]) -> str:
    return normalize_whitespace(
        record.get("rich_caption_draft")
        or record.get("copywriter_caption")
        or record.get("draft_caption")
        or record.get("distilled_caption_narrative")
        or ""
    )


def get_taggy_caption_draft(record: dict[str, Any]) -> str:
    return normalize_whitespace(
        record.get("taggy_caption_draft")
        or record.get("dense_caption_draft")
        or record.get("comma_caption_draft")
        or record.get("distilled_caption_comma")
        or ""
    )


def _claim_text(item: Any) -> str:
    if isinstance(item, dict):
        return normalize_whitespace(item.get("claim") or item.get("text") or item.get("value") or "")
    return normalize_whitespace(item)


def coerce_claim_records(value: Any, max_items: int = 120, max_chars: int = 320) -> list[dict[str, Any]]:
    if value is None:
        return []
    items = value if isinstance(value, list) else [value]
    out: list[dict[str, Any]] = []
    for item in items:
        if isinstance(item, dict):
            claim = truncate_text(item.get("claim") or item.get("text") or item.get("value") or "", max_chars)
            if not claim:
                continue
            out.append({
                "claim": claim,
                "support_count": item.get("support_count", 0),
                "supporting_model_families": item.get("supporting_model_families") or item.get("supporting_models") or [],
                "supporting_source_indices": item.get("supporting_source_indices") or item.get("source_indices") or [],
                "contradicted_by": item.get("contradicted_by") or item.get("contradictions") or [],
                "confidence": truncate_text(item.get("confidence") or "", max_chars),
                "reason": truncate_text(item.get("reason") or item.get("rationale") or "", max_chars),
            })
        else:
            claim = truncate_text(item, max_chars)
            if claim:
                out.append({"claim": claim})
        if len(out) >= max_items:
            break
    return out


def get_claim_bucket(record: dict[str, Any], key: str) -> list[dict[str, Any]]:
    return coerce_claim_records(record.get(key) or [])


def get_trigger_word(record: dict[str, Any], override: str = "") -> str:
    override = normalize_whitespace(override)
    if override:
        return override
    return first_nonempty(record, [
        "trigger_word",
        "caption_trigger",
        "lora_trigger",
        "trigger",
    ])


def get_user_caption_anchor(record: dict[str, Any], override: str = "") -> str:
    override = normalize_whitespace(override)
    if override:
        return override
    return first_nonempty(record, [
        "user_caption_anchor",
        "caption_anchor",
        "style_anchor",
        "caption_prefix",
    ])


def strip_leading_trigger(text: str, trigger_word: str) -> str:
    text = normalize_whitespace(text)
    trig = normalize_whitespace(trigger_word)
    if not text or not trig:
        return text
    pattern = r"^\s*" + re.escape(trig) + r"\s*[,;:\-–—]?\s*"
    return normalize_whitespace(re.sub(pattern, "", text, flags=re.IGNORECASE))


def prepend_trigger(text: str, trigger_word: str) -> str:
    text = strip_leading_trigger(text, trigger_word)
    trig = normalize_whitespace(trigger_word)
    if not trig:
        return text
    if not text:
        return trig
    return f"{trig}, {text}"


def _normalize_for_caption_match(text: Any) -> str:
    text = normalize_whitespace(text).lower()
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def _strip_leading_phrase(text: str, phrase: str) -> str:
    text = normalize_whitespace(text)
    phrase = normalize_whitespace(phrase)
    if not text or not phrase:
        return text
    pattern = r"^\s*" + re.escape(phrase) + r"\s*[,;:\-–—]?\s*"
    return normalize_whitespace(re.sub(pattern, "", text, flags=re.IGNORECASE))


def _anchor_already_present(text: str, anchor: str) -> bool:
    a = _normalize_for_caption_match(anchor)
    if not a:
        return True
    t = _normalize_for_caption_match(text)
    return a in t


def _flatten_audit_strings(value: Any) -> list[str]:
    out: list[str] = []
    if value is None:
        return out
    if isinstance(value, dict):
        for v in value.values():
            out.extend(_flatten_audit_strings(v))
        return out
    if isinstance(value, list):
        for item in value:
            out.extend(_flatten_audit_strings(item))
        return out
    text = normalize_whitespace(value)
    if text:
        out.append(text)
    return out


def _anchor_explicitly_rejected(anchor: str, response: dict[str, Any]) -> bool:
    """
    Return True only when the validator appears to have explicitly rejected or
    corrected the user caption anchor itself. The anchor is training intent, so
    absence from the VLM's final caption is not enough to suppress it.
    """
    anchor_norm = _normalize_for_caption_match(anchor)
    if not anchor_norm:
        return False

    removed = []
    for key in ("removed_or_rejected_details", "rejected_details", "removed_details"):
        removed.extend(_flatten_audit_strings(response.get(key)))
    for text in removed:
        if anchor_norm and anchor_norm in _normalize_for_caption_match(text):
            return True

    corrections = response.get("corrected_details") or response.get("corrections") or []
    if isinstance(corrections, list):
        for item in corrections:
            if isinstance(item, dict):
                src = item.get("from") or item.get("from_detail") or item.get("original") or ""
                if anchor_norm in _normalize_for_caption_match(src):
                    return True

    caution_words = {
        "contradict", "contradicted", "contradicts", "unsupported", "not visible",
        "not shown", "false", "wrong", "reject", "rejected", "remove", "removed",
    }
    notes = []
    for key in ("visual_validation_notes", "validation_notes", "uncertain_details"):
        notes.extend(_flatten_audit_strings(response.get(key)))
    for text in notes:
        text_norm = _normalize_for_caption_match(text)
        if anchor_norm in text_norm and any(word in text.lower() for word in caution_words):
            return True

    return False


def prepend_trigger_and_anchor(
    text: str,
    trigger_word: str,
    user_caption_anchor: str,
    response: dict[str, Any] | None = None,
) -> str:
    """
    Prefix final captions as:
        <trigger_word>, <user_caption_anchor>, <caption body>

    The trigger is always metadata. The anchor is user-supplied training intent:
    preserve it unless the VLM explicitly rejected/corrected that anchor. This
    keeps hard-to-see but user-important labels such as stylization, doll texture,
    unusual eye color, or illustration/photo intent available for LoRA training.
    """
    response = response or {}
    body = strip_leading_trigger(text, trigger_word)
    anchor = normalize_whitespace(user_caption_anchor)
    trig = normalize_whitespace(trigger_word)

    if anchor:
        body = _strip_leading_phrase(body, anchor)
        if _anchor_already_present(text, anchor):
            anchor = ""
        elif _anchor_explicitly_rejected(anchor, response):
            anchor = ""

    parts = [p for p in (trig, anchor, body) if normalize_whitespace(p)]
    return normalize_whitespace(", ".join(parts))


def normalize_comma_caption(text: str) -> str:
    text = normalize_whitespace(text)
    text = re.sub(r"\s*,\s*", ", ", text)
    text = re.sub(r"(,\s*){2,}", ", ", text)
    return text.strip(" ,")


def stable_filename_from_image_key(image_key: str) -> str:
    base = PureWindowsPath(str(image_key)).name if "\\" in str(image_key) else Path(str(image_key)).name
    base = base or "image"
    stem = Path(base).stem or "image"
    safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", stem).strip("._") or "image"
    digest = sha1_text(str(image_key))[:8]
    return f"{safe}__{digest}.txt"


# -----------------------------------------------------------------------------
# Image resolution
# -----------------------------------------------------------------------------


def pathish(value: str) -> str:
    return str(value or "").strip().strip('"')


def basename_cross_platform(value: str) -> str:
    text = pathish(value)
    if not text:
        return ""
    return PureWindowsPath(text).name if "\\" in text else Path(text).name


def candidate_image_names(record: dict[str, Any]) -> list[str]:
    names: list[str] = []
    for key in ("image", "image_key", "image_path", "source_image"):
        value = pathish(record.get(key, ""))
        if value and value not in names:
            names.append(value)
        base = basename_cross_platform(value)
        if base and base not in names:
            names.append(base)

    source = record.get("source_distiller") or record.get("source_pass_b") or {}
    if isinstance(source, dict):
        for key in ("image", "image_key", "image_path", "source_image"):
            value = pathish(source.get(key, ""))
            if value and value not in names:
                names.append(value)
            base = basename_cross_platform(value)
            if base and base not in names:
                names.append(base)

    source_records = record.get("source_caption_records") or []
    if isinstance(source_records, list):
        for src in source_records:
            if not isinstance(src, dict):
                continue
            value = pathish(src.get("image", "") or src.get("image_key", ""))
            if value and value not in names:
                names.append(value)
            base = basename_cross_platform(value)
            if base and base not in names:
                names.append(base)
    return names



_IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff"}
_IMAGE_ROOT_INDEX_CACHE: dict[str, dict[str, list[Path]]] = {}


def canonical_image_stem(value: Any) -> str:
    """
    Build a loose comparison key for image stems.

    CaptionForge caption records may use sanitized image_keys such as
    ``ChatGPT_Image_May_10__2026__07_06_01_PM__10`` while the source file on
    disk may be named ``ChatGPT Image May 10, 2026, 07_06_01 PM (10).png``.
    This canonical form removes punctuation, spaces, underscores, and case so
    those names can still be matched without forcing users to rename files.
    """
    text = pathish(value)
    if not text:
        return ""
    base = basename_cross_platform(text)
    stem = Path(base).stem or base
    return re.sub(r"[^a-z0-9]+", "", stem.lower())


def image_root_index(root: Path) -> dict[str, list[Path]]:
    """Index image files under a root by loose canonical stem, cached per root."""
    try:
        root_key = str(root.resolve())
    except Exception:
        root_key = str(root)
    cached = _IMAGE_ROOT_INDEX_CACHE.get(root_key)
    if cached is not None:
        return cached

    index: dict[str, list[Path]] = {}
    if not root.exists() or not root.is_dir():
        _IMAGE_ROOT_INDEX_CACHE[root_key] = index
        return index

    try:
        iterator = root.rglob("*")
        for p in iterator:
            if not p.is_file() or p.suffix.lower() not in _IMAGE_EXTENSIONS:
                continue
            key = canonical_image_stem(p.name)
            if key:
                index.setdefault(key, []).append(p)
    except Exception:
        pass

    for key in list(index.keys()):
        index[key] = sorted(index[key], key=lambda p: (len(str(p)), str(p).lower()))
    _IMAGE_ROOT_INDEX_CACHE[root_key] = index
    return index


def resolve_image_path(record: dict[str, Any], image_root: str = "") -> Optional[Path]:
    candidates = candidate_image_names(record)

    # 1. Direct absolute/relative paths embedded in records.
    for candidate in candidates:
        p = Path(candidate)
        if p.exists() and p.is_file():
            return p

    root_text = pathish(image_root)
    if not root_text:
        return None

    root = Path(root_text)
    if not root.exists() or not root.is_dir():
        return None

    # 2. Exact basename and relative-path matches under image_root.
    for candidate in candidates:
        candidate_text = pathish(candidate)
        if not candidate_text:
            continue

        # If a future caption record preserves a relative source path, honor it.
        rel = Path(candidate_text)
        if not rel.is_absolute() and any(sep in candidate_text for sep in ("/", "\\")):
            p = root / rel
            if p.exists() and p.is_file():
                return p

        base = basename_cross_platform(candidate_text)
        if not base:
            continue
        p = root / base
        if p.exists() and p.is_file():
            return p

    # 3. Exact basename search anywhere below image_root.
    basenames = {basename_cross_platform(c) for c in candidates if basename_cross_platform(c)}
    for base in basenames:
        matches = [p for p in root.rglob(base) if p.is_file()]
        if len(matches) == 1:
            return matches[0]
        if len(matches) > 1:
            return sorted(matches, key=lambda p: (len(str(p)), str(p).lower()))[0]

    # 4. Loose sanitized-stem match for user files with spaces, commas,
    # parentheses, and other punctuation. This lets a sanitized CaptionForge key
    # still resolve the original image file without imposing strict user naming.
    wanted = {canonical_image_stem(c) for c in candidates if canonical_image_stem(c)}
    if wanted:
        index = image_root_index(root)
        matches: list[Path] = []
        for key in wanted:
            matches.extend(index.get(key, []))
        if len(matches) == 1:
            return matches[0]
        if len(matches) > 1:
            return sorted(set(matches), key=lambda p: (len(str(p)), str(p).lower()))[0]

    return None


def image_to_base64(path: str | Path) -> str:
    return base64.b64encode(Path(path).read_bytes()).decode("ascii")


_IMAGE_BASE64_CACHE: dict[tuple[str, int, int], str] = {}


def image_to_base64_cached(path: str | Path) -> str:
    p = Path(path)
    stat = p.stat()
    key = (str(p.resolve()), int(stat.st_mtime_ns), int(stat.st_size))
    cached = _IMAGE_BASE64_CACHE.get(key)
    if cached is not None:
        return cached
    encoded = image_to_base64(p)
    _IMAGE_BASE64_CACHE[key] = encoded
    return encoded


def image_mime_type(path: str | Path) -> str:
    mime, _ = mimetypes.guess_type(str(path))
    return mime or "application/octet-stream"


# -----------------------------------------------------------------------------
# Prompt and response contract
# -----------------------------------------------------------------------------


def build_response_schema() -> dict[str, Any]:
    return {
        "type": "object",
        "properties": {
            "validated_caption_narrative": {"type": "string"},
            "validated_caption_comma": {"type": "string"},
            "validated_claims": {"type": "array", "items": {"type": "string"}},
            "added_visible_details": {"type": "array", "items": {"type": "string"}},
            "removed_or_rejected_details": {"type": "array", "items": {"type": "string"}},
            "corrected_details": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "from": {"type": "string"},
                        "to": {"type": "string"},
                        "reason": {"type": "string"},
                    },
                    "required": ["from", "to"],
                    "additionalProperties": True,
                },
            },
            "uncertain_details": {"type": "array", "items": {"type": "string"}},
            "visual_validation_notes": {"type": "array", "items": {"type": "string"}},
        },
        "required": [
            "validated_caption_narrative",
            "validated_caption_comma",
            "validated_claims",
            "added_visible_details",
            "removed_or_rejected_details",
            "corrected_details",
            "uncertain_details",
            "visual_validation_notes",
        ],
        "additionalProperties": True,
    }



def build_prompt_bundle(
    record: dict[str, Any],
    image_path: Optional[Path],
    config: VLMValidatorConfig,
) -> dict[str, Any]:
    image_key = str(record.get("image_key") or record.get("image") or "")
    trigger_word = get_trigger_word(record, config.trigger_word)
    anchor = get_user_caption_anchor(record, config.user_caption_anchor)

    rich_draft = truncate_text(
        strip_leading_trigger(get_rich_caption_draft(record), trigger_word),
        int(config.max_input_caption_chars),
    )
    taggy_draft = truncate_text(
        strip_leading_trigger(get_taggy_caption_draft(record), trigger_word),
        int(config.max_input_caption_chars),
    )

    accepted_claims = get_claim_bucket(record, "accepted_claims")
    singleton_candidates = get_claim_bucket(record, "singleton_candidates")
    rejected_or_unresolved_claims = get_claim_bucket(record, "rejected_or_unresolved_claims")
    claim_vote_summary = coerce_string_list(
        record.get("claim_vote_summary") or [],
        max_chars=int(config.max_audit_note_chars),
    )

    payload = {
        "image_key": image_key,
        "trigger_word": trigger_word,
        "user_caption_anchor": anchor,
        "important_metadata_policy": {
            "trigger_word": "Metadata/training token. Do not treat as visible evidence. Do not prepend in your response; the engine prepends it exactly.",
            "user_caption_anchor": (
                "User-provided training guidance, not pure visible evidence. Preserve it when compatible with the image; "
                "reject only if clearly contradicted."
            ),
        },
        "pass_b_pollster_record": {
            "accepted_claims": accepted_claims,
            "singleton_candidates": singleton_candidates,
            "rejected_or_unresolved_claims": rejected_or_unresolved_claims,
            "claim_vote_summary": claim_vote_summary,
            "rich_caption_draft": rich_draft,
            "taggy_caption_draft": taggy_draft,
        },
        "required_output": build_response_schema(),
    }

    prompt = (
        f"{normalize_whitespace(config.vlm_validator_prompt)}\n\n"
        "You are seeing one actual image. Validate the Pass B evidence against this image.\n"
        "Return strict JSON only. No markdown. No prose outside JSON.\n\n"

        "GROUNDING RULES:\n"
        "1. Keep all accepted claims that are visible or strongly implied by the image.\n"
        "2. Inspect singleton_candidates carefully. Promote them when visible; reject them when unsupported.\n"
        "3. Do not promote rejected_or_unresolved_claims unless the image clearly resolves the issue.\n"
        "4. Add clearly visible missing details when they improve LoRA-training value, especially face, eyes, makeup, hair, jewelry, accessories, clothing construction, materials, pose, hands, crop, background, lighting, and texture.\n"
        "5. Correct visibly wrong details instead of merely deleting them when a clear correction is visible.\n"
        "6. Do not summarize or compress rich supported details. A rich caption is the goal.\n"
        "7. Do not mention uncertainty, sources, captions, drafts, claims, ballots, or validation in final captions.\n"
        "8. Put uncertain/unsupported details only in audit fields, not in the final captions.\n"
        "9. Do not prepend the trigger word yourself. The engine will prepend trigger/anchor after parsing.\n\n"

        "FINAL CAPTION STYLE:\n"
        "- validated_caption_narrative: rich LoRA-training prose, one paragraph or a long fluent sentence.\n"
        "- validated_caption_comma: dense taggy comma-separated caption with nearly the same visual content.\n"
        "- validated_claims: list of supported claim strings you used.\n"
        "- added_visible_details: visible details you added from the image that were not explicit in Pass B.\n\n"

        "Input JSON:\n"
        f"{json.dumps(payload, ensure_ascii=False, indent=2)}\n"
    )

    image_info = {
        "resolved_path": str(image_path) if image_path else "",
        "exists": bool(image_path and image_path.exists()),
        "mime_type": image_mime_type(image_path) if image_path else "",
        "size_bytes": image_path.stat().st_size if image_path and image_path.exists() else 0,
        "sha1": sha1_file(image_path) if image_path and image_path.exists() else "",
    }

    return {
        "captionforge_pass": "VLM_validator_prompt",
        "prompt_schema_version": config.prompt_schema_version,
        "image_key": image_key,
        "image": str(record.get("image") or ""),
        "image_info": image_info,
        "prompt": prompt,
        "prompt_payload": payload,
        "prompt_sha1": sha1_text(prompt),
        "timestamp": iso_timestamp(),
    }



# -----------------------------------------------------------------------------
# JSON response parsing# -----------------------------------------------------------------------------
# JSON response parsing
# -----------------------------------------------------------------------------


def normalize_json_text(text: str) -> str:
    text = str(text or "").strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.IGNORECASE)
        text = re.sub(r"\s*```$", "", text)
    replacements = {
        "\u201c": '"',
        "\u201d": '"',
        "\u201e": '"',
        "\u201f": '"',
        "\u2018": "'",
        "\u2019": "'",
        "\u201a": "'",
        "\u201b": "'",
        "\u00a0": " ",
    }
    for src_char, dst_char in replacements.items():
        text = text.replace(src_char, dst_char)
    return text.strip()


def first_balanced_json_object_text(text: str) -> str:
    text = str(text or "")
    start = text.find("{")
    if start < 0:
        raise ValueError("Invalid VLM JSON response: no JSON object start found.")

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
                return text[start:idx + 1]

    return text[start:]


def extract_json_object(text: str) -> dict[str, Any]:
    text = normalize_json_text(text)
    if not text:
        raise ValueError("Empty VLM response.")

    last_error: Optional[Exception] = None
    for candidate in (text, first_balanced_json_object_text(text)):
        try:
            obj = json.loads(candidate)
            if not isinstance(obj, dict):
                raise ValueError("VLM response must be a JSON object.")
            return obj
        except json.JSONDecodeError as exc:
            last_error = exc

    raise ValueError(f"Invalid VLM JSON response: {last_error}") from last_error


def coerce_string_list(value: Any, max_chars: int = 260) -> list[str]:
    if value is None:
        return []
    if isinstance(value, list):
        out: list[str] = []
        for item in value:
            if isinstance(item, list):
                out.extend(coerce_string_list(item, max_chars=max_chars))
            else:
                text = truncate_text(item, max_chars)
                if text:
                    out.append(text)
        return out
    text = truncate_text(value, max_chars)
    return [text] if text else []


def coerce_corrections(value: Any, max_chars: int = 260) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        return []
    out: list[dict[str, Any]] = []
    for item in value:
        if isinstance(item, dict):
            src = truncate_text(item.get("from") or item.get("from_detail") or item.get("original") or "", max_chars)
            dst = truncate_text(item.get("to") or item.get("to_detail") or item.get("corrected") or "", max_chars)
            reason = truncate_text(item.get("reason") or item.get("evidence_note") or "", max_chars)
            if src or dst:
                out.append({"from": src, "to": dst, "reason": reason})
        else:
            text = truncate_text(item, max_chars)
            if text:
                out.append({"from": "", "to": text, "reason": ""})
    return out


def parse_vlm_response(
    response: dict[str, Any],
    *,
    backend: str,
    config: VLMValidatorConfig,
    prompt_bundle: dict[str, Any],
    raw_response: str,
    elapsed_sec: float,
) -> tuple[dict[str, Any], VLMValidatorParseResult, list[str]]:
    warnings: list[str] = []

    narrative = normalize_whitespace(response.get("validated_caption_narrative"))
    comma = normalize_comma_caption(response.get("validated_caption_comma"))

    if not narrative:
        warnings.append("validated_caption_narrative missing or empty.")
    if not comma:
        warnings.append("validated_caption_comma missing or empty.")

    narrative = truncate_text(narrative, int(config.max_output_caption_chars))
    comma = truncate_text(comma, int(config.max_output_caption_chars))

    payload = prompt_bundle.get("prompt_payload") or {}
    trigger_word = normalize_whitespace(payload.get("trigger_word") or get_trigger_word({}, config.trigger_word))
    user_caption_anchor = normalize_whitespace(payload.get("user_caption_anchor") or get_user_caption_anchor({}, config.user_caption_anchor))

    fields = {
        "validated_caption_narrative": prepend_trigger_and_anchor(narrative, trigger_word, user_caption_anchor, response),
        "validated_caption_comma": prepend_trigger_and_anchor(comma, trigger_word, user_caption_anchor, response),
        "validated_claims": coerce_string_list(
            response.get("validated_claims")
            or response.get("supported_claims")
            or response.get("kept_claims")
            or [],
            max_chars=int(config.max_audit_note_chars),
        ),
        "added_visible_details": coerce_string_list(
            response.get("added_visible_details")
            or response.get("added_details")
            or response.get("image_added_details")
            or [],
            max_chars=int(config.max_audit_note_chars),
        ),
        "removed_or_rejected_details": coerce_string_list(
            response.get("removed_or_rejected_details")
            or response.get("rejected_details")
            or response.get("removed_details")
            or [],
            max_chars=int(config.max_audit_note_chars),
        ),
        "corrected_details": coerce_corrections(
            response.get("corrected_details") or response.get("corrections") or [],
            max_chars=int(config.max_audit_note_chars),
        ),
        "uncertain_details": coerce_string_list(
            response.get("uncertain_details") or [],
            max_chars=int(config.max_audit_note_chars),
        ),
        "visual_validation_notes": coerce_string_list(
            response.get("visual_validation_notes")
            or response.get("validation_notes")
            or [],
            max_chars=int(config.max_audit_note_chars),
        ),
    }

    forbidden_phrases = [
        "in another image",
        "different image",
        "source caption",
        "draft caption",
        "caption says",
    ]
    for phrase in forbidden_phrases:
        if phrase in fields["validated_caption_narrative"].lower() or phrase in fields["validated_caption_comma"].lower():
            warnings.append(f"final caption contains discouraged phrase: {phrase!r}")

    parse = VLMValidatorParseResult(
        backend=backend,
        status="ok" if not warnings else "ok_with_parser_warnings",
        prompt_sha1=str(prompt_bundle.get("prompt_sha1") or ""),
        prompt_schema_version=config.prompt_schema_version,
        response_schema_version=config.response_schema_version,
        raw_response_sha1=sha1_text(raw_response) if raw_response else "",
        parser_warnings=warnings,
        prompt_chars=len(str(prompt_bundle.get("prompt") or "")),
        raw_response_chars=len(raw_response or ""),
        elapsed_sec=round(elapsed_sec, 3),
    )
    return fields, parse, warnings


# -----------------------------------------------------------------------------
# Manual response handling
# -----------------------------------------------------------------------------


def normalize_manual_response(value: Any) -> dict[str, Any]:
    if isinstance(value, str):
        return extract_json_object(value)
    if isinstance(value, dict):
        if "response" in value and isinstance(value.get("response"), (str, dict)):
            return normalize_manual_response(value["response"])
        return value
    raise ValueError("Manual VLM response must be a JSON object or JSON string.")


def load_manual_responses(path: str | Path) -> dict[str, Any]:
    if not str(path or "").strip():
        return {}
    p = Path(path)
    if p.is_dir():
        raise IsADirectoryError(f"manual_json_path points to a folder, not a file: {p}")
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
                    raise ValueError(f"Manual JSONL record at {p}:{line_number} is not an object.")
                key = str(obj.get("image_key") or "").strip()
                if not key:
                    key = "__single__" if "__single__" not in by_key else f"__single__:{line_number}"
                by_key[key] = obj
        return by_key

    obj = json.loads(p.read_text(encoding="utf-8"))
    if isinstance(obj, dict):
        if "image_key" in obj or "validated_caption_narrative" in obj or "response" in obj:
            by_key[str(obj.get("image_key") or "__single__")] = obj
        else:
            for key, value in obj.items():
                by_key[str(key)] = value
        return by_key
    if isinstance(obj, list):
        for idx, item in enumerate(obj):
            if not isinstance(item, dict):
                raise ValueError(f"Manual JSON list item {idx} is not an object.")
            by_key[str(item.get("image_key") or f"__single__:{idx}")] = item
        return by_key
    raise ValueError("Manual JSON must contain an object/list or JSONL object records.")


def manual_response_for_image(image_key: str, manual_by_key: dict[str, Any]) -> Optional[dict[str, Any]]:
    if not manual_by_key:
        return None
    if image_key in manual_by_key:
        return normalize_manual_response(manual_by_key[image_key])
    if "__single__" in manual_by_key and len(manual_by_key) == 1:
        return normalize_manual_response(manual_by_key["__single__"])
    return None


# -----------------------------------------------------------------------------
# Ollama backend
# -----------------------------------------------------------------------------


def normalize_ollama_base_url(url: str) -> str:
    return (url or "http://127.0.0.1:11434").strip().rstrip("/")


def ollama_request_json(*, base_url: str, endpoint: str, payload: Optional[dict[str, Any]] = None, timeout_sec: int = 900) -> Any:
    base_url = normalize_ollama_base_url(base_url)
    url = f"{base_url}{endpoint}"
    data = None
    headers = {"Content-Type": "application/json"}
    method = "GET"
    if payload is not None:
        data = json.dumps(payload).encode("utf-8")
        method = "POST"
    req = urllib.request.Request(url=url, data=data, headers=headers, method=method)
    try:
        with urllib.request.urlopen(req, timeout=int(timeout_sec)) as resp:
            text = resp.read().decode("utf-8", errors="replace")
    except urllib.error.URLError as exc:
        raise RuntimeError("Could not reach Ollama. Make sure Ollama is running and test with: ollama list") from exc
    if not text.strip():
        return None
    try:
        return json.loads(text)
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"Ollama returned invalid JSON from {endpoint}: {exc}") from exc


def ollama_model_names(base_url: str, timeout_sec: int = 30) -> set[str]:
    obj = ollama_request_json(base_url=base_url, endpoint="/api/tags", payload=None, timeout_sec=timeout_sec)
    names: set[str] = set()
    if isinstance(obj, dict):
        for item in obj.get("models", []) or []:
            if isinstance(item, dict) and item.get("name"):
                names.add(str(item["name"]))
    return names


def ollama_model_is_available(model_name: str, base_url: str, timeout_sec: int = 30) -> bool:
    model_name = str(model_name or "").strip()
    if not model_name:
        return False
    names = ollama_model_names(base_url, timeout_sec=timeout_sec)
    if model_name in names:
        return True
    if ":" not in model_name:
        return any(n.startswith(model_name + ":") for n in names)
    return False


def ensure_ollama_model_available(config: VLMValidatorConfig) -> None:
    model_name = str(config.vlm_model or "").strip()
    if not model_name:
        raise ValueError("--vlm-model is required when --vlm-backend ollama is used.")
    if ollama_model_is_available(model_name, config.ollama_base_url, timeout_sec=min(int(config.ollama_timeout_sec), 60)):
        return
    raise RuntimeError(f"Ollama model is not installed: {model_name}\nInstall it with: ollama pull {model_name}")


def resolve_ollama_format(config: VLMValidatorConfig) -> str | dict[str, Any]:
    mode = str(config.ollama_format_mode or "json").strip().lower()
    if mode == "schema":
        return build_response_schema()
    return "json"


def ollama_generate_vlm_response(
    prompt_bundle: dict[str, Any],
    image_path: Path,
    config: VLMValidatorConfig,
) -> str:
    ensure_ollama_model_available(config)
    if not image_path.exists():
        raise FileNotFoundError(f"Image does not exist for Ollama VLM call: {image_path}")

    payload: dict[str, Any] = {
        "model": str(config.vlm_model or "").strip(),
        "prompt": str(prompt_bundle.get("prompt") or ""),
        "images": [image_to_base64_cached(image_path)],
        "stream": False,
        "format": resolve_ollama_format(config),
        "options": {
            "temperature": float(config.ollama_temperature),
            "num_predict": int(config.ollama_num_predict),
            "top_p": float(config.ollama_top_p),
            "top_k": int(config.ollama_top_k),
        },
    }
    if int(getattr(config, "ollama_seed", -1)) >= 0:
        payload["options"]["seed"] = int(config.ollama_seed)
    if str(config.ollama_keep_alive or "").strip():
        payload["keep_alive"] = str(config.ollama_keep_alive).strip()

    print(
        f"[CaptionForge VLM Validator] Sending image to Ollama model: {config.vlm_model} "
        f"| num_predict={int(config.ollama_num_predict)} "
        f"| temperature={float(config.ollama_temperature):g} "
        f"| top_p={float(config.ollama_top_p):g} "
        f"| top_k={int(config.ollama_top_k)} "
        f"| seed={int(getattr(config, 'ollama_seed', -1))} "
        f"| format_mode={config.ollama_format_mode} "
        f"| image_cache=on",
        flush=True,
    )
    obj = ollama_request_json(
        base_url=config.ollama_base_url,
        endpoint="/api/generate",
        payload=payload,
        timeout_sec=int(config.ollama_timeout_sec),
    )
    if not isinstance(obj, dict):
        raise RuntimeError("Ollama returned a non-object response.")
    if obj.get("error"):
        raise RuntimeError(f"Ollama generation failed: {obj.get('error')}")
    response_text = str(obj.get("response") or "").strip()
    if not response_text:
        raise RuntimeError("Ollama returned an empty response.")
    return response_text


# -----------------------------------------------------------------------------
# Record construction
# -----------------------------------------------------------------------------


def make_prompt_only_fields(record: dict[str, Any], config: VLMValidatorConfig) -> dict[str, Any]:
    trigger = get_trigger_word(record, config.trigger_word)
    anchor = get_user_caption_anchor(record, config.user_caption_anchor)
    narrative_seed = get_rich_caption_draft(record) or get_distilled_narrative(record)
    comma_seed = get_taggy_caption_draft(record) or get_distilled_comma(record)
    narrative = prepend_trigger_and_anchor(strip_leading_trigger(narrative_seed, trigger), trigger, anchor, {})
    comma = prepend_trigger_and_anchor(normalize_comma_caption(strip_leading_trigger(comma_seed, trigger)), trigger, anchor, {})
    claim_seed = [_claim_text(c) for c in get_claim_bucket(record, "accepted_claims") if _claim_text(c)]
    return {
        "validated_caption_narrative": narrative,
        "validated_caption_comma": comma,
        "validated_claims": claim_seed,
        "added_visible_details": [],
        "removed_or_rejected_details": [],
        "corrected_details": [],
        "uncertain_details": ["prompt_only backend: no VLM validation performed"],
        "visual_validation_notes": ["Pass B draft captions passed through unchanged except trigger/anchor normalization."],
    }



def source_distiller_metadata(record: dict[str, Any], image_sha1: str = "") -> dict[str, Any]:
    params = record.get("params") if isinstance(record.get("params"), dict) else {}
    metrics = record.get("metrics") if isinstance(record.get("metrics"), dict) else {}
    parse = record.get("distill_parse_result") or record.get("llm_parse_result") or record.get("distiller_parse_result") or {}
    if not isinstance(parse, dict):
        parse = {}
    raw_response = str(record.get("raw_distill_response") or record.get("raw_llm_response") or record.get("raw_distiller_response") or "")

    accepted = get_claim_bucket(record, "accepted_claims")
    singletons = get_claim_bucket(record, "singleton_candidates")
    rejected = get_claim_bucket(record, "rejected_or_unresolved_claims")

    return {
        "captionforge_pass": record.get("captionforge_pass", ""),
        "image_key": record.get("image_key", ""),
        "image": record.get("image", ""),
        "status": record.get("status", ""),
        "engine": record.get("engine") or params.get("engine", ""),
        "engine_version": record.get("engine_version") or params.get("engine_version", ""),
        "llm_backend": params.get("llm_backend", params.get("text_llm_backend", "")),
        "llm_model": params.get("llm_model", params.get("text_llm_model", "")),
        "prompt_sha1": parse.get("prompt_sha1", params.get("prompt_sha1", "")),
        "trigger_word": get_trigger_word(record),
        "user_caption_anchor": get_user_caption_anchor(record),
        "accepted_claim_count": len(accepted),
        "singleton_candidate_count": len(singletons),
        "rejected_or_unresolved_claim_count": len(rejected),
        "rich_caption_draft_sha1": sha1_text(get_rich_caption_draft(record)),
        "taggy_caption_draft_sha1": sha1_text(get_taggy_caption_draft(record)),
        "parse_result": parse,
        "metrics": metrics,
        "raw_response_sha1": sha1_text(raw_response) if raw_response else "",
        "image_sha1": image_sha1,
    }



def make_metrics(prompt_bundle: dict[str, Any], raw_response: str, parse: VLMValidatorParseResult, image_path: Optional[Path]) -> dict[str, Any]:
    return {
        "prompt_chars": int(getattr(parse, "prompt_chars", 0) or len(str(prompt_bundle.get("prompt") or ""))),
        "raw_response_chars": len(raw_response or ""),
        "elapsed_sec": parse.elapsed_sec,
        "parser_warning_count": len(parse.parser_warnings or []),
        "image_exists": bool(image_path and image_path.exists()),
        "image_size_bytes": image_path.stat().st_size if image_path and image_path.exists() else 0,
    }


def build_validated_record(
    record: dict[str, Any],
    *,
    image_path: Optional[Path],
    config: VLMValidatorConfig,
    manual_response: Optional[dict[str, Any]] = None,
) -> ImageVLMValidatedRecord:
    image_key = str(record.get("image_key") or record.get("image") or "")
    prompt_bundle = build_prompt_bundle(record, image_path, config)
    raw_response = ""
    t0 = time.perf_counter()

    trigger = get_trigger_word(record, config.trigger_word)
    anchor = get_user_caption_anchor(record, config.user_caption_anchor)

    if config.vlm_backend == "prompt_only":
        fields = make_prompt_only_fields(record, config)
        elapsed = time.perf_counter() - t0
        parse = VLMValidatorParseResult(
            backend="prompt_only",
            status="prompt_only",
            prompt_sha1=str(prompt_bundle.get("prompt_sha1") or ""),
            prompt_schema_version=config.prompt_schema_version,
            response_schema_version=config.response_schema_version,
            prompt_chars=len(str(prompt_bundle.get("prompt") or "")),
            elapsed_sec=round(elapsed, 3),
        )
        parser_warnings: list[str] = []
        status = "prompt_only"

    elif config.vlm_backend == "manual_json":
        if manual_response is None:
            fields = make_prompt_only_fields(record, config)
            elapsed = time.perf_counter() - t0
            parse = VLMValidatorParseResult(
                backend="manual_json",
                status="missing_manual_response",
                prompt_sha1=str(prompt_bundle.get("prompt_sha1") or ""),
                prompt_schema_version=config.prompt_schema_version,
                response_schema_version=config.response_schema_version,
                error_class="missing_manual_response",
                error=f"No manual VLM response found for image_key={image_key!r}.",
                prompt_chars=len(str(prompt_bundle.get("prompt") or "")),
                elapsed_sec=round(elapsed, 3),
            )
            parser_warnings = [parse.error]
            status = "error"
        else:
            raw_response = json.dumps(manual_response, ensure_ascii=False, sort_keys=True)
            fields, parse, parser_warnings = parse_vlm_response(
                manual_response,
                backend="manual_json",
                config=config,
                prompt_bundle=prompt_bundle,
                raw_response=raw_response,
                elapsed_sec=time.perf_counter() - t0,
            )
            status = "ok"

    elif config.vlm_backend == "ollama":
        if image_path is None or not image_path.exists():
            raise FileNotFoundError(f"Could not resolve image for image_key={image_key!r}")
        raw_response = ollama_generate_vlm_response(prompt_bundle, image_path, config)
        try:
            response = extract_json_object(raw_response)
        except Exception as exc:
            raise VLMResponseParseError(str(exc), raw_response=raw_response) from exc
        fields, parse, parser_warnings = parse_vlm_response(
            response,
            backend="ollama",
            config=config,
            prompt_bundle=prompt_bundle,
            raw_response=raw_response,
            elapsed_sec=time.perf_counter() - t0,
        )
        status = "ok"
    else:
        raise ValueError(f"Unsupported vlm_backend={config.vlm_backend!r}. Use ollama, manual_json, or prompt_only.")

    image_sha1 = sha1_file(image_path) if image_path and image_path.exists() else ""

    return ImageVLMValidatedRecord(
        captionforge_pass="C_VLM_VALIDATED",
        image_key=image_key,
        image=str(record.get("image") or image_key),
        image_resolved_path=str(image_path) if image_path else "",
        status=status,
        validated_caption_narrative=fields["validated_caption_narrative"],
        validated_caption_comma=fields["validated_caption_comma"],
        trigger_word=trigger,
        user_caption_anchor=anchor,
        removed_or_rejected_details=fields["removed_or_rejected_details"],
        corrected_details=fields["corrected_details"],
        uncertain_details=fields["uncertain_details"],
        visual_validation_notes=fields["visual_validation_notes"],
        validated_claims=fields.get("validated_claims", []),
        added_visible_details=fields.get("added_visible_details", []),
        source_distiller=source_distiller_metadata(record, image_sha1=image_sha1),
        params={
            "engine": config.engine_name,
            "engine_version": config.engine_version,
            "vlm_backend": config.vlm_backend,
            "vlm_model": config.vlm_model,
            "prompt_schema_version": config.prompt_schema_version,
            "response_schema_version": config.response_schema_version,
            "prompt_sha1": str(prompt_bundle.get("prompt_sha1") or ""),
            "trigger_word": trigger,
            "user_caption_anchor": anchor,
            "config": record_to_json(config),
        },
        vlm_parse_result=parse,
        parser_warnings=parser_warnings,
        raw_vlm_response=raw_response if config.preserve_raw_vlm_response else "",
        metrics=make_metrics(prompt_bundle, raw_response, parse, image_path),
        timestamp=iso_timestamp(),
    )


def classify_vlm_error(exc: BaseException, raw_response: str = "") -> str:
    message = str(exc or "").lower()
    raw = str(raw_response or "")
    if isinstance(exc, TimeoutError) or "timed out" in message or "timeout" in message:
        return "timed_out"
    if isinstance(exc, FileNotFoundError) or "could not resolve image" in message or "image does not exist" in message:
        return "image_missing"
    if "ollama model is not installed" in message:
        return "ollama_model_missing"
    if "could not reach ollama" in message:
        return "ollama_unreachable"
    if "empty" in message and "response" in message:
        return "empty_response"
    if isinstance(exc, VLMResponseParseError):
        if raw.strip() and not raw.strip().endswith(("}", "]")):
            return "truncated_or_invalid_json"
        return "invalid_json"
    if isinstance(exc, json.JSONDecodeError) or "json" in message:
        if raw.strip() and not raw.strip().endswith(("}", "]")):
            return "truncated_or_invalid_json"
        return "invalid_json"
    return "error"


def make_error_record(
    record: dict[str, Any],
    *,
    image_path: Optional[Path],
    config: VLMValidatorConfig,
    exc: BaseException,
) -> ImageVLMValidatedRecord:
    image_key = str(record.get("image_key") or record.get("image") or "")
    prompt_bundle = build_prompt_bundle(record, image_path, config)
    raw_response = str(getattr(exc, "raw_response", "") or "")
    error_class = classify_vlm_error(exc, raw_response=raw_response)
    trigger = get_trigger_word(record, config.trigger_word)
    anchor = get_user_caption_anchor(record, config.user_caption_anchor)
    fields = make_prompt_only_fields(record, config)
    parse = VLMValidatorParseResult(
        backend=config.vlm_backend,
        status="error",
        prompt_sha1=str(prompt_bundle.get("prompt_sha1") or ""),
        prompt_schema_version=config.prompt_schema_version,
        response_schema_version=config.response_schema_version,
        raw_response_sha1=sha1_text(raw_response) if raw_response else "",
        error_class=error_class,
        error=str(exc),
        prompt_chars=len(str(prompt_bundle.get("prompt") or "")),
        raw_response_chars=len(raw_response),
        elapsed_sec=0.0,
    )
    image_sha1 = sha1_file(image_path) if image_path and image_path.exists() else ""
    return ImageVLMValidatedRecord(
        captionforge_pass="C_VLM_VALIDATED",
        image_key=image_key,
        image=str(record.get("image") or image_key),
        image_resolved_path=str(image_path) if image_path else "",
        status="error",
        validated_caption_narrative="",
        validated_caption_comma="",
        trigger_word=trigger,
        user_caption_anchor=anchor,
        removed_or_rejected_details=[],
        corrected_details=[],
        uncertain_details=fields["uncertain_details"],
        visual_validation_notes=[f"No VLM validation decision was made because the backend returned an error: {error_class}."],
        validated_claims=[],
        added_visible_details=[],
        source_distiller=source_distiller_metadata(record, image_sha1=image_sha1),
        params={
            "engine": config.engine_name,
            "engine_version": config.engine_version,
            "vlm_backend": config.vlm_backend,
            "vlm_model": config.vlm_model,
            "prompt_schema_version": config.prompt_schema_version,
            "response_schema_version": config.response_schema_version,
            "prompt_sha1": str(prompt_bundle.get("prompt_sha1") or ""),
            "trigger_word": trigger,
            "user_caption_anchor": anchor,
            "config": record_to_json(config),
        },
        vlm_parse_result=parse,
        parser_warnings=[str(exc)],
        raw_vlm_response=raw_response if config.preserve_raw_vlm_response else "",
        metrics=make_metrics(prompt_bundle, raw_response, parse, image_path),
        timestamp=iso_timestamp(),
    )


# -----------------------------------------------------------------------------
# Readable sidecars
# -----------------------------------------------------------------------------


def write_readable_sidecar(record: ImageVLMValidatedRecord, sidecar_dir: str | Path, dry_run: bool = False) -> Path:
    sidecar_path = Path(sidecar_dir) / stable_filename_from_image_key(record.image_key)
    if dry_run:
        return sidecar_path
    safe_mkdir(sidecar_path.parent)
    lines = [
        f"CaptionForge VLM Validated Caption",
        f"image_key: {record.image_key}",
        f"image: {record.image}",
        f"status: {record.status}",
        f"trigger_word: {record.trigger_word}",
        f"user_caption_anchor: {record.user_caption_anchor}",
        "",
        "[validated_caption_narrative]",
        record.validated_caption_narrative,
        "",
        "[validated_caption_comma]",
        record.validated_caption_comma,
        "",
        "[removed_or_rejected_details]",
    ]
    lines.extend(f"- {x}" for x in record.removed_or_rejected_details)
    lines.extend(["", "[validated_claims]"])
    lines.extend(f"- {x}" for x in record.validated_claims)
    lines.extend(["", "[added_visible_details]"])
    lines.extend(f"- {x}" for x in record.added_visible_details)
    lines.extend(["", "[corrected_details]"])
    lines.extend(f"- {x.get('from','')} -> {x.get('to','')} ({x.get('reason','')})" for x in record.corrected_details)
    lines.extend(["", "[uncertain_details]"])
    lines.extend(f"- {x}" for x in record.uncertain_details)
    lines.extend(["", "[visual_validation_notes]"])
    lines.extend(f"- {x}" for x in record.visual_validation_notes)
    sidecar_path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")
    return sidecar_path


# -----------------------------------------------------------------------------
# Batch orchestration
# -----------------------------------------------------------------------------


def existing_record_is_success(record: dict[str, Any]) -> bool:
    if not isinstance(record, dict):
        return False
    if str(record.get("status") or "").lower() not in {"ok", "prompt_only"}:
        return False
    parse = record.get("vlm_parse_result") or {}
    if isinstance(parse, dict):
        status = str(parse.get("status") or "").lower()
        if status and status not in {"ok", "ok_with_parser_warnings", "prompt_only"}:
            return False
    return True


def load_existing_records(path: str | Path) -> dict[str, dict[str, Any]]:
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
                print(f"[CaptionForge VLM Validator] WARNING: ignoring invalid resume line {p}:{line_number}", flush=True)
                continue
            if isinstance(obj, dict) and obj.get("image_key"):
                by_key[str(obj["image_key"])] = obj
    return by_key


def parse_rerun_image_keys(values: Iterable[str]) -> set[str]:
    keys: set[str] = set()
    for value in values or []:
        for part in str(value or "").split(","):
            part = part.strip()
            if part:
                keys.add(part)
    return keys


def filter_input_records(records: list[dict[str, Any]], image_key_filter: str = "") -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for record in records:
        # CaptionForge v0.2 Pass B records are pollster/copywriter records.
        # Legacy narrative/comma records are still accepted when present.
        has_pass_b_payload = bool(
            get_rich_caption_draft(record)
            or get_taggy_caption_draft(record)
            or get_claim_bucket(record, "accepted_claims")
            or get_claim_bucket(record, "singleton_candidates")
            or get_distilled_narrative(record)
            or get_distilled_comma(record)
        )
        if not has_pass_b_payload:
            continue
        status = str(record.get("status") or "ok").lower()
        if status != "ok":
            continue
        image_key = str(record.get("image_key") or record.get("image") or "")
        if image_key_filter:
            needle = image_key_filter.lower()
            if image_key != image_key_filter and needle not in image_key.lower():
                continue
        out.append(record)
    return out


def extract_validate_batch(batch: BatchVLMValidatorConfig, config: Optional[VLMValidatorConfig] = None) -> BatchVLMValidatorResult:
    config = config or VLMValidatorConfig()
    config.vlm_backend = str(config.vlm_backend or "ollama").strip().lower()
    if config.vlm_backend not in {"ollama", "manual_json", "prompt_only"}:
        raise ValueError("Unsupported VLM backend. Use: ollama, manual_json, prompt_only.")

    input_records = read_jsonl(batch.input_jsonl)
    items = filter_input_records(input_records, batch.image_key_filter)
    if batch.limit_images and batch.limit_images > 0:
        items = items[: int(batch.limit_images)]

    manual_by_key = load_manual_responses(batch.manual_json_path) if config.vlm_backend == "manual_json" else {}

    prompt_bundles: list[dict[str, Any]] = []
    for record in items:
        image_path = resolve_image_path(record, batch.image_root)
        prompt_bundles.append(build_prompt_bundle(record, image_path, config))
    if batch.write_prompt_jsonl:
        prompt_path = Path(batch.prompt_jsonl) if str(batch.prompt_jsonl or "").strip() else default_prompt_output_path(batch.input_jsonl)
        write_jsonl(prompt_path, prompt_bundles, overwrite=True, dry_run=batch.dry_run)
        print(f"[CaptionForge VLM Validator] Wrote prompt bundles: {prompt_path}", flush=True)

    result = BatchVLMValidatorResult()
    out: Optional[Path] = None
    sidecar_dir: Optional[Path] = None

    if batch.write_jsonl:
        out = Path(batch.output_jsonl) if str(batch.output_jsonl or "").strip() else default_output_path(batch.input_jsonl)
        explicit_rerun_keys = parse_rerun_image_keys(batch.rerun_image_keys)
        if batch.resume and out.exists():
            existing = load_existing_records(out)
            kept_existing: list[dict[str, Any]] = []
            filtered: list[dict[str, Any]] = []
            for record in items:
                key = str(record.get("image_key") or record.get("image") or "")
                old = existing.get(key)
                force = key in explicit_rerun_keys
                if old and existing_record_is_success(old) and not force:
                    kept_existing.append(old)
                    result.skipped += 1
                elif old and not batch.rerun_errors and not force:
                    kept_existing.append(old)
                    result.skipped += 1
                else:
                    filtered.append(record)
            items = filtered
            if not batch.dry_run:
                safe_mkdir(out.parent)
                with out.open("w", encoding="utf-8") as f:
                    for old in kept_existing:
                        f.write(json.dumps(old, ensure_ascii=False) + "\n")
            print(
                f"[CaptionForge VLM Validator] Resume mode: preserved={len(kept_existing)}, queued={len(items)}, output={out}",
                flush=True,
            )
        elif batch.overwrite and not batch.dry_run:
            safe_mkdir(out.parent)
            out.write_text("", encoding="utf-8")

        if batch.write_readable_sidecars:
            sidecar_dir = Path(batch.readable_sidecar_dir) if str(batch.readable_sidecar_dir or "").strip() else default_sidecar_dir(out)

    total = len(items)
    for idx, record in enumerate(items, start=1):
        image_key = str(record.get("image_key") or record.get("image") or "")
        print(f"[CaptionForge VLM Validator] [{idx}/{total}] Processing image_key: {image_key}", flush=True)
        t0 = time.perf_counter()
        image_path = resolve_image_path(record, batch.image_root)

        try:
            if image_path is None and config.vlm_backend == "ollama" and not config.allow_missing_images:
                raise FileNotFoundError(
                    f"Could not resolve image for image_key={image_key!r}. "
                    "Use --image-root to point at the original image directory."
                )
            manual_response = manual_response_for_image(image_key, manual_by_key) if config.vlm_backend == "manual_json" else None
            validated = build_validated_record(
                record,
                image_path=image_path,
                config=config,
                manual_response=manual_response,
            )
            if validated.status == "error":
                result.failed += 1
            result.records.append(validated)
        except KeyboardInterrupt:
            print(f"[CaptionForge VLM Validator] Interrupted while processing image_key: {image_key}", flush=True)
            raise
        except Exception as exc:
            result.failed += 1
            validated = make_error_record(record, image_path=image_path, config=config, exc=exc)
            result.records.append(validated)

        elapsed = time.perf_counter() - t0
        print(
            f"[CaptionForge VLM Validator] [{idx}/{total}] Finished image_key: {image_key} "
            f"in {elapsed:.1f}s | status={validated.status}",
            flush=True,
        )
        if out is not None:
            append_jsonl_record(out, validated, dry_run=batch.dry_run)
            print(f"[CaptionForge VLM Validator] [{idx}/{total}] Wrote record to: {out}", flush=True)
        if sidecar_dir is not None:
            sidecar_path = write_readable_sidecar(validated, sidecar_dir, dry_run=batch.dry_run)
            print(f"[CaptionForge VLM Validator] [{idx}/{total}] Wrote readable sidecar to: {sidecar_path}", flush=True)

    return result


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="CaptionForge VLM validation/pruning/correction engine for B_DISTILL JSONL."
    )
    parser.add_argument("--input-jsonl", required=True, help="Path to B_DISTILL distiller JSONL.")
    parser.add_argument("--image-root", default="", help="Original image directory/root used to resolve image paths.")
    parser.add_argument("--output-jsonl", default="", help="Path to VLM validated JSONL. Defaults beside input.")
    parser.add_argument("--readable-sidecar-dir", default="", help="Directory for readable .txt sidecars. Defaults beside output.")
    parser.add_argument("--no-readable-sidecars", action="store_true", help="Disable readable sidecar writing.")

    parser.add_argument("--trigger-word", default="", help="Override trigger word. If omitted, inherited from distiller records.")
    parser.add_argument("--user-caption-anchor", default="", help="Override user caption anchor. If omitted, inherited from distiller records.")
    parser.add_argument("--vlm-validator-prompt", default="", help="Override the user-editable validator/richness prompt.")

    parser.add_argument("--vlm-backend", default="ollama", choices=["ollama", "manual_json", "prompt_only"], help="VLM backend.")
    parser.add_argument("--vlm-model", default="", help="VLM model name. For Ollama, e.g. llama3.2-vision:11b.")
    parser.add_argument("--limit-images", type=int, default=0, help="Process only first N distiller records. 0 = no limit.")
    parser.add_argument("--image-key-filter", default="", help="Process only image_keys that exactly match or contain this string.")
    parser.add_argument("--dry-run", action="store_true", help="Run without writing output JSONL; prints JSONL result.")
    parser.add_argument("--append", action="store_true", help="Append output JSONL instead of overwriting.")
    parser.add_argument("--resume", action="store_true", help="Preserve existing successful records and rerun only missing/error/flagged keys.")
    parser.add_argument("--no-rerun-errors", action="store_true", help="In --resume mode, preserve existing error records instead of rerunning them.")
    parser.add_argument("--rerun-image-key", action="append", default=[], help="Force rerun for one image_key in --resume mode. May repeat or comma-separate.")
    parser.add_argument("--allow-missing-images", action="store_true", help="Do not fail immediately when images cannot be resolved. Useful with prompt_only/manual_json.")

    parser.add_argument("--max-input-caption-chars", type=int, default=2400, help="Max chars per input distiller caption in prompt.")
    parser.add_argument("--max-output-caption-chars", type=int, default=1800, help="Max chars retained per validated caption.")
    parser.add_argument("--max-audit-note-chars", type=int, default=260, help="Max chars retained per audit note/correction.")
    parser.add_argument("--preserve-raw-vlm-response", action="store_true", help="Store raw VLM response in output JSONL.")

    parser.add_argument("--manual-json-path", default="", help="JSON/JSONL file containing manual VLM responses for manual_json backend.")
    parser.add_argument("--write-prompt-jsonl", action="store_true", help="Write prompt bundles for review/debugging.")
    parser.add_argument("--prompt-jsonl", default="", help="Prompt bundle JSONL path. Defaults beside input.")

    parser.add_argument("--ollama-base-url", default="http://127.0.0.1:11434", help="Ollama server URL.")
    parser.add_argument("--ollama-timeout-sec", type=int, default=900, help="Ollama request timeout in seconds.")
    parser.add_argument("--ollama-keep-alive", default="10m", help="Ollama keep_alive value.")
    parser.add_argument("--ollama-num-predict", type=int, default=2200, help="Ollama num_predict cap. Default tuned slightly richer than the first smoke test.")
    parser.add_argument("--ollama-temperature", type=float, default=0.0, help="Ollama sampling temperature.")
    parser.add_argument("--ollama-top-p", type=float, default=0.92, help="Ollama top_p sampling cap.")
    parser.add_argument("--ollama-top-k", type=int, default=80, help="Ollama top_k sampling cap. Higher than the first validator test for modestly richer captions.")
    parser.add_argument("--ollama-seed", type=int, default=-1, help="Optional Ollama seed. -1 omits seed.")
    parser.add_argument("--ollama-format-mode", default="schema", choices=["json", "schema"], help="Ollama response format mode.")
    return parser


def main(argv: Optional[list[str]] = None) -> int:
    args = build_arg_parser().parse_args(argv)

    config = VLMValidatorConfig(
        vlm_backend=args.vlm_backend,
        vlm_model=args.vlm_model,
        vlm_validator_prompt=args.vlm_validator_prompt or VLMValidatorConfig().vlm_validator_prompt,
        trigger_word=args.trigger_word,
        user_caption_anchor=args.user_caption_anchor,
        preserve_raw_vlm_response=bool(args.preserve_raw_vlm_response),
        max_input_caption_chars=max(0, int(args.max_input_caption_chars)),
        max_output_caption_chars=max(0, int(args.max_output_caption_chars)),
        max_audit_note_chars=max(0, int(args.max_audit_note_chars)),
        allow_missing_images=bool(args.allow_missing_images),
        ollama_base_url=args.ollama_base_url,
        ollama_timeout_sec=int(args.ollama_timeout_sec),
        ollama_keep_alive=args.ollama_keep_alive,
        ollama_num_predict=int(args.ollama_num_predict),
        ollama_temperature=float(args.ollama_temperature),
        ollama_top_p=float(args.ollama_top_p),
        ollama_top_k=int(args.ollama_top_k),
        ollama_seed=int(args.ollama_seed),
        ollama_format_mode=args.ollama_format_mode,
    )

    output_jsonl = args.output_jsonl or str(default_output_path(args.input_jsonl))
    batch = BatchVLMValidatorConfig(
        input_jsonl=args.input_jsonl,
        image_root=args.image_root,
        output_jsonl=output_jsonl,
        readable_sidecar_dir=args.readable_sidecar_dir,
        write_jsonl=True,
        write_readable_sidecars=not bool(args.no_readable_sidecars),
        dry_run=bool(args.dry_run),
        limit_images=int(args.limit_images),
        overwrite=not bool(args.append),
        resume=bool(args.resume),
        rerun_errors=not bool(args.no_rerun_errors),
        rerun_image_keys=list(args.rerun_image_key or []),
        image_key_filter=args.image_key_filter,
        manual_json_path=args.manual_json_path,
        write_prompt_jsonl=bool(args.write_prompt_jsonl),
        prompt_jsonl=args.prompt_jsonl,
    )

    result = extract_validate_batch(batch, config)
    print(
        f"[CaptionForge VLM Validator] processed={result.processed}, skipped={result.skipped}, "
        f"failed={result.failed}, output={output_jsonl}",
        flush=True,
    )
    if args.dry_run:
        print(result.jsonl_text)
    return 0 if result.failed == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())



# =============================================================================
# CaptionForge Experimental Reversed Pipeline: Pass B VLM Cleaner
# =============================================================================

CLEANER_PASS = "B_VLM_CLEANED"


def _cf_cleaner_get_raw_caption(record: dict[str, Any]) -> str:
    return normalize_whitespace(record.get("caption") or record.get("raw_caption") or "")


def _cf_cleaner_pass_a_usable(record: dict[str, Any]) -> bool:
    pass_name = str(record.get("captionforge_pass") or "").upper()
    if pass_name not in {"", "A", "A_RAW_CAPTIONS"}:
        return False
    status = str(record.get("status") or "ok").lower()
    if status != "ok":
        return False
    return bool(_cf_cleaner_get_raw_caption(record))


def _cf_cleaner_record_key(record: dict[str, Any]) -> str:
    return normalize_whitespace(record.get("image_key") or record.get("image") or record.get("source") or "")


def build_cleaner_response_schema() -> dict[str, Any]:
    return {
        "type": "object",
        "properties": {
            "cleaned_caption": {"type": "string"},
            "removed_or_rejected_details": {"type": "array", "items": {"type": "string"}},
            "corrected_details": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "from": {"type": "string"},
                        "to": {"type": "string"},
                        "reason": {"type": "string"},
                    },
                    "required": ["from", "to"],
                    "additionalProperties": True,
                },
            },
            "uncertain_details": {"type": "array", "items": {"type": "string"}},
            "visual_validation_notes": {"type": "array", "items": {"type": "string"}},
        },
        "required": [
            "cleaned_caption",
            "removed_or_rejected_details",
            "corrected_details",
            "uncertain_details",
            "visual_validation_notes",
        ],
        "additionalProperties": True,
    }


DEFAULT_VLM_CLEANER_PROMPT = (
    "You are CaptionForge Pass B: an image-grounded caption cleaner. "
    "You receive one image and one raw caption written by another image captioning model. "
    "Your job is not to summarize, improve style, write a new caption, or make the caption shorter. "
    "Your job is to check every visual statement in the caption against the image. "
    "Keep statements that are clearly supported by the image. Delete statements that are false, contradicted, "
    "not visible, or too speculative. Correct small wording errors only when the intended visible detail is obvious. "
    "Do not add new visual details. Do not compress the caption. Do not replace specific details with generic phrases. "
    "Preserve the original level of detail whenever supported. Output only the cleaned caption and audit fields."
)


def build_cleaner_prompt_bundle(record: dict[str, Any], image_path: Optional[Path], config: VLMValidatorConfig) -> dict[str, Any]:
    raw_caption = truncate_text(_cf_cleaner_get_raw_caption(record), int(config.max_input_caption_chars))
    image_key = _cf_cleaner_record_key(record)
    instruction = normalize_whitespace(config.vlm_validator_prompt or DEFAULT_VLM_CLEANER_PROMPT)
    if "image-grounded caption cleaner" not in instruction.lower() and "caption cleaner" not in instruction.lower():
        instruction = DEFAULT_VLM_CLEANER_PROMPT

    payload = {
        "image_key": image_key,
        "source_model_family": str(record.get("model_family") or ""),
        "source_model_name": str(record.get("model_name") or ""),
        "source_ensemble_run_index": record.get("ensemble_run_index", 0),
        "raw_caption": raw_caption,
        "required_output": build_cleaner_response_schema(),
    }
    prompt = (
        f"{instruction}\n\n"
        "Return strict JSON only. No markdown. No prose outside JSON.\n\n"
        "CLEANING RULES:\n"
        "1. Work statement by statement through the raw caption.\n"
        "2. Keep each supported statement with as much original specificity as possible.\n"
        "3. Delete unsupported, false, contradicted, invisible, or speculative statements.\n"
        "4. Correct a statement only when the correction is obvious from the image.\n"
        "5. Do not add details that are absent from the raw caption, even if visible in the image.\n"
        "6. Do not summarize or shorten supported detail.\n"
        "7. Do not replace concrete details with generic phrases.\n"
        "8. The cleaned_caption should be one coherent paragraph, but it may remain detailed.\n\n"
        "Input JSON:\n"
        f"{json.dumps(payload, ensure_ascii=False, indent=2)}\n"
    )
    image_info = {
        "resolved_path": str(image_path) if image_path else "",
        "exists": bool(image_path and image_path.exists()),
        "mime_type": image_mime_type(image_path) if image_path else "",
        "size_bytes": image_path.stat().st_size if image_path and image_path.exists() else 0,
        "sha1": sha1_file(image_path) if image_path and image_path.exists() else "",
    }
    return {
        "captionforge_pass": "B_VLM_CLEANER_PROMPT",
        "prompt_schema_version": "captionforge-vlm-cleaner-prompt-v0.1.0",
        "image_key": image_key,
        "image": str(record.get("image") or ""),
        "image_info": image_info,
        "prompt": prompt,
        "prompt_payload": payload,
        "prompt_sha1": sha1_text(prompt),
        "timestamp": iso_timestamp(),
    }


def _cf_parse_cleaner_json(raw_response: str) -> dict[str, Any]:
    obj = extract_json_object(raw_response)
    cleaned = normalize_whitespace(
        obj.get("cleaned_caption")
        or obj.get("validated_caption")
        or obj.get("caption")
        or obj.get("final_caption")
        or ""
    )
    if not cleaned:
        raise VLMResponseParseError("Cleaner response did not contain cleaned_caption.", raw_response)
    return {
        "cleaned_caption": cleaned,
        "removed_or_rejected_details": coerce_string_list(obj.get("removed_or_rejected_details") or obj.get("removed_details") or obj.get("rejected_details") or []),
        "corrected_details": coerce_corrections(obj.get("corrected_details") or obj.get("corrections") or []),
        "uncertain_details": coerce_string_list(obj.get("uncertain_details") or []),
        "visual_validation_notes": coerce_string_list(obj.get("visual_validation_notes") or obj.get("validation_notes") or []),
    }


def _cf_cleaner_record(
    *,
    source_record: dict[str, Any],
    image_path: Optional[Path],
    config: VLMValidatorConfig,
    prompt_bundle: dict[str, Any],
    parsed: dict[str, Any],
    raw_response: str,
    status: str,
    error: str = "",
    elapsed_sec: float = 0.0,
) -> dict[str, Any]:
    raw_caption = _cf_cleaner_get_raw_caption(source_record)
    cleaned = normalize_whitespace(parsed.get("cleaned_caption") or "")
    return {
        "captionforge_pass": CLEANER_PASS,
        "engine": "CaptionForge VLM Validator Engine",
        "engine_version": getattr(config, "engine_version", "0.2.0"),
        "contract": "vlm_statement_cleaner_v0.1",
        "image_key": _cf_cleaner_record_key(source_record),
        "image": str(source_record.get("image") or ""),
        "image_resolved_path": str(image_path) if image_path else "",
        "status": status,
        "model_family": str(source_record.get("model_family") or ""),
        "model_name": str(source_record.get("model_name") or ""),
        "ensemble_run_index": int(source_record.get("ensemble_run_index") or 0),
        "raw_caption": raw_caption,
        "caption": cleaned,
        "cleaned_caption": cleaned,
        "cleaned_caption_narrative": cleaned,
        "cleaned_caption_comma": normalize_comma_caption(cleaned),
        "removed_or_rejected_details": parsed.get("removed_or_rejected_details", []),
        "corrected_details": parsed.get("corrected_details", []),
        "uncertain_details": parsed.get("uncertain_details", []),
        "visual_validation_notes": parsed.get("visual_validation_notes", []),
        "source_caption_record": source_record,
        "params": {
            "vlm_backend": config.vlm_backend,
            "vlm_model": config.vlm_model,
            "ollama_num_predict": config.ollama_num_predict,
            "ollama_temperature": config.ollama_temperature,
            "ollama_top_p": config.ollama_top_p,
            "ollama_top_k": config.ollama_top_k,
            "ollama_seed": int(getattr(config, "ollama_seed", -1)),
        },
        "prompt_sha1": prompt_bundle.get("prompt_sha1", ""),
        "raw_vlm_response": raw_response if config.preserve_raw_vlm_response else "",
        "metrics": {
            "raw_caption_chars": len(raw_caption),
            "cleaned_caption_chars": len(cleaned),
            "removed_count": len(parsed.get("removed_or_rejected_details", [])),
            "corrected_count": len(parsed.get("corrected_details", [])),
            "raw_response_chars": len(raw_response or ""),
            "elapsed_sec": round(float(elapsed_sec), 3),
        },
        "timestamp": iso_timestamp(),
        "error": error,
    }


def extract_clean_batch(batch: BatchVLMValidatorConfig, config: Optional[VLMValidatorConfig] = None) -> BatchVLMValidatorResult:
    """Experimental reversed-pipeline Pass B: clean each raw Pass A caption against the image.

    Input: A_RAW_CAPTIONS JSONL, one record per raw caption.
    Output: B_VLM_CLEANED JSONL, one record per input caption. The VLM is deliberately
    narrow: it deletes/corrects unsupported statements without adding new details or compressing.
    """
    config = config or VLMValidatorConfig()
    records = [r for r in read_jsonl(batch.input_jsonl) if _cf_cleaner_pass_a_usable(r)]
    if batch.image_key_filter:
        records = [r for r in records if _cf_cleaner_record_key(r) == batch.image_key_filter]
    if batch.limit_images and batch.limit_images > 0:
        # Limit by image group count rather than raw-record count.
        allowed: list[str] = []
        for rec in records:
            key = _cf_cleaner_record_key(rec)
            if key not in allowed:
                allowed.append(key)
            if len(allowed) >= int(batch.limit_images):
                break
        records = [r for r in records if _cf_cleaner_record_key(r) in set(allowed)]

    out_path = Path(batch.output_jsonl) if batch.output_jsonl else default_output_path(batch.input_jsonl)
    prompt_path = Path(batch.prompt_jsonl) if batch.prompt_jsonl else default_prompt_output_path(out_path)
    sidecar_dir = Path(batch.readable_sidecar_dir) if batch.readable_sidecar_dir else default_sidecar_dir(out_path)

    if batch.write_jsonl and batch.overwrite and not batch.dry_run:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text("", encoding="utf-8")
    if batch.write_prompt_jsonl and batch.overwrite and not batch.dry_run:
        prompt_path.parent.mkdir(parents=True, exist_ok=True)
        prompt_path.write_text("", encoding="utf-8")
    if batch.write_readable_sidecars and not batch.dry_run:
        sidecar_dir.mkdir(parents=True, exist_ok=True)

    result = BatchVLMValidatorResult()
    total = len(records)
    print(f"[CaptionForge VLM Cleaner] Found {total} raw caption record(s).", flush=True)

    old_format = str(getattr(config, "ollama_format_mode", "json") or "json")
    config.ollama_format_mode = "json"
    try:
        for i, rec in enumerate(records, start=1):
            key = _cf_cleaner_record_key(rec)
            image_path = resolve_image_path(rec, batch.image_root)
            print(f"[{i}/{total}] CLEAN {key} | {rec.get('model_family','')} run {rec.get('ensemble_run_index',0)}", flush=True)
            if image_path is None and config.vlm_backend == "ollama" and not config.allow_missing_images:
                parsed = {"cleaned_caption": ""}
                out = _cf_cleaner_record(
                    source_record=rec, image_path=None, config=config, prompt_bundle={}, parsed=parsed,
                    raw_response="", status="error", error=f"Image could not be resolved for {key}", elapsed_sec=0.0,
                )
            else:
                prompt_bundle = build_cleaner_prompt_bundle(rec, image_path, config)
                if batch.write_prompt_jsonl:
                    append_jsonl_record(prompt_path, prompt_bundle, dry_run=batch.dry_run)
                if config.vlm_backend == "prompt_only":
                    parsed = {"cleaned_caption": _cf_cleaner_get_raw_caption(rec), "visual_validation_notes": ["prompt_only backend; caption was not image-cleaned."]}
                    raw = ""
                    elapsed = 0.0
                    status = "prompt_only"
                    error = ""
                else:
                    t0 = time.time()
                    raw = ollama_generate_vlm_response(prompt_bundle, image_path, config) if config.vlm_backend == "ollama" else ""
                    elapsed = time.time() - t0
                    parsed = _cf_parse_cleaner_json(raw)
                    status = "ok"
                    error = ""
                out = _cf_cleaner_record(
                    source_record=rec, image_path=image_path, config=config, prompt_bundle=prompt_bundle, parsed=parsed,
                    raw_response=raw, status=status, error=error, elapsed_sec=elapsed,
                )
            result.records.append(out)  # type: ignore[arg-type]
            result.failed += int(out.get("status") not in {"ok", "prompt_only"})
            if batch.write_jsonl:
                append_jsonl_record(out_path, out, dry_run=batch.dry_run)
            if batch.write_readable_sidecars and not batch.dry_run:
                stem = stable_filename_from_image_key(f"{key}_{rec.get('model_family','model')}_{rec.get('ensemble_run_index',0)}")
                (sidecar_dir / stem).write_text(json.dumps(json_safe(out), ensure_ascii=False, indent=2), encoding="utf-8")
    finally:
        config.ollama_format_mode = old_format

    print(
        f"[CaptionForge VLM Cleaner] Done. ok={result.processed} failed={result.failed} output={out_path}",
        flush=True,
    )
    return result
