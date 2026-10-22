"""Pass A ledger integrity and persisted Ollama operation identity."""

from __future__ import annotations

import json
import os
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Any


OLLAMA_OPERATION_FIELDS = (
    "image_key", "model_name", "ensemble_run_index", "prompt", "system_prompt",
    "seed", "temperature", "top_p", "top_k", "repetition_penalty",
    "max_new_tokens", "max_size",
)


def ollama_operation_key(record: dict[str, Any]) -> tuple | None:
    """Incomplete legacy identities must never collapse distinct witnesses."""
    if record.get("captionforge_pass") != "A" or record.get("model_family") != "ollama":
        return None
    if any(field not in record for field in OLLAMA_OPERATION_FIELDS):
        return None
    if not record["image_key"] or not record["model_name"]:
        return None
    values = tuple(record[field] for field in OLLAMA_OPERATION_FIELDS)
    if any(not isinstance(value, (str, int, float, type(None))) for value in values):
        return None
    return values


def successful_caption(record: dict[str, Any]) -> bool:
    return record.get("status") == "ok" and isinstance(record.get("caption"), str) and bool(record["caption"].strip())


def read_pass_a_jsonl(path: Path, *, repair_tail: bool = False) -> list[dict[str, Any]]:
    """Recover a damaged final record; fail closed on interior corruption.

    Readers leave bytes untouched. Before an append, repair_tail saves a durable
    timestamped copy and atomically replaces the ledger with its valid prefix.
    Valid records are copied byte for byte, never reserialized.
    """
    if not path.exists():
        return []
    original = path.read_bytes()
    lines = original.split(b"\n")
    last_record_line = max((index for index, line in enumerate(lines) if line.strip()), default=-1)
    records: list[dict[str, Any]] = []
    offset = 0
    repaired = original
    for index, line in enumerate(lines):
        if line.strip():
            try:
                obj = json.loads(line.decode("utf-8"))
            except (UnicodeDecodeError, json.JSONDecodeError) as exc:
                if index != last_record_line:
                    raise RuntimeError(f"Invalid Pass A JSONL at {path}:{index + 1} (byte {offset}): {exc}. File left unchanged.") from exc
                repaired = original[:offset]
                print(f"[CaptionForge] Incomplete Pass A tail at {path}:{index + 1}; recovering {len(records)} complete records.", flush=True)
                break
            if not isinstance(obj, dict):
                raise RuntimeError(f"Invalid Pass A JSONL object at {path}:{index + 1} (byte {offset}). File left unchanged.")
            records.append(obj)
        offset += len(line) + 1
    if repaired and not repaired.endswith(b"\n"):
        repaired += b"\n"
    if repair_tail and repaired != original:
        backup = path.with_name(f"{path.name}.{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}.bak")
        with backup.open("xb") as stream:
            stream.write(original)
            stream.flush()
            os.fsync(stream.fileno())
        print(f"[CaptionForge] Pass A original bytes preserved at {backup}", flush=True)
        temporary_path = None
        try:
            with tempfile.NamedTemporaryFile(dir=path.parent, prefix=f".{path.name}.", delete=False) as stream:
                temporary_path = Path(stream.name)
                stream.write(repaired)
                stream.flush()
                os.fsync(stream.fileno())
            if path.read_bytes() != original:
                raise RuntimeError(f"Pass A JSONL changed during recovery: {path}. Stop other writers and retry; backup: {backup}")
            os.replace(temporary_path, path)
        finally:
            if temporary_path is not None and temporary_path.exists():
                temporary_path.unlink()
    return records


def deduplicate_ollama_completions(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Keep latest success per fully specified operation without editing evidence."""
    positions: dict[tuple, int] = {}
    selected: list[dict[str, Any]] = []
    for record in records:
        key = ollama_operation_key(record) if successful_caption(record) else None
        if key is not None:
            if key in positions:
                selected[positions[key]] = record
                continue
            positions[key] = len(selected)
        selected.append(record)
    return selected
