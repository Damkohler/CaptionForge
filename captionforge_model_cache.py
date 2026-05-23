"""
CaptionForge Global Model Cache Manager

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
    - The **CaptionForge Global Model Cache Manager** provides a shared,
      process-local cache for heavyweight captioning and language models used
      by CaptionForge engines.

    - It is designed for workflows where multiple local captioning systems may
      be available, but limited VRAM makes simultaneous residency impractical.

    - The cache manager:
            • Builds stable model cache keys
            • Reuses already-loaded model bundles
            • Registers newly loaded model handles
            • Evicts incompatible or excess resident models
            • Supports optional engine-provided unload callbacks
            • Clears Python references and CUDA allocator leftovers
            • Provides lightweight cache diagnostics

    - Typical cached objects may include:
            • Qwen / VLM caption model bundles
            • Joy caption model bundles
            • future cleanup LLMs
            • future validator / claim-refinement models
            • engine-specific processor/tokenizer/model containers

- Execution Model
    - The cache is global, process-local, and protected by a re-entrant lock.

    - Default policy keeps only one heavyweight CaptionForge model resident
      at a time, which is appropriate for constrained VRAM systems.

    - Eviction behavior is intentionally conservative:
            • Engine code remains responsible for loading models
            • Engine code may provide unload callbacks
            • The cache deletes references but does not assume model internals
            • CUDA cleanup is attempted only when PyTorch/CUDA are available

    - Cache keys include fields that materially affect model residency:
            • role
            • family
            • model path
            • device
            • quantization mode
            • dtype
            • revision

- Design Philosophy
    - CaptionForge treats model loading as an engine concern and model residency
      as shared infrastructure.

    - This keeps Qwen, Joy, future LLM validators, and future caption engines
      under one cache policy without making any single engine canonical.

    - The module is intentionally small, explicit, and dependency-light so it can
      be reused across CaptionForge nodes without introducing a larger runtime
      framework.

- ⚠️ Development Status
    - This is early CaptionForge infrastructure intended for local ComfyUI use.
    - Cache policy and diagnostics may evolve as additional engines and live LLM
      refinement passes are added.
    - The default single-model residency policy is conservative and may be
      adjusted in future versions for larger VRAM systems.

- Attribution & License
  - Concept and implementation by **J. L. Córdova**
    with development assistance from **ChatGPT (OpenAI)**.

  - Copyright (c) 2026 J. L. Córdova

  - Released under the **MIT License**.
"""

from __future__ import annotations

MANIFEST = {
    "name": "CaptionForge Global Model Cache Manager",
    "version": (0, 1, 0),
    "author": "J. L. Córdova",
    "description": (
        "Shared process-local cache manager for heavyweight CaptionForge model bundles. "
        "Provides stable cache keys, model registration, reuse, eviction, optional unload "
        "callbacks, CUDA cleanup, and lightweight diagnostics for model-agnostic captioning "
        "workflows. Designed to prevent unnecessary Qwen/Joy/future-LLM co-residency on "
        "limited-VRAM ComfyUI systems while keeping engine-specific loading logic separate "
        "from shared residency policy."
    ),
}

import gc
import time
import threading
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Optional


try:
    import torch
except Exception:  # Keeps import safe if torch is unavailable during metadata inspection.
    torch = None


UnloadFn = Callable[[Any], None]


# Default: only one heavyweight CaptionForge model resident.
_MAX_LOADED_MODELS = 1

# Global process-local cache.
_CACHE_LOCK = threading.RLock()
_MODEL_CACHE: Dict[str, "CacheEntry"] = {}


@dataclass
class CacheEntry:
    key: str
    obj: Any
    family: str
    model_path: str
    device: str
    quantization: str = "none"
    role: str = "caption"
    unload_fn: Optional[UnloadFn] = None
    created_at: float = field(default_factory=time.time)
    last_used_at: float = field(default_factory=time.time)
    keep: bool = False

    def touch(self) -> None:
        self.last_used_at = time.time()


def normalize_cache_part(value: Any) -> str:
    if value is None:
        return "none"
    text = str(value).strip()
    return text if text else "none"


def make_cache_key(
    *,
    family: str,
    model_path: str,
    device: str = "cuda",
    quantization: str = "none",
    dtype: str = "auto",
    revision: str = "default",
    role: str = "caption",
) -> str:
    """
    Build a stable cache key.

    Include fields that change the actual resident model object:
    - family: qwen, joy, cleanup_llm, validator, etc.
    - model_path: local path or HF repo id
    - device: cuda/cpu
    - quantization: fp16, bf16, fp8, 8bit, none, etc.
    - dtype: bf16/fp16/fp32/auto
    - revision: optional model revision
    - role: caption/cleanup/validator
    """
    parts = [
        normalize_cache_part(role).lower(),
        normalize_cache_part(family).lower(),
        normalize_cache_part(model_path),
        normalize_cache_part(device).lower(),
        normalize_cache_part(quantization).lower(),
        normalize_cache_part(dtype).lower(),
        normalize_cache_part(revision),
    ]
    return "::".join(parts)


def set_max_loaded_models(max_loaded_models: int) -> None:
    """
    Adjust global cache capacity.

    Normally leave this at 1 for 16 GB VRAM systems.
    """
    global _MAX_LOADED_MODELS

    value = int(max_loaded_models)
    if value < 0:
        raise ValueError("max_loaded_models must be >= 0")

    with _CACHE_LOCK:
        _MAX_LOADED_MODELS = value
        _enforce_capacity_locked()


def get_max_loaded_models() -> int:
    with _CACHE_LOCK:
        return _MAX_LOADED_MODELS


def get_cached_model(key: str) -> Optional[Any]:
    """
    Return cached model bundle/object if present.
    """
    with _CACHE_LOCK:
        entry = _MODEL_CACHE.get(key)
        if entry is None:
            return None

        entry.touch()
        print(f"[CaptionForge Cache] Reusing cached model: {key}")
        return entry.obj


def register_model(
    key: str,
    obj: Any,
    *,
    family: str,
    model_path: str,
    device: str = "cuda",
    quantization: str = "none",
    role: str = "caption",
    unload_fn: Optional[UnloadFn] = None,
    keep: bool = False,
) -> Any:
    """
    Register a loaded model bundle/object.

    The object can be:
    - a model
    - a tuple/dict containing model + processor/tokenizer
    - an engine-specific bundle dataclass
    """
    with _CACHE_LOCK:
        existing = _MODEL_CACHE.get(key)
        if existing is not None:
            existing.obj = obj
            existing.family = family
            existing.model_path = model_path
            existing.device = device
            existing.quantization = quantization
            existing.role = role
            existing.unload_fn = unload_fn
            existing.keep = keep
            existing.touch()
        else:
            _MODEL_CACHE[key] = CacheEntry(
                key=key,
                obj=obj,
                family=family,
                model_path=model_path,
                device=device,
                quantization=quantization,
                role=role,
                unload_fn=unload_fn,
                keep=keep,
            )

        print(f"[CaptionForge Cache] Registered model: {key}")
        _enforce_capacity_locked(protected_key=key)
        return obj


def prepare_for_model_load(
    key: str,
    *,
    policy: str = "evict_other_caption_models",
    family: Optional[str] = None,
    role: str = "caption",
    max_loaded_models: Optional[int] = None,
) -> None:
    """
    Call this immediately before loading a heavyweight model.

    Policies:
    - keep_this_model:
        Do not evict anything immediately. Capacity is still enforced after register_model.
    - evict_other_caption_models:
        Evict other caption-role models before loading this one.
    - unload_after_run:
        Same pre-load behavior as evict_other_caption_models; caller should evict this model after run.
    - none:
        Do nothing before load.
    """
    with _CACHE_LOCK:
        if max_loaded_models is not None:
            set_max_loaded_models(max_loaded_models)

        if policy == "none":
            return

        if policy == "keep_this_model":
            return

        if policy in {"evict_other_caption_models", "unload_after_run"}:
            for existing_key, entry in list(_MODEL_CACHE.items()):
                if existing_key == key:
                    continue

                # Future cleanup/validator models can use role="cleanup" or "validator".
                # For now, the default policy primarily protects against Qwen/Joy co-residency.
                if entry.role == role or entry.role == "caption":
                    _evict_locked(existing_key, reason=f"policy={policy}")

            _cuda_cleanup()
            return

        raise ValueError(
            f"Unknown CaptionForge cache policy: {policy!r}. "
            "Expected one of: keep_this_model, evict_other_caption_models, unload_after_run, none."
        )


def evict_model(key: str, *, reason: str = "manual") -> bool:
    """
    Evict one cached model by key.
    """
    with _CACHE_LOCK:
        did_evict = _evict_locked(key, reason=reason)
        _cuda_cleanup()
        return did_evict


def evict_family(family: str, *, reason: str = "manual_family_evict") -> int:
    """
    Evict all models for a family, e.g. 'qwen' or 'joy'.
    """
    family_norm = normalize_cache_part(family).lower()
    count = 0

    with _CACHE_LOCK:
        for key, entry in list(_MODEL_CACHE.items()):
            if entry.family.lower() == family_norm:
                if _evict_locked(key, reason=reason):
                    count += 1

        _cuda_cleanup()
        return count


def unload_all(*, include_keep: bool = True, reason: str = "manual_unload_all") -> int:
    """
    Clear the entire CaptionForge cache.
    """
    count = 0

    with _CACHE_LOCK:
        for key in list(_MODEL_CACHE.keys()):
            entry = _MODEL_CACHE.get(key)
            if entry is None:
                continue
            if entry.keep and not include_keep:
                continue

            if _evict_locked(key, reason=reason):
                count += 1

        _cuda_cleanup()
        return count


def unload_after_run(key: str, *, enabled: bool) -> None:
    """
    Convenience helper for node/engine code after a captioning run.
    """
    if enabled:
        evict_model(key, reason="unload_after_run")


def cache_info() -> Dict[str, Any]:
    """
    Return lightweight cache diagnostics for console logging or future UI/debug nodes.
    """
    with _CACHE_LOCK:
        entries = []
        for key, entry in _MODEL_CACHE.items():
            entries.append(
                {
                    "key": key,
                    "family": entry.family,
                    "model_path": entry.model_path,
                    "device": entry.device,
                    "quantization": entry.quantization,
                    "role": entry.role,
                    "keep": entry.keep,
                    "age_sec": round(time.time() - entry.created_at, 3),
                    "idle_sec": round(time.time() - entry.last_used_at, 3),
                }
            )

        return {
            "max_loaded_models": _MAX_LOADED_MODELS,
            "loaded_count": len(_MODEL_CACHE),
            "entries": entries,
        }


def _enforce_capacity_locked(protected_key: Optional[str] = None) -> None:
    """
    Enforce max loaded model count.

    Eviction order:
    1. Non-keep entries first
    2. Least recently used first
    3. Avoid protected_key if possible
    """
    if _MAX_LOADED_MODELS < 0:
        return

    while len(_MODEL_CACHE) > _MAX_LOADED_MODELS:
        candidates = []

        for key, entry in _MODEL_CACHE.items():
            if key == protected_key:
                continue
            if entry.keep:
                continue
            candidates.append(entry)

        # If everything is protected/keep, do not force-evict.
        if not candidates:
            break

        victim = min(candidates, key=lambda e: e.last_used_at)
        _evict_locked(victim.key, reason="capacity")


def _evict_locked(key: str, *, reason: str = "unspecified") -> bool:
    entry = _MODEL_CACHE.pop(key, None)
    if entry is None:
        return False

    print(f"[CaptionForge Cache] Evicting model: {key} | reason={reason}")

    try:
        if entry.unload_fn is not None:
            entry.unload_fn(entry.obj)
    except Exception as exc:
        print(f"[CaptionForge Cache] Warning: unload_fn failed for {key}: {exc}")

    try:
        entry.obj = None
    except Exception:
        pass

    del entry
    gc.collect()
    return True


def _cuda_cleanup() -> None:
    """
    Clear Python and CUDA allocator leftovers after model eviction.
    """
    gc.collect()

    if torch is None:
        return

    try:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            try:
                torch.cuda.ipc_collect()
            except Exception:
                # ipc_collect can fail harmlessly on some Windows/CUDA setups.
                pass
    except Exception as exc:
        print(f"[CaptionForge Cache] Warning: CUDA cleanup failed: {exc}")