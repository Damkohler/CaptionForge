"""
CaptionForge Global Model Cache Manager

- CaptionForge
  - This module is part of **CaptionForge**, a model-agnostic captioning
    framework for ComfyUI developed by **J. L. Córdova**.

  - Repository:
    https://github.com/Damkohler/CaptionForge

- CaptionForge focuses on practical dataset-captioning infrastructure for
  LoRA dataset preparation, using multi-engine caption generation, JSONL
  audit trails, claim extraction and refinement, text-LLM distillation,
  image-aware VLM validation, and consensus-oriented caption improvement
  to produce grounded, auditable training captions.

- Engine Purpose
    - The **CaptionForge Global Model Cache Manager** provides shared,
      process-local residency management for heavyweight CaptionForge model
      bundles.

    - It supports local ComfyUI workflows where Joy, Qwen, and other
      captioning engines may be available, but limited VRAM makes simultaneous
      residency impractical.

    - The cache manager is responsible for:
            • stable cache-key construction
            • reuse of already-loaded model bundles
            • registration of newly loaded model handles
            • eviction of incompatible or excess resident models
            • optional engine-provided unload callbacks
            • Python reference cleanup
            • best-effort CUDA allocator cleanup
            • lightweight cache diagnostics

    - Cached objects may include model instances, processor/tokenizer bundles,
      engine-specific dataclass containers, or dictionaries/tuples wrapping
      those objects.

- CaptionForge Pipeline Role
    - This module is shared infrastructure, not a captioning pass.

    - It supports CaptionForge Pass A caption witness engines by reducing
      avoidable model reloads while also protecting constrained-VRAM systems
      from accidental multi-model co-residency.

    - It does not own model loading logic. Engines remain responsible for
      constructing their own models and may register unload callbacks when they
      need custom teardown behavior.

- Execution Model
    - The cache is global, process-local, and protected by a re-entrant lock.

    - The default policy keeps only one heavyweight CaptionForge model resident
      at a time.

    - Cache keys include fields that materially affect model residency:
            • role
            • family
            • model path
            • device
            • quantization mode
            • dtype
            • revision

    - CUDA cleanup is attempted only when PyTorch and CUDA are available.

- Design Philosophy
    - CaptionForge treats model loading as an engine concern and model residency
      as shared infrastructure.

    - This keeps the cache small, explicit, and dependency-light while allowing
      multiple CaptionForge engines to share one conservative VRAM policy.

    - The module favors predictable behavior over aggressive automatic memory
      management.

- Development Status
    - CaptionForge v0.1.0 experimental developer-preview infrastructure.
    - Cache policy and diagnostics may evolve as CaptionForge's supported engine
      set matures.

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

MANIFEST = {
    "name": "CaptionForge Global Model Cache Manager",
    "version": CAPTIONFORGE_VERSION,
    "author": "J. L. Córdova",
    "description": (
        "Shared process-local cache manager for heavyweight CaptionForge model "
        "bundles. Provides stable cache keys, model registration, reuse, "
        "eviction, optional unload callbacks, CUDA cleanup, and lightweight "
        "diagnostics while keeping engine-specific loading logic separate from "
        "shared residency policy."
    ),
}

import gc
import os
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

# Safe eviction/cuda-cleanup defaults.
# Existing fast policies use synchronization + cleanup but no cooldown.
# Explicit *_safe policies add a short cooldown to reduce driver/watchdog stress
# during rapid Qwen/Joy/bitsandbytes model swaps.
def _env_float(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name, default))
    except Exception:
        return float(default)


_SAFE_EVICTION_COOLDOWN_SEC = max(0.0, _env_float("CAPTIONFORGE_SAFE_EVICTION_COOLDOWN_SEC", 1.25))
_FAST_EVICTION_COOLDOWN_SEC = max(0.0, _env_float("CAPTIONFORGE_FAST_EVICTION_COOLDOWN_SEC", 0.0))
_VERBOSE_SAFE_EVICTION = os.environ.get("CAPTIONFORGE_CACHE_VERBOSE", "0").strip().lower() in {"1", "true", "yes", "on"}

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
        Evict other caption-role models before loading this one, then run standard CUDA cleanup.
    - evict_other_caption_models_safe:
        Same as evict_other_caption_models, but with extra synchronization and a short cooldown.
    - unload_after_run:
        Same pre-load behavior as evict_other_caption_models; caller should evict this model after run.
    - unload_after_run_safe:
        Same pre-load behavior as evict_other_caption_models_safe; caller should safely evict after run.
    - none:
        Do nothing before load.
    """
    policy = (policy or "").strip() or "evict_other_caption_models"
    safe_mode = policy.endswith("_safe")

    with _CACHE_LOCK:
        if max_loaded_models is not None:
            set_max_loaded_models(max_loaded_models)

        if policy == "none":
            return

        if policy == "keep_this_model":
            return

        if policy in {
            "evict_other_caption_models",
            "evict_other_caption_models_safe",
            "unload_after_run",
            "unload_after_run_safe",
        }:
            did_evict = False
            for existing_key, entry in list(_MODEL_CACHE.items()):
                if existing_key == key:
                    continue

                # Future cleanup/validator models can use role="cleanup" or "validator".
                # For now, the default policy primarily protects against Qwen/Joy co-residency.
                if entry.role == role or entry.role == "caption":
                    did_evict = _evict_locked(existing_key, reason=f"policy={policy}", safe=safe_mode) or did_evict

            _cuda_cleanup(
                reason=f"post_prepare_for_model_load:{policy}",
                synchronize=True,
                cooldown_sec=_SAFE_EVICTION_COOLDOWN_SEC if safe_mode and did_evict else _FAST_EVICTION_COOLDOWN_SEC,
            )
            return

        raise ValueError(
            f"Unknown CaptionForge cache policy: {policy!r}. "
            "Expected one of: keep_this_model, evict_other_caption_models, "
            "evict_other_caption_models_safe, unload_after_run, unload_after_run_safe, none."
        )


def evict_model(key: str, *, reason: str = "manual", safe: bool = False) -> bool:
    """
    Evict one cached model by key.
    """
    with _CACHE_LOCK:
        did_evict = _evict_locked(key, reason=reason, safe=safe)
        _cuda_cleanup(
            reason=f"post_evict_model:{reason}",
            synchronize=True,
            cooldown_sec=_SAFE_EVICTION_COOLDOWN_SEC if safe and did_evict else _FAST_EVICTION_COOLDOWN_SEC,
        )
        return did_evict


def evict_family(family: str, *, reason: str = "manual_family_evict", safe: bool = False) -> int:
    """
    Evict all models for a family, e.g. 'qwen' or 'joy'.
    """
    family_norm = normalize_cache_part(family).lower()
    count = 0

    with _CACHE_LOCK:
        for key, entry in list(_MODEL_CACHE.items()):
            if entry.family.lower() == family_norm:
                if _evict_locked(key, reason=reason, safe=safe):
                    count += 1

        _cuda_cleanup(
            reason=f"post_evict_family:{family_norm}",
            synchronize=True,
            cooldown_sec=_SAFE_EVICTION_COOLDOWN_SEC if safe and count else _FAST_EVICTION_COOLDOWN_SEC,
        )
        return count


def unload_all(*, include_keep: bool = True, reason: str = "manual_unload_all", safe: bool = False) -> int:
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

            if _evict_locked(key, reason=reason, safe=safe):
                count += 1

        _cuda_cleanup(
            reason=f"post_unload_all:{reason}",
            synchronize=True,
            cooldown_sec=_SAFE_EVICTION_COOLDOWN_SEC if safe and count else _FAST_EVICTION_COOLDOWN_SEC,
        )
        return count


def unload_after_run(key: str, *, enabled: bool, safe: bool = False) -> None:
    """
    Convenience helper for node/engine code after a captioning run.
    """
    if enabled:
        evict_model(key, reason="unload_after_run_safe" if safe else "unload_after_run", safe=safe)


def set_safe_eviction_cooldown(seconds: float) -> None:
    """
    Adjust the cooldown used by explicit *_safe cache policies.
    """
    global _SAFE_EVICTION_COOLDOWN_SEC
    _SAFE_EVICTION_COOLDOWN_SEC = max(0.0, float(seconds))


def get_safe_eviction_cooldown() -> float:
    return _SAFE_EVICTION_COOLDOWN_SEC


def set_fast_eviction_cooldown(seconds: float) -> None:
    """
    Adjust the cooldown used by normal/fast eviction policies. Defaults to 0.0.
    """
    global _FAST_EVICTION_COOLDOWN_SEC
    _FAST_EVICTION_COOLDOWN_SEC = max(0.0, float(seconds))


def get_fast_eviction_cooldown() -> float:
    return _FAST_EVICTION_COOLDOWN_SEC


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
            "safe_eviction_cooldown_sec": _SAFE_EVICTION_COOLDOWN_SEC,
            "fast_eviction_cooldown_sec": _FAST_EVICTION_COOLDOWN_SEC,
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
        _evict_locked(victim.key, reason="capacity", safe=False)
        _cuda_cleanup(reason="post_capacity_evict", synchronize=True, cooldown_sec=_FAST_EVICTION_COOLDOWN_SEC)


def _evict_locked(key: str, *, reason: str = "unspecified", safe: bool = False) -> bool:
    entry = _MODEL_CACHE.pop(key, None)
    if entry is None:
        return False

    print(f"[CaptionForge Cache] Evicting model: {key} | reason={reason}")

    # Synchronize before tearing down CUDA-backed model objects. This adds little
    # overhead when kernels are already idle, but gives bitsandbytes/accelerate
    # paths a cleaner boundary before references are cleared.
    _cuda_synchronize(reason=f"pre_unload:{reason}", verbose=safe or _VERBOSE_SAFE_EVICTION)

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

    # A second synchronize catches any work triggered by engine unload hooks.
    _cuda_synchronize(reason=f"post_unload:{reason}", verbose=safe or _VERBOSE_SAFE_EVICTION)
    return True


def _cuda_synchronize(*, reason: str = "", verbose: bool = False) -> None:
    if torch is None:
        return

    try:
        if torch.cuda.is_available():
            if verbose:
                print(f"[CaptionForge Cache] CUDA synchronize start ({reason})")
            torch.cuda.synchronize()
            if verbose:
                print(f"[CaptionForge Cache] CUDA synchronize done ({reason})")
    except Exception as exc:
        print(f"[CaptionForge Cache] Warning: CUDA synchronize failed ({reason}): {exc}")


def _cuda_cleanup(
    *,
    reason: str = "cuda_cleanup",
    synchronize: bool = True,
    cooldown_sec: float = 0.0,
) -> None:
    """
    Clear Python and CUDA allocator leftovers after model eviction.

    synchronize=True gives native CUDA/bitsandbytes/accelerate teardown a cleaner
    boundary. cooldown_sec is intentionally opt-in/nonzero only for safe policies
    unless the user sets environment overrides.
    """
    gc.collect()

    if torch is None:
        if cooldown_sec > 0:
            time.sleep(cooldown_sec)
        return

    if synchronize:
        _cuda_synchronize(reason=f"pre_cleanup:{reason}", verbose=_VERBOSE_SAFE_EVICTION)

    try:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            try:
                torch.cuda.ipc_collect()
            except Exception:
                # ipc_collect can fail harmlessly on some Windows/CUDA setups.
                pass
    except Exception as exc:
        print(f"[CaptionForge Cache] Warning: CUDA cleanup failed ({reason}): {exc}")

    gc.collect()

    if synchronize:
        _cuda_synchronize(reason=f"post_cleanup:{reason}", verbose=_VERBOSE_SAFE_EVICTION)

    if cooldown_sec > 0:
        if _VERBOSE_SAFE_EVICTION:
            print(f"[CaptionForge Cache] Cooldown {cooldown_sec:.2f}s ({reason})")
        time.sleep(float(cooldown_sec))
