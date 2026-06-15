"""
JLC CaptionForge — Experimental Reversed Capstone Node

Pipeline:
    A_RAW_CAPTIONS
      -> B_VLM_CLEANED       (image-aware statement cleaner, one cleaned caption per raw caption)
      -> C_RECONCILED        (text-only reconciler/copywriter, one coherent caption per image)
      -> D_FINAL_EXPORT      (TXT/JSONL sidecars)

This node is intentionally experimental. It keeps the VLM task narrow and lets the
text LLM write last.
"""

from __future__ import annotations

MANIFEST = {
    "name": "JLC CaptionForge Reversed Node",
    "version": (0, 1, 0),
    "author": "J. L. Córdova",
    "description": "Experimental CaptionForge capstone: VLM cleans each raw caption first, then text LLM reconciles cleaned captions into final coherent captions.",
}

import json
import random
import re
from datetime import datetime
from pathlib import Path, PureWindowsPath
from typing import Any

try:
    import folder_paths
except Exception:  # pragma: no cover
    folder_paths = None

try:
    from .captionforge_ollama_model_dropdowns import load_ollama_model_dropdowns
except Exception:  # pragma: no cover
    load_ollama_model_dropdowns = None

from ..engines.captionforge_vlm_validator_engine import (
    BatchVLMValidatorConfig,
    DEFAULT_VLM_CLEANER_PROMPT,
    VLMValidatorConfig,
    extract_clean_batch,
)
from ..engines.captionforge_distiller_engine import (
    BatchConfig as DistillerBatchConfig,
    DEFAULT_RECONCILER_INSTRUCTIONS,
    DistillerConfig,
    process_reconciler_batch,
)

CAPTIONFORGE_REVERSED_NODE_VERSION = "0.1.0"
MAX_SEED_32 = 0xFFFFFFFF
SEED_MODES = ["fixed", "increment", "decrement", "random"]
FINAL_CAPTION_STYLES = ["narrative", "comma", "both"]


def _load_model_choices() -> tuple[list[str], list[str], str, str]:
    fallback_distiller = ["llama3.1:8b", "hermes3:8b", "gpt-oss:20b", "custom"]
    fallback_validator = ["gemma4:e4b", "qwen2.5vl:7b", "minicpm-v", "custom"]
    if load_ollama_model_dropdowns is None:
        return fallback_distiller, fallback_validator, fallback_distiller[0], fallback_validator[0]
    try:
        data = load_ollama_model_dropdowns(__file__)
        dist = list(data.get("distiller_models") or fallback_distiller)
        val = list(data.get("validator_models") or fallback_validator)
        if "custom" not in dist:
            dist.append("custom")
        if "custom" not in val:
            val.append("custom")
        return dist, val, str(data.get("distiller_default") or dist[0]), str(data.get("validator_default") or val[0])
    except Exception:
        return fallback_distiller, fallback_validator, fallback_distiller[0], fallback_validator[0]


DISTILLER_MODEL_CHOICES, VALIDATOR_MODEL_CHOICES, DEFAULT_DISTILLER_MODEL, DEFAULT_VALIDATOR_MODEL = _load_model_choices()


def _default_output_dir() -> str:
    if folder_paths is not None:
        try:
            return str(Path(folder_paths.get_output_directory()) / "CaptionForge")
        except Exception:
            pass
    return str(Path.cwd() / "output" / "CaptionForge")


def _clean_run_name(value: Any) -> str:
    text = str(value or "captionforge_run").strip()
    text = re.sub(r"[^A-Za-z0-9_.-]+", "_", text).strip("._")
    return text or "captionforge_run"


def _normalize_text(value: Any) -> str:
    return re.sub(r"\s+", " ", str(value or "").replace("\n", " ")).strip()


def _safe_bool(value: Any, default: bool = False) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return default
    text = str(value).strip().lower()
    if text in {"1", "true", "yes", "y", "on"}:
        return True
    if text in {"0", "false", "no", "n", "off"}:
        return False
    return default


def _coerce_int(value: Any, default: int, min_value: int | None = None, max_value: int | None = None) -> int:
    try:
        out = int(value)
    except Exception:
        out = int(default)
    if min_value is not None:
        out = max(min_value, out)
    if max_value is not None:
        out = min(max_value, out)
    return out


def _coerce_float(value: Any, default: float, min_value: float | None = None, max_value: float | None = None) -> float:
    try:
        out = float(value)
    except Exception:
        out = float(default)
    if min_value is not None:
        out = max(min_value, out)
    if max_value is not None:
        out = min(max_value, out)
    return out


def _seed_for_stage(base_seed: Any, seed_mode: Any, stage_index: int) -> int:
    base = _coerce_int(base_seed, -1, -1, MAX_SEED_32)
    mode = str(seed_mode or "fixed").strip().lower()
    if base < 0:
        return -1
    if mode == "increment":
        return min(MAX_SEED_32, base + stage_index)
    if mode == "decrement":
        return max(0, base - stage_index)
    if mode == "random":
        rng = random.Random(base)
        out = base
        for _ in range(stage_index + 1):
            out = rng.randint(0, MAX_SEED_32)
        return out
    return base


def _resolve_ollama_model_name(value: Any, custom_value: Any, fallback: str) -> str:
    text = str(value or "").strip()
    custom = str(custom_value or "").strip()
    if text.lower() == "custom":
        return custom or fallback
    return text or custom or fallback


def _load_prompt_file_if_path(text: str) -> str:
    stripped = str(text or "").strip()
    if not stripped:
        return ""
    try:
        p = Path(stripped)
        if p.exists() and p.is_file():
            return p.read_text(encoding="utf-8")
    except Exception:
        pass
    return str(text or "")


def _derive_paths(output_dir: Path, run_name: str) -> dict[str, str]:
    return {
        "output_dir": str(output_dir),
        "run_name": run_name,
        "pass_a_jsonl": str(output_dir / f"{run_name}__A_RAW_CAPTIONS.jsonl"),
        "cleaned_jsonl": str(output_dir / f"{run_name}__B_VLM_CLEANED.jsonl"),
        "cleaned_prompt_jsonl": str(output_dir / f"{run_name}__B_VLM_CLEANER_prompts.jsonl"),
        "cleaned_readable_dir": str(output_dir / f"{run_name}__B_VLM_CLEANED_readable"),
        "reconciled_jsonl": str(output_dir / f"{run_name}__C_RECONCILED.jsonl"),
        "reconciled_readable_jsonl": str(output_dir / f"{run_name}__C_RECONCILED_readable.jsonl"),
        "reconciled_readable_json": str(output_dir / f"{run_name}__C_RECONCILED_readable.json"),
        "reconciled_prompt_jsonl": str(output_dir / f"{run_name}__C_RECONCILER_prompts.jsonl"),
        "final_jsonl": str(output_dir / f"{run_name}__D_FINAL_EXPORT.jsonl"),
        "final_txt_dir": str(output_dir / f"{run_name}__TXT"),
        "output_paths_json": str(output_dir / f"{run_name}__output_paths.json"),
    }


def _basename_cross_platform(value: Any) -> str:
    text = str(value or "").strip()
    if not text:
        return ""
    return PureWindowsPath(text).name if "\\" in text else Path(text).name


def _safe_txt_stem_for_source_filename(stem: Any) -> str:
    text = str(stem or "").strip()
    if not text:
        return "image"
    text = re.sub(r'[<>:"/\\|?*\x00-\x1f]+', "_", text)
    text = text.rstrip(" .")
    return text or "image"


def _txt_stem_from_record(record: dict[str, Any]) -> str:
    for candidate in (
        record.get("image_resolved_path"),
        record.get("image"),
        record.get("image_key"),
    ):
        base = _basename_cross_platform(candidate)
        if base:
            return _safe_txt_stem_for_source_filename(Path(base).stem or base)
    return "image"


def _select_caption(record: dict[str, Any], style: str) -> str:
    narrative = _normalize_text(record.get("validated_caption_narrative"))
    comma = _normalize_text(record.get("validated_caption_comma"))
    style = str(style or "narrative").strip().lower()
    if style == "comma":
        return comma or narrative
    if style == "both":
        if narrative and comma and narrative != comma:
            return f"{narrative}\n{comma}"
        return narrative or comma
    return narrative or comma


def _write_final_outputs(reconciled_jsonl: Path, final_jsonl: Path, final_txt_dir: Path, caption_style: str, write_txt: bool, write_jsonl: bool) -> tuple[list[dict[str, Any]], str, int, int]:
    records: list[dict[str, Any]] = []
    if reconciled_jsonl.exists():
        for line in reconciled_jsonl.read_text(encoding="utf-8").splitlines():
            if line.strip():
                records.append(json.loads(line))
    final_records: list[dict[str, Any]] = []
    captions: list[str] = []
    ok = failed = 0
    if write_jsonl:
        final_jsonl.parent.mkdir(parents=True, exist_ok=True)
        final_jsonl.write_text("", encoding="utf-8")
    if write_txt:
        final_txt_dir.mkdir(parents=True, exist_ok=True)
    for rec in records:
        caption = _select_caption(rec, caption_style)
        is_ok = str(rec.get("status") or "").lower() == "ok" and bool(caption)
        ok += int(is_ok)
        failed += int(not is_ok)
        out = {
            "captionforge_pass": "D_FINAL_EXPORT",
            "engine": "jlc_captionforge_reversed_node",
            "engine_version": CAPTIONFORGE_REVERSED_NODE_VERSION,
            "image_key": rec.get("image_key", ""),
            "image": rec.get("image", ""),
            "status": "ok" if is_ok else "error",
            "caption_style": caption_style,
            "final_caption": caption if is_ok else "",
            "source_reconciler": rec,
            "timestamp": datetime.now().isoformat(timespec="seconds"),
        }
        final_records.append(out)
        if is_ok:
            captions.append(caption)
            if write_txt:
                (final_txt_dir / f"{_txt_stem_from_record(rec)}.txt").write_text(caption.rstrip() + "\n", encoding="utf-8")
        if write_jsonl:
            with final_jsonl.open("a", encoding="utf-8", newline="\n") as f:
                f.write(json.dumps(out, ensure_ascii=False) + "\n")
    return final_records, "\n\n".join(captions), ok, failed


class JLC_CaptionForge_Reversed:
    """Experimental reversed CaptionForge capstone node."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "Input - captions JSONL": ("STRING", {"default": "", "multiline": False}),
                "Input - image path": ("STRING", {"default": "", "multiline": False}),
                "Output - folder": ("STRING", {"default": "", "multiline": False}),
                "Output - run name": ("STRING", {"default": "captionforge_run", "multiline": False}),
                "Output - overwrite outputs": ("BOOLEAN", {"default": True}),
                "LoRA - trigger word": ("STRING", {"default": "", "multiline": False}),
                "LoRA - user caption anchor": ("STRING", {"default": "", "multiline": False}),

                "VLM Cleaner - enabled": ("BOOLEAN", {"default": True}),
                "VLM Cleaner - model": (VALIDATOR_MODEL_CHOICES, {"default": DEFAULT_VALIDATOR_MODEL}),
                "VLM Cleaner - custom Ollama model": ("STRING", {"default": "", "multiline": False}),
                "VLM Cleaner - base seed": ("INT", {"default": -1, "min": -1, "max": MAX_SEED_32, "step": 1}),
                "VLM Cleaner - seed mode": (SEED_MODES, {"default": "fixed"}),
                "VLM Cleaner - prompt": ("STRING", {"default": DEFAULT_VLM_CLEANER_PROMPT, "multiline": True}),
                "VLM Cleaner - num predict": ("INT", {"default": 1800, "min": 64, "max": 12000, "step": 64}),
                "VLM Cleaner - temperature": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 2.0, "step": 0.01}),
                "VLM Cleaner - top p": ("FLOAT", {"default": 0.92, "min": 0.0, "max": 1.0, "step": 0.01}),
                "VLM Cleaner - top k": ("INT", {"default": 80, "min": 0, "max": 500, "step": 1}),
                "VLM Cleaner - write prompt JSONL": ("BOOLEAN", {"default": False}),
                "VLM Cleaner - preserve raw response": ("BOOLEAN", {"default": False}),

                "Reconciler - enabled": ("BOOLEAN", {"default": True}),
                "Reconciler - model": (DISTILLER_MODEL_CHOICES, {"default": DEFAULT_DISTILLER_MODEL}),
                "Reconciler - custom Ollama model": ("STRING", {"default": "", "multiline": False}),
                "Reconciler - base seed": ("INT", {"default": -1, "min": -1, "max": MAX_SEED_32, "step": 1}),
                "Reconciler - seed mode": (SEED_MODES, {"default": "fixed"}),
                "Reconciler - prompt": ("STRING", {"default": DEFAULT_RECONCILER_INSTRUCTIONS, "multiline": True}),
                "Reconciler - max caption chars for LLM": ("INT", {"default": 2400, "min": 0, "max": 12000, "step": 64}),
                "Reconciler - num predict": ("INT", {"default": 3200, "min": 64, "max": 12000, "step": 64}),
                "Reconciler - temperature": ("FLOAT", {"default": 0.18, "min": 0.0, "max": 2.0, "step": 0.01}),
                "Reconciler - top p": ("FLOAT", {"default": 0.90, "min": 0.0, "max": 1.0, "step": 0.01}),
                "Reconciler - top k": ("INT", {"default": 60, "min": 0, "max": 500, "step": 1}),
                "Reconciler - write prompt JSONL": ("BOOLEAN", {"default": False}),
                "Reconciler - preserve raw response": ("BOOLEAN", {"default": False}),

                "Final - caption style": (FINAL_CAPTION_STYLES, {"default": "narrative"}),
                "Final - write TXT sidecars": ("BOOLEAN", {"default": True}),
                "Final - write JSONL": ("BOOLEAN", {"default": True}),
            }
        }

    RETURN_TYPES = ("STRING", "STRING", "STRING", "STRING")
    RETURN_NAMES = ("final_captions", "final_jsonl_records", "output_paths_json", "status")
    FUNCTION = "forge"
    CATEGORY = "JLC/Captioning/CaptionForge"

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        return float("NaN")

    def forge(self, **kwargs):
        output_dir = Path(str(kwargs.get("Output - folder") or "").strip() or _default_output_dir())
        output_dir.mkdir(parents=True, exist_ok=True)
        run_name = _clean_run_name(kwargs.get("Output - run name") or "captionforge_run")
        paths = _derive_paths(output_dir, run_name)

        caption_jsonl = str(kwargs.get("Input - captions JSONL") or "").strip() or paths["pass_a_jsonl"]
        image_root = str(kwargs.get("Input - image path") or "").strip() or str(Path(caption_jsonl).parent)
        overwrite = _safe_bool(kwargs.get("Output - overwrite outputs"), True)
        trigger_word = _normalize_text(kwargs.get("LoRA - trigger word"))
        user_caption_anchor = _normalize_text(kwargs.get("LoRA - user caption anchor"))

        if not Path(caption_jsonl).exists():
            raise FileNotFoundError(f"Caption JSONL not found: {caption_jsonl}")

        cleaner_enabled = _safe_bool(kwargs.get("VLM Cleaner - enabled"), True)
        reconciler_enabled = _safe_bool(kwargs.get("Reconciler - enabled"), True)
        if not cleaner_enabled and not reconciler_enabled:
            raise RuntimeError("CaptionForge Reversed has nothing to do: both Cleaner and Reconciler are disabled.")

        cleaner_model = _resolve_ollama_model_name(
            kwargs.get("VLM Cleaner - model"),
            kwargs.get("VLM Cleaner - custom Ollama model"),
            DEFAULT_VALIDATOR_MODEL,
        )
        reconciler_model = _resolve_ollama_model_name(
            kwargs.get("Reconciler - model"),
            kwargs.get("Reconciler - custom Ollama model"),
            DEFAULT_DISTILLER_MODEL,
        )

        cleaner_result_info: Any = "skipped"
        if cleaner_enabled:
            cleaner_prompt = _load_prompt_file_if_path(str(kwargs.get("VLM Cleaner - prompt") or "")) or DEFAULT_VLM_CLEANER_PROMPT
            cleaner_config = VLMValidatorConfig(
                vlm_backend="ollama",
                vlm_model=cleaner_model,
                vlm_validator_prompt=cleaner_prompt,
                trigger_word=trigger_word,
                user_caption_anchor=user_caption_anchor,
                preserve_raw_vlm_response=_safe_bool(kwargs.get("VLM Cleaner - preserve raw response"), False),
                ollama_num_predict=_coerce_int(kwargs.get("VLM Cleaner - num predict"), 1800, 64, 12000),
                ollama_temperature=_coerce_float(kwargs.get("VLM Cleaner - temperature"), 0.0, 0.0, 2.0),
                ollama_top_p=_coerce_float(kwargs.get("VLM Cleaner - top p"), 0.92, 0.0, 1.0),
                ollama_top_k=_coerce_int(kwargs.get("VLM Cleaner - top k"), 80, 0, 500),
                ollama_format_mode="json",
            )
            cleaner_config.ollama_seed = _seed_for_stage(kwargs.get("VLM Cleaner - base seed"), kwargs.get("VLM Cleaner - seed mode"), 0)
            cleaner_batch = BatchVLMValidatorConfig(
                input_jsonl=caption_jsonl,
                image_root=image_root,
                output_jsonl=paths["cleaned_jsonl"],
                readable_sidecar_dir=paths["cleaned_readable_dir"],
                write_jsonl=True,
                write_readable_sidecars=True,
                dry_run=False,
                overwrite=overwrite,
                write_prompt_jsonl=_safe_bool(kwargs.get("VLM Cleaner - write prompt JSONL"), False),
                prompt_jsonl=paths["cleaned_prompt_jsonl"],
            )
            cleaner_result = extract_clean_batch(cleaner_batch, cleaner_config)
            cleaner_result_info = f"processed={cleaner_result.processed} failed={cleaner_result.failed}"
        else:
            if not Path(paths["cleaned_jsonl"]).exists():
                raise FileNotFoundError(f"Cleaner disabled but cleaned JSONL does not exist: {paths['cleaned_jsonl']}")

        reconciler_rc: Any = "skipped"
        if reconciler_enabled:
            reconciler_prompt = _load_prompt_file_if_path(str(kwargs.get("Reconciler - prompt") or "")) or DEFAULT_RECONCILER_INSTRUCTIONS
            reconciler_config = DistillerConfig(
                llm_backend="ollama",
                llm_model=reconciler_model,
                instructions=reconciler_prompt,
                strategy="cleaned_caption_reconciler",
                max_caption_chars_for_llm=_coerce_int(kwargs.get("Reconciler - max caption chars for LLM"), 2400, 0, 12000),
                ollama_num_predict=_coerce_int(kwargs.get("Reconciler - num predict"), 3200, 64, 12000),
                ollama_temperature=_coerce_float(kwargs.get("Reconciler - temperature"), 0.18, 0.0, 2.0),
                ollama_top_p=_coerce_float(kwargs.get("Reconciler - top p"), 0.90, 0.0, 1.0),
                ollama_top_k=_coerce_int(kwargs.get("Reconciler - top k"), 60, 0, 500),
                preserve_raw_response=_safe_bool(kwargs.get("Reconciler - preserve raw response"), False),
                trigger_word=trigger_word,
                user_caption_anchor=user_caption_anchor,
            )
            reconciler_config.ollama_seed = _seed_for_stage(kwargs.get("Reconciler - base seed"), kwargs.get("Reconciler - seed mode"), 1)
            reconciler_batch = DistillerBatchConfig(
                input_jsonl=paths["cleaned_jsonl"],
                output_jsonl=paths["reconciled_jsonl"],
                readable_jsonl=paths["reconciled_readable_jsonl"],
                readable_json=paths["reconciled_readable_json"],
                prompt_jsonl=paths["reconciled_prompt_jsonl"],
                dry_run=False,
                append_output=not overwrite,
                skip_existing=False,
                no_readable_sidecars=False,
                write_prompt_jsonl=_safe_bool(kwargs.get("Reconciler - write prompt JSONL"), False),
            )
            reconciler_rc = process_reconciler_batch(reconciler_batch, reconciler_config)
        else:
            if not Path(paths["reconciled_jsonl"]).exists():
                raise FileNotFoundError(f"Reconciler disabled but reconciled JSONL does not exist: {paths['reconciled_jsonl']}")

        caption_style = str(kwargs.get("Final - caption style") or "narrative")
        final_records, final_captions, final_ok, final_failed = _write_final_outputs(
            Path(paths["reconciled_jsonl"]),
            Path(paths["final_jsonl"]),
            Path(paths["final_txt_dir"]),
            caption_style,
            _safe_bool(kwargs.get("Final - write TXT sidecars"), True),
            _safe_bool(kwargs.get("Final - write JSONL"), True),
        )

        output_paths = dict(paths)
        output_paths.update({
            "caption_jsonl": caption_jsonl,
            "image_root": image_root,
            "cleaner_model_resolved": cleaner_model,
            "reconciler_model_resolved": reconciler_model,
        })
        output_paths_json = json.dumps(output_paths, ensure_ascii=False, indent=2)
        Path(paths["output_paths_json"]).write_text(output_paths_json + "\n", encoding="utf-8")

        final_jsonl_records = "\n".join(json.dumps(r, ensure_ascii=False) for r in final_records)
        status = (
            f"[JLC CaptionForge Reversed v{CAPTIONFORGE_REVERSED_NODE_VERSION}] complete | "
            f"cleaner={cleaner_enabled} {cleaner_result_info} | "
            f"reconciler={reconciler_enabled} rc={reconciler_rc} | "
            f"final_ok={final_ok} final_failed={final_failed} | run={run_name} output={output_dir}"
        )
        print(status, flush=True)
        return (final_captions, final_jsonl_records, output_paths_json, status)


NODE_CLASS_MAPPINGS = {
    "JLC_CaptionForge_Reversed": JLC_CaptionForge_Reversed,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "JLC_CaptionForge_Reversed": "\u2003JLC CaptionForge (Reversed Experimental)",
}
