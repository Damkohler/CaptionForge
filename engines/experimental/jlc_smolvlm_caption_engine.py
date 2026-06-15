#!/usr/bin/env python
"""
JLC SmolVLM Caption Engine
"""
from __future__ import annotations

MANIFEST = {
    "name": "JLC SmolVLM Caption Engine",
    "version": (0, 1, 0),
    "author": "J. L. Córdova",
    "description": "SmolVLM CaptionForge Pass A engine.",
}

import fnmatch
import json
import random
import re
import shutil
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable, Optional

from PIL import Image
import torch

from .captionforge_model_cache import (
    make_cache_key,
    get_cached_model,
    register_model,
    prepare_for_model_load,
    unload_after_run,
)

try:
    from .captionforge_caption_prompt_kit import (
        CAPTION_TYPE_CHOICES,
        CAPTION_LENGTH_CHOICES,
        EXTRA_OPTIONS as PROMPT_KIT_EXTRA_OPTIONS,
        build_caption_prompt_spec,
    )
except Exception:  # pragma: no cover - direct-file fallback
    CAPTION_TYPE_CHOICES = ["Descriptive", "LoRA Literal", "SFW Character Caption"]
    CAPTION_LENGTH_CHOICES = ["any", "very short", "short", "medium-length", "long", "very long"]
    PROMPT_KIT_EXTRA_OPTIONS = []

    def build_caption_prompt_spec(caption_type="Descriptive", caption_length="medium-length", extra_options=None, name_input="", dialect="smolvlm"):
        prompt = "Describe this image clearly and factually. Keep it SFW and mention only visible details."
        return prompt, {
            "caption_type": caption_type,
            "caption_length": str(caption_length),
            "extra_options": tuple(extra_options or ()),
            "name_input": name_input,
            "dialect": dialect,
            "source": "fallback_smol_prompt",
        }

SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"}
PROMPT_PRESETS: dict[str, str] = {
    "caption": "Describe this image clearly and factually. Keep it SFW.",
    "detailed": "Describe this image for dataset captioning. Mention visible subject, pose, expression, clothing, background, lighting, and style. Keep it SFW.",
    "lora_literal": "Write a concise SFW LoRA training caption for this image. Mention only visible details.",
    "sfw_character": "Write a SFW character-focused caption. Describe visible character design, pose, expression, hair, outfit, scene, lighting, and style.",
    "taggy": "Use concise comma-separated visual phrases only. Include visible subject, pose, clothing, style, lighting, and background.",
    "style_focus": "Describe the subject, appearance, pose, clothing, visual style, lighting, and background of this image.",
}
DEFAULT_PROMPT_PRESET = "detailed"
MEMORY_MODES = {"Default": {}, "Balanced (8-bit)": {"load_in_8bit": True}}

@dataclass
class GenerationConfig:
    max_new_tokens: int = 256
    num_beams: int = 3
    temperature: float = 0.0
    top_p: float = 0.90
    top_k: int = 50
    repetition_penalty: float = 1.0
    seed: Optional[int] = None

@dataclass
class CleanupConfig:
    trigger: str = ""
    prefix: str = ""
    suffix: str = ""
    forbidden_phrases: list[str] = field(default_factory=list)
    replacement_rules: list[tuple[str, str]] = field(default_factory=list)
    replace_case_insensitive: bool = True
    replace_whole_words_only: bool = False
    strip_boilerplate_prefixes: bool = True
    strip_trailing_period: bool = True

@dataclass
class BatchCaptionConfig:
    input_path: str = ""
    recursive: bool = True
    filename_glob: str = "*"
    extensions: set[str] = field(default_factory=lambda: set(SUPPORTED_EXTENSIONS))
    output_dir: str = ""
    write_txt: bool = True
    write_jsonl: bool = False
    jsonl_filename: str = "captions.jsonl"
    also_jsonl_path: str = ""
    write_run_config: bool = True
    run_config_filename: str = ""
    overwrite: bool = False
    backup_existing: bool = True
    dry_run: bool = False
    limit: int = 0
    skip_existing_txt: bool = True
    skip_existing_jsonl_images: bool = False

@dataclass
class CaptionRecord:
    image: str
    caption: str
    raw_caption: str
    model_name: str
    model_path: str
    prompt: str
    seed: Optional[int]
    temperature: float
    top_p: float
    top_k: int
    num_beams: int
    max_new_tokens: int
    max_size: int
    timestamp: str
    status: str = "ok"
    error: str = ""
    captionforge_pass: str = "A"
    model_family: str = "smolvlm"
    ensemble_run_index: int = 0
    image_key: str = ""
    prompt_metadata: dict[str, Any] = field(default_factory=dict)

@dataclass
class BatchCaptionResult:
    records: list[CaptionRecord] = field(default_factory=list)
    skipped: int = 0
    failed: int = 0


def timestamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def iso_timestamp() -> str:
    return datetime.now().isoformat(timespec="seconds")


def safe_mkdir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_prompt(prompt_preset: str = DEFAULT_PROMPT_PRESET, custom_prompt: str = "") -> str:
    custom = (custom_prompt or "").strip()
    return custom or PROMPT_PRESETS.get(prompt_preset, PROMPT_PRESETS[DEFAULT_PROMPT_PRESET])


def normalize_caption(text: str, strip_boilerplate_prefixes: bool = True, strip_trailing_period: bool = True) -> str:
    value = str(text or "").strip().replace("\n", " ")
    prefixes = [
        "The image shows ", "The image depicts ", "This image shows ", "This image depicts ",
        "A photo of ", "An image of ", "Assistant: ",
    ]
    if strip_boilerplate_prefixes:
        for prefix in prefixes:
            if value.lower().startswith(prefix.lower()):
                value = value[len(prefix):].strip()
                break
    value = re.sub(r"\s+", " ", value).strip()
    if strip_trailing_period and value.endswith("."):
        value = value[:-1].strip()
    return value


def apply_replacements(text: str, rules: list[tuple[str, str]], case_insensitive: bool = True, whole_words_only: bool = False) -> str:
    flags = re.IGNORECASE if case_insensitive else 0
    out = text
    for old, new in rules or []:
        old = old.strip()
        if not old:
            continue
        patt = re.escape(old)
        if whole_words_only:
            patt = r"\b" + patt + r"\b"
        out = re.sub(patt, new.strip(), out, flags=flags)
    return out


def cleanup_caption(caption: str, config: CleanupConfig) -> str:
    text = normalize_caption(caption, config.strip_boilerplate_prefixes, config.strip_trailing_period)
    text = apply_replacements(text, config.replacement_rules, config.replace_case_insensitive, config.replace_whole_words_only)
    for phrase in config.forbidden_phrases or []:
        phrase = phrase.strip()
        if phrase:
            text = re.sub(re.escape(phrase), "", text, flags=re.IGNORECASE)
    text = normalize_caption(text, config.strip_boilerplate_prefixes, config.strip_trailing_period)
    parts = []
    for x in (config.trigger, config.prefix):
        x = (x or "").strip().strip(" ,")
        if x:
            parts.append(x)
    if text:
        parts.append(text)
    final = ", ".join(parts).strip(" ,")
    if config.suffix:
        final += config.suffix
    return final.strip()


def iter_image_files(input_path: str | Path, recursive: bool = True, filename_glob: str = "*", extensions: Optional[set[str]] = None) -> Iterable[Path]:
    p = Path(input_path)
    exts = extensions or set(SUPPORTED_EXTENSIONS)
    glob_text = (filename_glob or "*").strip() or "*"
    if p.is_file():
        if p.suffix.lower() in exts and fnmatch.fnmatch(p.name, glob_text):
            yield p
        return
    if not p.exists():
        raise FileNotFoundError(f"input_path does not exist: {p}")
    it = p.rglob("*") if recursive else p.glob("*")
    for child in sorted(it):
        if child.is_file() and child.suffix.lower() in exts and fnmatch.fnmatch(child.name, glob_text):
            yield child


def load_image_file(path: str | Path) -> Image.Image:
    return Image.open(path).convert("RGB")


def resize_for_model(image: Image.Image, max_size: int) -> Image.Image:
    if max_size <= 0:
        return image
    w, h = image.size
    longest = max(w, h)
    if longest <= max_size:
        return image
    scale = max_size / float(longest)
    return image.resize((max(1, round(w * scale)), max(1, round(h * scale))), Image.Resampling.LANCZOS)


def sidecar_txt_path(image_path: Path, output_dir: str | Path | None = None) -> Path:
    if output_dir:
        out = Path(output_dir)
        safe_mkdir(out)
        return out / f"{image_path.stem}.txt"
    return image_path.with_suffix(".txt")


def backup_existing_file(path: Path, dry_run: bool = False) -> Optional[Path]:
    if not path.exists():
        return None
    backup = path.with_name(f"{path.name}.bak_{timestamp()}")
    if not dry_run:
        shutil.copy2(path, backup)
    return backup


def write_text_sidecar(path: Path, text: str, overwrite: bool = False, backup_existing: bool = True, dry_run: bool = False) -> bool:
    if path.exists() and not overwrite:
        return False
    if dry_run:
        return True
    safe_mkdir(path.parent)
    if path.exists() and overwrite and backup_existing:
        backup_existing_file(path, dry_run=False)
    path.write_text(text.rstrip() + "\n", encoding="utf-8")
    return True


def load_existing_jsonl_images(path: Path) -> set[str]:
    seen = set()
    if not path.exists():
        return seen
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            try:
                obj = json.loads(line)
            except Exception:
                continue
            image_value = obj.get("image") or obj.get("image_path") or obj.get("source")
            if image_value:
                seen.add(str(image_value))
    return seen


def record_to_json(record: CaptionRecord) -> dict[str, Any]:
    return asdict(record)


def append_jsonl_records(path: Path, records: list[CaptionRecord], dry_run: bool = False) -> None:
    if dry_run or not records:
        return
    safe_mkdir(path.parent)
    with path.open("a", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(record_to_json(rec), ensure_ascii=False) + "\n")


def json_safe(value):
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


def write_run_config_json(path: Path, config: dict[str, Any], dry_run: bool = False) -> None:
    if dry_run:
        return
    safe_mkdir(path.parent)
    path.write_text(json.dumps(json_safe(config), indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def model_folder_has_weights(local_path: Path) -> bool:
    if not local_path.exists() or not local_path.is_dir():
        return False
    patterns = ["*.safetensors", "*.bin", "*.pt", "*.pth"]
    return any(any(local_path.rglob(pat)) for pat in patterns)


def _try_get_comfy_model_management():
    try:
        import comfy.model_management as mm
        return mm
    except Exception:
        return None

@dataclass(frozen=True)
class SmolVLMModelInfo:
    repo_id: str
    local_folder: str
    notes: str = ""

MODEL_REGISTRY = {
    "SmolVLM-Instruct": SmolVLMModelInfo(
        repo_id="HuggingFaceTB/SmolVLM-Instruct",
        local_folder="SmolVLM-Instruct",
        notes="Default SmolVLM family entry.",
    ),
}

@dataclass
class SmolVLMCaptionConfig:
    model_name: str = "SmolVLM-Instruct"
    model_path: str = ""
    model_root: str = "models/LLM/JLC_SmolVLMCaption"
    memory_mode: str = "Default"
    dtype: str = "bf16"
    device: str = "auto"
    trust_remote_code: bool = True
    keep_loaded: bool = True
    cache_policy: str = ""
    quiet_transformers_load: bool = True
    max_size: int = 768

    # If prompt is non-empty, it wins and preserves legacy/custom behavior.
    # If prompt is empty, CaptionForge builds a generic Qwen/Smol prompt-kit prompt.
    prompt: str = ""
    caption_type: str = "SFW Character Caption"
    caption_length: str = "medium-length"
    extra_options: tuple[str, ...] = (
        "Keep the caption SFW and suitable for dataset training.",
        "Describe only visible details; do not infer identity, story, or intent.",
        "Use one dense natural-language caption rather than a tag list.",
    )
    name_input: str = ""
    allow_download: bool = True
    use_comfy_model_management: bool = True


def get_model_info(model_name: str) -> SmolVLMModelInfo:
    if model_name not in MODEL_REGISTRY:
        raise KeyError(f"Unknown SmolVLM model_name: {model_name}")
    return MODEL_REGISTRY[model_name]


def get_registry_model_path(model_name: str, model_root: str | Path) -> Path:
    return Path(model_root) / get_model_info(model_name).local_folder


def download_registry_model_if_needed(model_name: str, model_root: str | Path, metadata_only: bool = False, allow_download: bool = True) -> Path:
    local_path = get_registry_model_path(model_name, model_root)
    if not metadata_only and model_folder_has_weights(local_path):
        return local_path
    if not allow_download:
        if metadata_only:
            safe_mkdir(local_path)
            return local_path
        raise FileNotFoundError(f"Model folder does not contain weights and allow_download=False: {local_path}")
    from huggingface_hub import snapshot_download
    info = get_model_info(model_name)
    safe_mkdir(local_path)
    if metadata_only:
        snapshot_download(repo_id=info.repo_id, local_dir=str(local_path), ignore_patterns=["*.safetensors", "*.bin", "*.pt", "*.pth", "*.gguf", "*.onnx", "*.ckpt"])
    else:
        snapshot_download(repo_id=info.repo_id, local_dir=str(local_path))
    return local_path


def probe_registry_model_download(model_name: str, model_root: str | Path) -> str:
    local_path = download_registry_model_if_needed(model_name, model_root, metadata_only=True, allow_download=True)
    return f"JLC SmolVLM Caption download probe completed.\n\nModel: {model_name}\nFolder: {local_path}\nLarge model weight files were intentionally skipped."


def _unload_bundle(bundle: dict[str, Any]) -> None:
    model = bundle.get("model") if isinstance(bundle, dict) else None
    if model is not None:
        try:
            if _model_uses_accelerate_hooks(model):
                print("[JLC SmolVLM Engine] Accelerate-dispatched model detected; skipping model.to('cpu') during unload.")
            else:
                model.to("cpu")
        except Exception as exc:
            print(f"[JLC SmolVLM Engine] Non-fatal unload warning: {exc}")
    if isinstance(bundle, dict):
        bundle.clear()
    try:
        import gc
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass



def _format_bytes(num_bytes: int | None) -> str:
    if not num_bytes:
        return "unknown"
    value = float(num_bytes)
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if value < 1024.0 or unit == "TB":
            return f"{value:.2f} {unit}"
        value /= 1024.0
    return f"{num_bytes} B"


def _folder_weight_size_bytes(path: Path) -> int:
    total = 0
    for pattern in ("*.safetensors", "*.bin", "*.pt", "*.pth", "*.gguf", "*.ckpt"):
        try:
            total += sum(p.stat().st_size for p in path.rglob(pattern) if p.is_file())
        except Exception:
            pass
    return total


def _cuda_diagnostic_line() -> str:
    if not torch.cuda.is_available():
        return "CUDA unavailable"
    try:
        idx = torch.cuda.current_device()
        props = torch.cuda.get_device_properties(idx)
        allocated = torch.cuda.memory_allocated(idx)
        reserved = torch.cuda.memory_reserved(idx)
        total = getattr(props, "total_memory", 0)
        return (
            f"CUDA:{idx} {props.name} | "
            f"allocated={_format_bytes(allocated)}, reserved={_format_bytes(reserved)}, total={_format_bytes(total)}"
        )
    except Exception as exc:
        return f"CUDA diagnostics unavailable: {exc}"


def _model_uses_accelerate_hooks(model: Any) -> bool:
    if model is None:
        return False
    if getattr(model, "hf_device_map", None):
        return True
    if getattr(model, "_hf_hook", None) is not None:
        return True
    try:
        for module in model.modules():
            if getattr(module, "_hf_hook", None) is not None:
                return True
    except Exception:
        pass
    return False

class SmolVLMCaptionEngine:
    def __init__(self, config: SmolVLMCaptionConfig, generation: Optional[GenerationConfig] = None, cleanup: Optional[CleanupConfig] = None) -> None:
        self.config = config
        self.generation = generation or GenerationConfig()
        self.cleanup = cleanup or CleanupConfig()
        self.processor = None
        self.model = None
        self.local_model_path: Optional[Path] = None
        self.model_size_bytes: Optional[int] = None
        self._last_prompt_metadata: dict[str, Any] = {}
        self._comfy_mm = _try_get_comfy_model_management() if config.use_comfy_model_management else None
        self.inference_device = self._resolve_inference_device(config.device)

    def resolve_caption_prompt(self) -> tuple[str, dict[str, Any]]:
        custom = (self.config.prompt or "").replace("\\n", "\n").strip()
        if custom:
            metadata = {
                "source": "custom_or_legacy_engine_prompt",
                "dialect": "smolvlm",
                "caption_type": getattr(self.config, "caption_type", "custom"),
                "caption_length": getattr(self.config, "caption_length", "custom"),
                "extra_options": list(getattr(self.config, "extra_options", ())),
                "name_input": getattr(self.config, "name_input", ""),
            }
            return custom, metadata

        prompt, spec = build_caption_prompt_spec(
            caption_type=getattr(self.config, "caption_type", "SFW Character Caption"),
            caption_length=getattr(self.config, "caption_length", "medium-length"),
            extra_options=getattr(self.config, "extra_options", ()),
            name_input=getattr(self.config, "name_input", ""),
            dialect="smolvlm",
        )
        metadata = spec.to_metadata() if hasattr(spec, "to_metadata") else dict(spec)
        return prompt, metadata

    def _resolve_inference_device(self, device: str) -> torch.device:
        if self._comfy_mm is not None:
            try:
                dev = self._comfy_mm.get_torch_device()
                if dev.type == "cuda":
                    return dev
            except Exception:
                pass
        if device == "auto":
            if not torch.cuda.is_available():
                raise RuntimeError("CaptionForge SmolVLM requires CUDA for release builds.")
            return torch.device("cuda")
        dev = torch.device(device)
        if dev.type != "cuda":
            raise RuntimeError(f"CaptionForge SmolVLM release mode does not allow inference device={device!r}")
        return dev

    def _resolve_dtype(self, dtype: str):
        text = dtype.lower().strip()
        if text in {"bf16", "bfloat16"}:
            return torch.bfloat16
        if text in {"fp16", "float16"}:
            return torch.float16
        if text in {"fp32", "float32"}:
            return torch.float32
        return torch.bfloat16

    def _cache_policy(self) -> str:
        policy = (self.config.cache_policy or "").strip()
        return policy or ("evict_other_caption_models" if self.config.keep_loaded else "unload_after_run")

    def _cache_key(self, local_path: Path) -> str:
        return make_cache_key(role="caption", family="smolvlm", model_path=str(local_path.resolve()), device=str(self.inference_device), quantization=self.config.memory_mode, dtype=self.config.dtype)

    def resolve_model_path(self) -> Path:
        if self.config.model_path.strip():
            return Path(self.config.model_path).expanduser()
        return download_registry_model_if_needed(self.config.model_name, self.config.model_root, metadata_only=False, allow_download=self.config.allow_download)

    def _module_size(self, model) -> int:
        total = 0
        try:
            for p in model.parameters():
                total += p.numel() * p.element_size()
            for b in model.buffers():
                total += b.numel() * b.element_size()
        except Exception:
            pass
        return total

    def load(self) -> None:
        local_path = self.resolve_model_path()
        self.local_model_path = local_path
        cache_key = self._cache_key(local_path)
        cached = get_cached_model(cache_key)
        if cached is not None:
            self.processor = cached.get("processor")
            self.model = cached.get("model")
            self.model_size_bytes = cached.get("model_size_bytes")
            print(f"[JLC SmolVLM Engine] Reusing cached model: {local_path}")
            return
        prepare_for_model_load(cache_key, policy=self._cache_policy(), role="caption")
        if self.generation.seed is not None:
            set_seed(self.generation.seed)
        print(f"[JLC SmolVLM Engine] Local model path: {local_path}")
        print(f"[JLC SmolVLM Engine] Local weight size: {_format_bytes(_folder_weight_size_bytes(local_path))}")
        print(f"[JLC SmolVLM Engine] { _cuda_diagnostic_line() }")
        from transformers import AutoProcessor
        try:
            from transformers import AutoModelForImageTextToText as SmolModel
        except Exception:
            try:
                from transformers import AutoModelForVision2Seq as SmolModel
            except Exception:
                from transformers import AutoModelForCausalLM as SmolModel
        print(f"[JLC SmolVLM Engine] Loading processor: {local_path}")
        self.processor = AutoProcessor.from_pretrained(str(local_path), trust_remote_code=self.config.trust_remote_code)
        kwargs: dict[str, Any] = {"torch_dtype": self._resolve_dtype(self.config.dtype), "trust_remote_code": self.config.trust_remote_code}
        if self.config.memory_mode == "Balanced (8-bit)":
            from transformers import BitsAndBytesConfig
            kwargs["quantization_config"] = BitsAndBytesConfig(load_in_8bit=True)
            kwargs["device_map"] = {"": self.inference_device.index or 0}
        print(f"[JLC SmolVLM Engine] Loading model: {local_path}")
        print(f"[JLC SmolVLM Engine] dtype={kwargs.get('torch_dtype')}, device={self.inference_device}, memory_mode={self.config.memory_mode}")
        self.model = SmolModel.from_pretrained(str(local_path), **kwargs)
        device_map = getattr(self.model, "hf_device_map", None)
        if device_map:
            print(f"[JLC SmolVLM Engine] hf_device_map: {device_map}")
        else:
            try:
                print(f"[JLC SmolVLM Engine] first parameter device: {next(self.model.parameters()).device}")
            except Exception:
                pass
        self.model.eval()
        if self.config.memory_mode == "Default":
            self.model.to(self.inference_device)
        self.model_size_bytes = self._module_size(self.model)
        print(f"[JLC SmolVLM Engine] Estimated loaded module size: {_format_bytes(self.model_size_bytes)}")
        register_model(cache_key, {"processor": self.processor, "model": self.model, "model_size_bytes": self.model_size_bytes}, family="smolvlm", model_path=str(local_path), device=str(self.inference_device), quantization=self.config.memory_mode, role="caption", unload_fn=_unload_bundle)
        print("[JLC SmolVLM Engine] Model loaded.")

    def unload(self) -> None:
        if self.local_model_path is not None:
            unload_after_run(self._cache_key(self.local_model_path), enabled=True)
        self.processor = None
        self.model = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    @torch.inference_mode()
    def caption_pil(self, image: Image.Image) -> tuple[str, str]:
        if self.processor is None or self.model is None:
            self.load()
        if self.generation.seed is not None:
            set_seed(self.generation.seed)
        image = resize_for_model(image.convert("RGB"), self.config.max_size)
        prompt, prompt_metadata = self.resolve_caption_prompt()
        self._last_prompt_metadata = dict(prompt_metadata)
        messages = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": prompt}]}]
        if hasattr(self.processor, "apply_chat_template"):
            text = self.processor.apply_chat_template(messages, add_generation_prompt=True)
            inputs = self.processor(text=text, images=[image], return_tensors="pt")
        else:
            inputs = self.processor(text=prompt, images=image, return_tensors="pt")
        inputs = {k: (v.to(self.inference_device) if hasattr(v, "to") else v) for k, v in inputs.items()}
        if "pixel_values" in inputs and hasattr(self.model, "dtype") and torch.is_floating_point(inputs["pixel_values"]):
            inputs["pixel_values"] = inputs["pixel_values"].to(dtype=self.model.dtype)
        gen_kwargs = {"max_new_tokens": int(self.generation.max_new_tokens), "num_beams": max(1, int(self.generation.num_beams)), "do_sample": bool(self.generation.temperature > 0), "use_cache": True}
        if self.generation.temperature > 0:
            gen_kwargs.update({"temperature": float(self.generation.temperature), "top_p": float(self.generation.top_p), "top_k": int(self.generation.top_k)})
        if self.generation.repetition_penalty != 1.0:
            gen_kwargs["repetition_penalty"] = float(self.generation.repetition_penalty)
        generated = self.model.generate(**inputs, **gen_kwargs)
        input_len = inputs["input_ids"].shape[-1] if isinstance(inputs.get("input_ids"), torch.Tensor) else 0
        trimmed = generated[:, input_len:] if input_len and generated.shape[-1] > input_len else generated
        raw = self.processor.batch_decode(trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0].strip()
        final = cleanup_caption(raw, self.cleanup)
        if self._cache_policy() == "unload_after_run":
            self.unload()
        return final, raw

    def caption_path(self, image_path: str | Path) -> CaptionRecord:
        p = Path(image_path)
        final, raw = self.caption_pil(load_image_file(p))
        return CaptionRecord(image=str(p), caption=final, raw_caption=raw, model_name=self.config.model_name, model_path=str(self.local_model_path or self.config.model_path), prompt=self.resolve_caption_prompt()[0], seed=self.generation.seed, temperature=self.generation.temperature, top_p=self.generation.top_p, top_k=self.generation.top_k, num_beams=self.generation.num_beams, max_new_tokens=self.generation.max_new_tokens, max_size=self.config.max_size, timestamp=iso_timestamp(), image_key=str(p.resolve()), prompt_metadata=dict(self._last_prompt_metadata))

    def caption_batch(self, batch: BatchCaptionConfig) -> BatchCaptionResult:
        if not batch.input_path.strip():
            raise ValueError("BatchCaptionConfig.input_path is required.")
        result = BatchCaptionResult()
        images = list(iter_image_files(batch.input_path, batch.recursive, batch.filename_glob, batch.extensions))
        if batch.limit > 0:
            images = images[: int(batch.limit)]
        if not images:
            return result
        output_dir = Path(batch.output_dir) if batch.output_dir.strip() else None
        jsonl_path = (output_dir or images[0].parent) / (batch.jsonl_filename.strip() or "captions.jsonl") if batch.write_jsonl else None
        also_jsonl_path = Path(batch.also_jsonl_path) if batch.also_jsonl_path.strip() else None
        seen_jsonl = set()
        if batch.skip_existing_jsonl_images:
            if jsonl_path is not None:
                seen_jsonl |= load_existing_jsonl_images(jsonl_path)
            if also_jsonl_path is not None:
                seen_jsonl |= load_existing_jsonl_images(also_jsonl_path)
        if batch.write_run_config:
            outdir = output_dir or images[0].parent
            name = batch.run_config_filename.strip() or f"jlc_smolvlm_caption_run_config_{timestamp()}.json"
            write_run_config_json(outdir / name, self.build_run_config(batch), dry_run=batch.dry_run)
        for image_path in images:
            try:
                txt_path = sidecar_txt_path(image_path, output_dir)
                if batch.skip_existing_txt and batch.write_txt and txt_path.exists() and not batch.overwrite:
                    result.skipped += 1
                    continue
                if batch.skip_existing_jsonl_images and str(image_path) in seen_jsonl:
                    result.skipped += 1
                    continue
                rec = self.caption_path(image_path)
                result.records.append(rec)
                if batch.write_txt:
                    write_text_sidecar(txt_path, rec.caption, batch.overwrite, batch.backup_existing, batch.dry_run)
                if jsonl_path is not None:
                    append_jsonl_records(jsonl_path, [rec], batch.dry_run)
                if also_jsonl_path is not None:
                    append_jsonl_records(also_jsonl_path, [rec], batch.dry_run)
            except Exception as exc:
                result.failed += 1
                err = CaptionRecord(image=str(image_path), caption="", raw_caption="", model_name=self.config.model_name, model_path=str(self.local_model_path or self.config.model_path), prompt=self.resolve_caption_prompt()[0], seed=self.generation.seed, temperature=self.generation.temperature, top_p=self.generation.top_p, top_k=self.generation.top_k, num_beams=self.generation.num_beams, max_new_tokens=self.generation.max_new_tokens, max_size=self.config.max_size, timestamp=iso_timestamp(), status="error", error=str(exc), image_key=str(Path(image_path).resolve()), prompt_metadata=self.resolve_caption_prompt()[1])
                result.records.append(err)
                if jsonl_path is not None:
                    append_jsonl_records(jsonl_path, [err], batch.dry_run)
                if also_jsonl_path is not None:
                    append_jsonl_records(also_jsonl_path, [err], batch.dry_run)
        return result

    def build_run_config(self, batch: Optional[BatchCaptionConfig] = None) -> dict[str, Any]:
        cleanup = asdict(self.cleanup)
        cleanup["replacement_rules"] = [list(x) for x in self.cleanup.replacement_rules]
        return {"timestamp": iso_timestamp(), "engine": "JLC SmolVLM Caption Engine", "smolvlm_config": asdict(self.config), "prompt_metadata": self.resolve_caption_prompt()[1], "generation": asdict(self.generation), "cleanup": cleanup, "batch": json_safe(asdict(batch)) if batch is not None else None}
