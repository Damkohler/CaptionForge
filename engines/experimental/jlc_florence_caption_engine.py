#!/usr/bin/env python
"""
JLC Florence Caption Engine

CaptionForge Pass A engine for Microsoft Florence-2-family image captioning.

Drop target:
    CaptionForge/engines/jlc_florence_caption_engine.py

Notes:
    - Python/Transformers-native; no Ollama.
    - No bitsandbytes in v1. Florence is intended as a light witness model.
    - Uses CaptionForge global model cache just like Joy/Qwen.
    - Supports direct caption tasks:
        <CAPTION>
        <DETAILED_CAPTION>
        <MORE_DETAILED_CAPTION>
    - Leaves room for future Florence task expansion, but Pass A defaults to
      caption text, not boxes/regions/OCR.
"""

from __future__ import annotations

MANIFEST = {
    "name": "JLC Florence Caption Engine",
    "version": (0, 1, 9),
    "author": "J. L. Córdova",
    "description": (
        "Shared Florence-2 CaptionForge Pass A engine for local image captioning, "
        "folder traversal, prompt/task resolution, TXT sidecar writing, JSONL audit "
        "streaming, run-config export, CUDA inference, and CaptionForge global cache "
        "integration. Designed as a lightweight raw-caption witness alongside Joy and Qwen."
    ),
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


# -------------------------------------------------------------------------
# Registry
# -------------------------------------------------------------------------

@dataclass(frozen=True)
class FlorenceModelInfo:
    repo_id: str
    local_folder: str
    notes: str = ""


MODEL_REGISTRY: dict[str, FlorenceModelInfo] = {
    "Florence-2-base-ft": FlorenceModelInfo(
        repo_id="microsoft/Florence-2-base-ft",
        local_folder="Florence-2-base-ft",
        notes="Fine-tuned Florence-2 base checkpoint; good first lightweight CaptionForge witness.",
    ),
    "Florence-2-large-ft": FlorenceModelInfo(
        repo_id="microsoft/Florence-2-large-ft",
        local_folder="Florence-2-large-ft",
        notes="Fine-tuned Florence-2 large checkpoint; heavier but usually stronger than base.",
    ),
    "Florence-2-base": FlorenceModelInfo(
        repo_id="microsoft/Florence-2-base",
        local_folder="Florence-2-base",
        notes="Original Florence-2 base checkpoint.",
    ),
    "Florence-2-large": FlorenceModelInfo(
        repo_id="microsoft/Florence-2-large",
        local_folder="Florence-2-large",
        notes="Original Florence-2 large checkpoint.",
    ),
    "Florence-2-large-no-flash-attn": FlorenceModelInfo(
        repo_id="multimodalart/Florence-2-large-no-flash-attn",
        local_folder="Florence-2-large-no-flash-attn",
        notes="Community Florence-2 large variant intended to avoid flash-attn dependency friction.",
    ),

    "Florence-2-base-PromptGen": FlorenceModelInfo(
        repo_id="MiaoshouAI/Florence-2-base-PromptGen",
        local_folder="Florence-2-base-PromptGen",
        notes="Community Florence-2 base variant with enhanced prompt generation capabilities.",
    )
}

SUPPORTED_EXTENSIONS = {
    ".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff",
}

TASK_PROMPTS: dict[str, str] = {
    "Caption": "<CAPTION>",
    "Detailed Caption": "<DETAILED_CAPTION>",
    "More Detailed Caption": "<MORE_DETAILED_CAPTION>",
}

DEFAULT_TASK_NAME = "Detailed Caption"

# Kept as a list for ComfyUI widget parity with other engines.
MEMORY_MODES: dict[str, dict[str, Any]] = {
    "Default": {},
}


# -------------------------------------------------------------------------
# Dataclasses
# -------------------------------------------------------------------------

@dataclass
class GenerationConfig:
    max_new_tokens: int = 384
    num_beams: int = 3
    temperature: float = 0.0
    top_p: float = 0.90
    top_k: int = 50
    seed: Optional[int] = None

    @property
    def do_sample(self) -> bool:
        return self.temperature > 0


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
class FlorenceCaptionConfig:
    model_name: str = "Florence-2-base-ft"
    model_path: str = ""
    model_root: str = "models/LLM/JLC_FlorenceCaption"

    memory_mode: str = "Default"
    dtype: str = "fp32"          # auto, bf16, fp16, fp32; Florence coerces bf16 to fp32 for stability
    device: str = "auto"         # auto, cuda, cuda:0. CPU disabled in release mode.
    trust_remote_code: bool = True
    keep_loaded: bool = True
    cache_policy: str = ""        # '', evict_other_caption_models, keep_this_model, unload_after_run, none
    quiet_transformers_load: bool = True

    max_size: int = 1024
    task_prompt: str = TASK_PROMPTS[DEFAULT_TASK_NAME]
    allow_download: bool = True
    use_comfy_model_management: bool = True


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
    task_prompt: str
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
    model_family: str = "florence"
    ensemble_run_index: int = 0
    image_key: str = ""


@dataclass
class BatchCaptionResult:
    records: list[CaptionRecord] = field(default_factory=list)
    skipped: int = 0
    failed: int = 0

    @property
    def processed(self) -> int:
        return len([r for r in self.records if r.status == "ok"])

    @property
    def captions_text(self) -> str:
        return "\n\n".join(r.caption for r in self.records if r.status == "ok")

    @property
    def jsonl_text(self) -> str:
        return "\n".join(json.dumps(record_to_json(r), ensure_ascii=False) for r in self.records)


# -------------------------------------------------------------------------
# Helpers
# -------------------------------------------------------------------------

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


def parse_extensions(value: str) -> set[str]:
    if not value.strip():
        return set(SUPPORTED_EXTENSIONS)
    out: set[str] = set()
    for item in value.split(","):
        item = item.strip().lower()
        if not item:
            continue
        if not item.startswith("."):
            item = "." + item
        out.add(item)
    return out


def resolve_task_prompt(task_name: str = DEFAULT_TASK_NAME, custom_task_prompt: str = "") -> str:
    custom = (custom_task_prompt or "").strip()
    if custom:
        return custom
    return TASK_PROMPTS.get(task_name, TASK_PROMPTS[DEFAULT_TASK_NAME])


UNWANTED_PREFIXES = [
    "The image depicts ", "The image shows ", "This image depicts ", "This image shows ",
    "An image of ", "A photo of ", "A photograph of ", "The photo shows ", "This photo shows ",
]


def normalize_caption(caption: str, strip_boilerplate_prefixes: bool = True, strip_trailing_period: bool = True) -> str:
    text = str(caption or "").strip()
    if strip_boilerplate_prefixes:
        for prefix in UNWANTED_PREFIXES:
            if text.lower().startswith(prefix.lower()):
                text = text[len(prefix):].strip()
                break
    text = text.replace("\n", " ")
    text = re.sub(r"\s+", " ", text).strip()
    text = re.sub(r"\s+,", ",", text)
    text = re.sub(r",\s*,+", ",", text)
    text = re.sub(r"\s+\.", ".", text)
    if strip_trailing_period and text.endswith("."):
        text = text[:-1].strip()
    return text


def apply_replacements(caption: str, rules: list[tuple[str, str]], case_insensitive: bool = True, whole_words_only: bool = False) -> str:
    if not rules:
        return caption
    result = caption
    flags = re.IGNORECASE if case_insensitive else 0
    for old, new in rules:
        old = old.strip()
        if not old:
            continue
        pattern = re.escape(old)
        if whole_words_only:
            pattern = r"\b" + pattern + r"\b"
        result = re.sub(pattern, new.strip(), result, flags=flags)
    return result


def remove_forbidden_phrases(caption: str, forbidden_phrases: list[str]) -> str:
    result = caption
    for phrase in forbidden_phrases or []:
        phrase = phrase.strip()
        if phrase:
            result = re.sub(re.escape(phrase), "", result, flags=re.IGNORECASE)
    result = re.sub(r"\s+,", ",", result)
    result = re.sub(r",\s*,+", ",", result)
    result = re.sub(r"\s+", " ", result)
    return result.strip(" ,")


def add_trigger_prefix_suffix(caption: str, trigger: str = "", prefix: str = "", suffix: str = "") -> str:
    parts: list[str] = []
    for item in (trigger, prefix):
        item = (item or "").strip().strip(" ,")
        if item:
            parts.append(item)
    if caption:
        parts.append(caption.lstrip(" ,"))
    final = ", ".join(parts)
    final = re.sub(r",\s*,+", ",", final).strip(" ,")
    suffix = (suffix or "").strip()
    if suffix:
        final = final + suffix
    return final.strip()


def cleanup_caption(caption: str, config: CleanupConfig) -> str:
    text = normalize_caption(caption, config.strip_boilerplate_prefixes, config.strip_trailing_period)
    text = apply_replacements(text, config.replacement_rules, config.replace_case_insensitive, config.replace_whole_words_only)
    text = remove_forbidden_phrases(text, config.forbidden_phrases)
    text = normalize_caption(text, config.strip_boilerplate_prefixes, config.strip_trailing_period)
    return add_trigger_prefix_suffix(text, trigger=config.trigger, prefix=config.prefix, suffix=config.suffix)


def iter_image_files(input_path: str | Path, recursive: bool = True, filename_glob: str = "*", extensions: Optional[set[str]] = None) -> Iterable[Path]:
    p = Path(input_path)
    exts = extensions or set(SUPPORTED_EXTENSIONS)
    filename_glob = (filename_glob or "*").strip() or "*"

    if p.is_file():
        if p.suffix.lower() in exts and fnmatch.fnmatch(p.name, filename_glob):
            yield p
        return

    if not p.exists():
        raise FileNotFoundError(f"input_path does not exist: {p}")
    if not p.is_dir():
        raise NotADirectoryError(f"input_path is not a file or folder: {p}")

    iterator = p.rglob("*") if recursive else p.glob("*")
    for child in sorted(iterator):
        if child.is_file() and child.suffix.lower() in exts and fnmatch.fnmatch(child.name, filename_glob):
            yield child


def load_image_file(path: str | Path) -> Image.Image:
    return Image.open(path).convert("RGB")


def resize_for_model(image: Image.Image, max_size: int) -> Image.Image:
    if max_size <= 0:
        return image
    width, height = image.size
    longest = max(width, height)
    if longest <= max_size:
        return image
    scale = max_size / float(longest)
    return image.resize((max(1, round(width * scale)), max(1, round(height * scale))), Image.Resampling.LANCZOS)


def square_pad_for_florence(image: Image.Image, fill: tuple[int, int, int] = (0, 0, 0)) -> Image.Image:
    """Pad a PIL image to square dimensions for Florence-2 vision tower compatibility.

    Some Florence-2 remote-code revisions assert that the encoded visual feature
    map is square. CaptionForge normally preserves aspect ratio during resizing,
    which can produce non-square feature maps for portrait/landscape images.
    Padding after resize preserves the visible image content while satisfying
    Florence's square-feature-map assumption.
    """
    image = image.convert("RGB")
    width, height = image.size
    if width == height:
        return image

    side = max(width, height)
    canvas = Image.new("RGB", (side, side), fill)
    left = (side - width) // 2
    top = (side - height) // 2
    canvas.paste(image, (left, top))
    return canvas


def prepare_florence_image(image: Image.Image, max_size: int) -> Image.Image:
    """Resize then square-pad an image for Florence-2 generation.

    Florence-2 remote vision code can assert on non-square feature maps. In
    addition, very large square inputs can create excessive visual tokens and,
    on some Windows/CUDA stacks, can hard-crash the host process rather than
    raising a Python OOM. Keep Florence conservative as a lightweight witness.
    """
    requested = int(max_size or 0)
    effective_max = 768 if requested <= 0 else min(requested, 768)
    return square_pad_for_florence(resize_for_model(image, effective_max))


def sidecar_txt_path(image_path: Path, output_dir: str | Path | None = None) -> Path:
    if output_dir:
        out = Path(output_dir)
        safe_mkdir(out)
        return out / f"{image_path.stem}.txt"
    return image_path.with_suffix(".txt")


def backup_existing_file(path: Path, dry_run: bool = False) -> Optional[Path]:
    if not path.exists():
        return None
    backup_path = path.with_name(f"{path.name}.bak_{timestamp()}")
    if not dry_run:
        shutil.copy2(path, backup_path)
    return backup_path


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
    seen: set[str] = set()
    if not path.exists():
        return seen
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            image_value = record.get("image") or record.get("image_path") or record.get("source")
            if image_value:
                seen.add(str(image_value))
                try:
                    seen.add(str(Path(image_value)))
                except Exception:
                    pass
    return seen


def record_to_json(record: CaptionRecord) -> dict[str, Any]:
    return asdict(record)


def append_jsonl_records(path: Path, records: list[CaptionRecord], dry_run: bool = False) -> None:
    if dry_run or not records:
        return
    safe_mkdir(path.parent)
    with path.open("a", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record_to_json(record), ensure_ascii=False) + "\n")


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


# -------------------------------------------------------------------------
# Model path / download helpers
# -------------------------------------------------------------------------

def model_folder_has_weights(local_path: Path) -> bool:
    if not local_path.exists() or not local_path.is_dir():
        return False
    patterns = ["*.safetensors", "*.bin", "*.pt", "*.pth", "*.gguf", "*.ckpt"]
    return any(any(local_path.rglob(pattern)) for pattern in patterns)


def get_registry_model_path(model_name: str, model_root: str | Path) -> Path:
    if model_name not in MODEL_REGISTRY:
        raise KeyError(f"Unknown Florence model_name: {model_name}. Known models: {', '.join(MODEL_REGISTRY)}")
    return Path(model_root) / MODEL_REGISTRY[model_name].local_folder


def download_registry_model_if_needed(model_name: str, model_root: str | Path, metadata_only: bool = False, allow_download: bool = True) -> Path:
    local_path = get_registry_model_path(model_name, model_root)
    if not metadata_only and model_folder_has_weights(local_path):
        return local_path
    if not allow_download:
        if metadata_only:
            safe_mkdir(local_path)
            return local_path
        raise FileNotFoundError(f"Model folder does not contain weights and allow_download=False: {local_path}")

    try:
        from huggingface_hub import snapshot_download
    except Exception as exc:
        raise RuntimeError("Missing dependency: huggingface_hub. Install it in the active venv: pip install huggingface_hub") from exc

    info = MODEL_REGISTRY[model_name]
    safe_mkdir(Path(model_root))
    safe_mkdir(local_path)
    if metadata_only:
        print(f"[JLC Florence Engine] Probe download for {info.repo_id} -> {local_path}")
        snapshot_download(
            repo_id=info.repo_id,
            local_dir=str(local_path),
            ignore_patterns=["*.safetensors", "*.bin", "*.pt", "*.pth", "*.gguf", "*.onnx", "*.ckpt", "*.h5", "*.msgpack", "*.tflite"],
        )
        return local_path

    print(f"[JLC Florence Engine] Downloading {info.repo_id} -> {local_path}")
    snapshot_download(repo_id=info.repo_id, local_dir=str(local_path))
    return local_path


def probe_registry_model_download(model_name: str, model_root: str | Path) -> str:
    local_path = download_registry_model_if_needed(model_name, model_root, metadata_only=True, allow_download=True)
    files: list[str] = []
    try:
        for p in sorted(local_path.rglob("*")):
            if p.is_file():
                files.append(str(p.relative_to(local_path)))
    except Exception:
        files = []
    preview = "\n".join(files[:40])
    if len(files) > 40:
        preview += f"\n... plus {len(files) - 40} more files"
    return (
        "JLC Florence Caption download probe completed.\n\n"
        f"Model: {model_name}\nFolder: {local_path}\n\n"
        "Large model weight files were intentionally skipped.\n\n"
        f"Files found:\n{preview if preview else '(no files listed)'}"
    )


# -------------------------------------------------------------------------
# Optional Comfy bridge
# -------------------------------------------------------------------------

def _try_get_comfy_model_management():
    try:
        import comfy.model_management as model_management
        return model_management
    except Exception:
        return None


def _unload_florence_bundle(bundle: dict[str, Any]) -> None:
    model = bundle.get("model") if isinstance(bundle, dict) else None
    if model is not None:
        try:
            model.to("cpu")
        except Exception:
            pass
        try:
            del model
        except Exception:
            pass
    if isinstance(bundle, dict):
        bundle.clear()




# -------------------------------------------------------------------------
# Florence-2 compatibility helpers
# -------------------------------------------------------------------------

def patch_florence2_config_compat(local_path: str | Path) -> None:
    """
    Patch Florence-2 config.json for Transformers/remote-code compatibility.

    Some Florence-2 remote configuration code expects
    text_config.forced_bos_token_id to exist before checking it. Certain
    Transformers versions surface an AttributeError if the field is absent in
    config.json:

        'Florence2LanguageConfig' object has no attribute 'forced_bos_token_id'

    This is a local metadata-only patch. It does not alter model weights.
    """
    config_path = Path(local_path) / "config.json"

    if not config_path.exists():
        return

    try:
        original_text = config_path.read_text(encoding="utf-8")
        data = json.loads(original_text)
    except Exception:
        return

    text_config = data.get("text_config")
    if not isinstance(text_config, dict):
        return

    changed = False

    if "forced_bos_token_id" not in text_config:
        text_config["forced_bos_token_id"] = None
        changed = True

    if not changed:
        return

    backup_path = config_path.with_name("config.json.bak_captionforge")
    try:
        if not backup_path.exists():
            backup_path.write_text(original_text, encoding="utf-8")
    except Exception:
        pass

    config_path.write_text(
        json.dumps(data, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    print(
        "[JLC Florence Engine] Patched Florence-2 config compatibility: "
        "text_config.forced_bos_token_id = null"
    )



def _patch_text_in_file(path: Path, replacements: list[tuple[str, str]], label: str) -> bool:
    """Apply simple text replacements to a Python source file, with one backup."""
    if not path.exists() or not path.is_file():
        return False

    try:
        original_text = path.read_text(encoding="utf-8")
    except Exception:
        return False

    new_text = original_text
    for old, new in replacements:
        new_text = new_text.replace(old, new)

    if new_text == original_text:
        return False

    backup_path = path.with_name(path.name + ".bak_captionforge")
    try:
        if not backup_path.exists():
            backup_path.write_text(original_text, encoding="utf-8")
    except Exception:
        pass

    try:
        path.write_text(new_text, encoding="utf-8")
    except Exception:
        return False

    print(f"[JLC Florence Engine] Patched Florence-2 remote code compatibility: {label}: {path}")
    return True



def patch_florence2_tokenizer_compat() -> None:
    """
    Patch tokenizer classes for Florence-2 processor compatibility.

    Some Florence-2 processor remote-code versions access
    tokenizer.additional_special_tokens directly. In newer Transformers builds,
    RobertaTokenizer may not expose that as a direct attribute unless it is
    present in special_tokens_map, which raises:

        RobertaTokenizer has no attribute additional_special_tokens

    Add a conservative read-only property at class level so Florence processor
    construction can proceed.
    """
    try:
        import transformers
    except Exception:
        return

    def _get_additional_special_tokens(self):
        try:
            value = self.special_tokens_map.get("additional_special_tokens", [])
            return value if value is not None else []
        except Exception:
            return []

    patched = []
    for class_name in ("RobertaTokenizer", "RobertaTokenizerFast"):
        cls = getattr(transformers, class_name, None)
        if cls is None:
            continue
        try:
            existing = cls.__dict__.get("additional_special_tokens")
            if existing is None:
                setattr(cls, "additional_special_tokens", property(_get_additional_special_tokens))
                patched.append(class_name)
        except Exception:
            continue

    if patched:
        print(
            "[JLC Florence Engine] Patched tokenizer compatibility: "
            + ", ".join(patched)
            + ".additional_special_tokens"
        )

def patch_florence2_remote_code_compat(local_path: str | Path) -> None:
    """
    Patch Florence-2 remote-code source for newer Transformers compatibility.

    The config.json metadata patch is not enough in some ComfyUI/Transformers
    combinations because the Florence remote config class directly reads
    self.forced_bos_token_id before the attribute exists. The failing file is
    configuration_florence2.py, often loaded from Hugging Face's cached
    transformers_modules directory. This helper patches both the local model
    copy and any already-cached Florence copies.

    This is a source-compatibility patch only. It does not alter model weights.
    """
    replacements = [
        (
            'if self.forced_bos_token_id is None and kwargs.get("force_bos_token_to_be_generated", False):',
            'if getattr(self, "forced_bos_token_id", None) is None and kwargs.get("force_bos_token_to_be_generated", False):',
        ),
        (
            "if self.forced_bos_token_id is None and kwargs.get('force_bos_token_to_be_generated', False):",
            "if getattr(self, 'forced_bos_token_id', None) is None and kwargs.get('force_bos_token_to_be_generated', False):",
        ),
        (
            "tokenizer.additional_special_tokens",
            "tokenizer.special_tokens_map.get('additional_special_tokens', [])",
        ),
        (
            "class Florence2PreTrainedModel(PreTrainedModel):\n",
            "class Florence2PreTrainedModel(PreTrainedModel):\n    _supports_sdpa = False\n    _supports_flash_attn_2 = False\n",
        ),
        (
            "class Florence2ForConditionalGeneration(Florence2PreTrainedModel):\n",
            "class Florence2ForConditionalGeneration(Florence2PreTrainedModel):\n    _supports_sdpa = False\n    _supports_flash_attn_2 = False\n",
        ),
    ]

    local_path = Path(local_path)
    candidates: list[Path] = []

    local_config_py = local_path / "configuration_florence2.py"
    if local_config_py.exists():
        candidates.append(local_config_py)

    local_processing_py = local_path / "processing_florence2.py"
    if local_processing_py.exists():
        candidates.append(local_processing_py)

    local_modeling_py = local_path / "modeling_florence2.py"
    if local_modeling_py.exists():
        candidates.append(local_modeling_py)

    # Also patch already-cached dynamic modules. This matters after the first
    # failed attempt because Transformers may keep using the cached remote file.
    hf_modules_root = Path.home() / ".cache" / "huggingface" / "modules" / "transformers_modules"
    if hf_modules_root.exists():
        try:
            candidates.extend(hf_modules_root.rglob("configuration_florence2.py"))
            candidates.extend(hf_modules_root.rglob("processing_florence2.py"))
            candidates.extend(hf_modules_root.rglob("modeling_florence2.py"))
        except Exception:
            pass

    seen: set[Path] = set()
    patched_any = False
    for path in candidates:
        try:
            resolved = path.resolve()
        except Exception:
            resolved = path
        if resolved in seen:
            continue
        seen.add(resolved)
        patched_any = _patch_text_in_file(path, replacements, "Florence-2 remote compatibility") or patched_any

    if not patched_any:
        print("[JLC Florence Engine] Florence-2 remote-code compatibility patch: no source edits needed.")


def _get_nested_attr(obj: Any, path: str):
    current = obj
    for part in path.split("."):
        if current is None or not hasattr(current, part):
            return None
        current = getattr(current, part)
    return current


def patch_florence2_tied_language_weights(model: Any) -> None:
    """
    Repair Florence-2 tied language weights after remote-code/checkpoint loading.

    In some Transformers/remote-code combinations, Florence-2 loads with these
    reported as missing/newly initialized:
      - language_model.model.encoder.embed_tokens.weight
      - language_model.model.decoder.embed_tokens.weight
      - language_model.lm_head.weight

    That produces fluent-looking nonsense because the language model's token
    embeddings / lm_head are random. The checkpoint normally carries a shared
    embedding matrix, so tie encoder, decoder, and lm_head back to it when the
    structure is present.
    """
    try:
        if hasattr(model, "tie_weights"):
            model.tie_weights()
    except Exception:
        pass

    shared = None
    for candidate in (
        "language_model.model.shared",
        "language_model.shared",
        "model.shared",
        "shared",
    ):
        module = _get_nested_attr(model, candidate)
        weight = getattr(module, "weight", None)
        if weight is not None:
            shared = module
            break

    if shared is None or getattr(shared, "weight", None) is None:
        print("[JLC Florence Engine] Florence weight-tie patch skipped: no shared embedding found.")
        return

    patched: list[str] = []

    for candidate in (
        "language_model.model.encoder.embed_tokens",
        "language_model.model.decoder.embed_tokens",
    ):
        module = _get_nested_attr(model, candidate)
        if module is not None and hasattr(module, "weight"):
            try:
                module.weight = shared.weight
                patched.append(candidate + ".weight")
            except Exception:
                pass

    for candidate in (
        "language_model.lm_head",
        "lm_head",
    ):
        module = _get_nested_attr(model, candidate)
        if module is not None and hasattr(module, "weight"):
            try:
                module.weight = shared.weight
                patched.append(candidate + ".weight")
            except Exception:
                pass

    if patched:
        print("[JLC Florence Engine] Patched Florence tied language weights: " + ", ".join(patched))
    else:
        print("[JLC Florence Engine] Florence weight-tie patch found shared embedding but no target modules.")


# -------------------------------------------------------------------------
# Engine
# -------------------------------------------------------------------------

class FlorenceCaptionEngine:
    def __init__(self, config: FlorenceCaptionConfig, generation: Optional[GenerationConfig] = None, cleanup: Optional[CleanupConfig] = None) -> None:
        self.config = config
        self.generation = generation or GenerationConfig()
        self.cleanup = cleanup or CleanupConfig()
        self.processor = None
        self.model = None
        self.local_model_path: Optional[Path] = None
        self.model_size_bytes: Optional[int] = None
        self._comfy_mm = _try_get_comfy_model_management() if config.use_comfy_model_management else None
        self.inference_device = self._resolve_inference_device(config.device)
        self.offload_device = self._resolve_offload_device(config.device)

    def resolve_model_path(self) -> Path:
        if self.config.model_path.strip():
            return Path(self.config.model_path).expanduser()
        return download_registry_model_if_needed(self.config.model_name, self.config.model_root, metadata_only=False, allow_download=self.config.allow_download)

    def _cache_policy(self) -> str:
        policy = getattr(self.config, "cache_policy", "")
        policy = policy.strip() if isinstance(policy, str) else ""
        return policy or ("evict_other_caption_models" if getattr(self.config, "keep_loaded", True) else "unload_after_run")

    def _cache_key(self, local_path: Path) -> str:
        return make_cache_key(
            role="caption",
            family="florence",
            model_path=str(local_path.resolve()),
            device=str(self.inference_device),
            quantization=self.config.memory_mode,
            dtype="fp32-stable",
            revision="v0.1.9-tied-language-weights",
        )

    def _resolve_inference_device(self, device: str) -> torch.device:
        if self._comfy_mm is not None:
            try:
                dev = self._comfy_mm.get_torch_device()
                if dev.type != "cuda":
                    raise RuntimeError("CaptionForge Florence requires CUDA for release builds. Silent CPU fallback is disabled.")
                return dev
            except RuntimeError:
                raise
            except Exception:
                pass
        if device == "auto":
            if not torch.cuda.is_available():
                raise RuntimeError("CaptionForge Florence requires CUDA for release builds. Silent CPU fallback is disabled.")
            return torch.device("cuda")
        dev = torch.device(device)
        if dev.type != "cuda":
            raise RuntimeError(f"CaptionForge Florence release mode does not allow inference device={device!r}.")
        return dev

    def _resolve_offload_device(self, device: str) -> torch.device:
        if self._comfy_mm is not None:
            try:
                return self._comfy_mm.unet_offload_device()
            except Exception:
                pass
        return torch.device("cpu")

    @staticmethod
    def _resolve_dtype(dtype: str) -> torch.dtype | str:
        text = dtype.lower().strip()
        if text == "auto":
            # Florence remote code has been more stable in fp32 than auto/bf16
            # inside ComfyUI/Windows/Transformers compatibility stacks.
            return torch.float32
        if text in {"bf16", "bfloat16"}:
            # Do not use bf16 for Florence v0.1.x. It can reach native CUDA/torch
            # paths that hard-crash the host process on some Windows stacks.
            print("[JLC Florence Engine] Coercing requested bf16 dtype to fp32 for Florence stability.")
            return torch.float32
        if text in {"fp16", "float16"}:
            return torch.float16
        if text in {"fp32", "float32"}:
            return torch.float32
        raise ValueError(f"Unsupported dtype: {dtype}")

    def _module_size(self, module) -> int:
        if self._comfy_mm is not None:
            try:
                return int(self._comfy_mm.module_size(module))
            except Exception:
                pass
        total = 0
        try:
            for p in module.parameters():
                total += p.numel() * p.element_size()
            for b in module.buffers():
                total += b.numel() * b.element_size()
        except Exception:
            total = 0
        return total

    def _free_memory(self, bytes_needed: Optional[int], device: torch.device) -> None:
        if bytes_needed is None:
            return
        if self._comfy_mm is not None:
            try:
                self._comfy_mm.free_memory(bytes_needed, device)
                return
            except Exception:
                pass
        if device.type == "cuda" and torch.cuda.is_available():
            torch.cuda.empty_cache()

    def _soft_empty_cache(self) -> None:
        if self._comfy_mm is not None:
            try:
                self._comfy_mm.soft_empty_cache()
                return
            except Exception:
                pass
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def load(self) -> None:
        local_path = self.resolve_model_path()
        if not local_path.exists():
            raise FileNotFoundError(f"Model path does not exist: {local_path}")
        self.local_model_path = local_path
        cache_key = self._cache_key(local_path)
        cached = get_cached_model(cache_key)
        if cached is not None:
            self.processor = cached.get("processor")
            self.model = cached.get("model")
            self.model_size_bytes = cached.get("model_size_bytes")
            print(f"[JLC Florence Engine] Reusing cached model bundle: {local_path}")
            return

        prepare_for_model_load(cache_key, policy=self._cache_policy(), role="caption")
        if self.generation.seed is not None:
            set_seed(self.generation.seed)

        try:
            from transformers import AutoProcessor, AutoModelForCausalLM
        except Exception as exc:
            raise RuntimeError(
                "Could not import Florence Transformers classes. Update dependencies:\n"
                "  pip install -U transformers accelerate pillow huggingface_hub timm einops"
            ) from exc

        torch_dtype = self._resolve_dtype(self.config.dtype)

        patch_florence2_config_compat(local_path)
        patch_florence2_tokenizer_compat()
        patch_florence2_remote_code_compat(local_path)

        print(f"[JLC Florence Engine] Loading processor: {local_path}")
        self.processor = AutoProcessor.from_pretrained(
            str(local_path),
            trust_remote_code=self.config.trust_remote_code,
        )

        print(f"[JLC Florence Engine] Loading model: {local_path} requested_dtype={self.config.dtype} resolved_dtype={torch_dtype}")
        self.model = AutoModelForCausalLM.from_pretrained(
            str(local_path),
            torch_dtype=torch_dtype,
            trust_remote_code=self.config.trust_remote_code,
            attn_implementation="eager",
        )
        patch_florence2_tied_language_weights(self.model)
        self.model.eval()
        self.model_size_bytes = self._module_size(self.model)
        self._free_memory(self.model_size_bytes, self.inference_device)
        self.model.to(self.inference_device)

        register_model(
            cache_key,
            {"processor": self.processor, "model": self.model, "model_size_bytes": self.model_size_bytes, "local_path": str(local_path)},
            family="florence",
            model_path=str(local_path),
            device=str(self.inference_device),
            quantization=self.config.memory_mode,
            role="caption",
            unload_fn=_unload_florence_bundle,
            keep=self._cache_policy() == "keep_this_model",
        )
        print(f"[JLC Florence Engine] Loaded model: {self.config.model_name}")

    def prepare_for_inference(self) -> None:
        if self.processor is None or self.model is None:
            self.load()
        if self.model is None:
            raise RuntimeError("Florence model failed to load.")
        self._free_memory(self.model_size_bytes, self.inference_device)
        self.model.to(self.inference_device)

    def cleanup_after_inference(self) -> None:
        if self._cache_policy() != "unload_after_run":
            return
        if self.model is not None:
            self.model.to(self.offload_device)
            self._soft_empty_cache()

    def unload(self) -> None:
        if self.local_model_path is not None:
            unload_after_run(self._cache_key(self.local_model_path), enabled=True)
        self.processor = None
        self.model = None
        self.model_size_bytes = None
        self._soft_empty_cache()

    @torch.inference_mode()
    def caption_pil(self, image: Image.Image) -> tuple[str, str]:
        if self.generation.seed is not None:
            set_seed(self.generation.seed)

        self.prepare_for_inference()
        if self.model is None or self.processor is None:
            raise RuntimeError("Model is not loaded.")

        image = image.convert("RGB")
        image_for_model = prepare_florence_image(image, self.config.max_size)
        task_prompt = (self.config.task_prompt or TASK_PROMPTS[DEFAULT_TASK_NAME]).strip()

        inputs = self.processor(text=task_prompt, images=image_for_model, return_tensors="pt")
        inputs = {k: v.to(self.inference_device) if hasattr(v, "to") else v for k, v in inputs.items()}

        # Florence's processor emits pixel_values as float32, while the model may
        # be loaded as bf16/fp16. The vision tower convolution requires image
        # inputs and weights/biases to have matching floating dtypes.
        model_dtype = getattr(self.model, "dtype", None)
        if model_dtype is None:
            try:
                model_dtype = next(self.model.parameters()).dtype
            except Exception:
                model_dtype = None

        if (
            isinstance(model_dtype, torch.dtype)
            and "pixel_values" in inputs
            and hasattr(inputs["pixel_values"], "dtype")
            and torch.is_floating_point(inputs["pixel_values"])
        ):
            inputs["pixel_values"] = inputs["pixel_values"].to(dtype=model_dtype)

        generation_kwargs: dict[str, Any] = {
            "max_new_tokens": int(self.generation.max_new_tokens),
            "num_beams": max(1, int(self.generation.num_beams)),
            "do_sample": bool(self.generation.temperature > 0),
            # Florence-2 remote code in some Transformers builds still expects
            # tuple-style past_key_values, while newer Transformers may pass
            # EncoderDecoderCache objects. Disabling generation cache avoids
            # that compatibility path.
            "use_cache": False,
        }
        if self.generation.temperature > 0:
            generation_kwargs["temperature"] = float(self.generation.temperature)
            generation_kwargs["top_p"] = float(self.generation.top_p)
            generation_kwargs["top_k"] = int(self.generation.top_k)

        try:
            generated_ids = self.model.generate(**inputs, **generation_kwargs)
        finally:
            self.cleanup_after_inference()

        generated_text = self.processor.batch_decode(generated_ids, skip_special_tokens=False)[0]
        raw_caption = generated_text.strip()

        # Florence processor normally returns a dict keyed by task prompt.
        try:
            parsed = self.processor.post_process_generation(
                generated_text,
                task=task_prompt,
                image_size=(image_for_model.width, image_for_model.height),
            )
            value = parsed.get(task_prompt, parsed) if isinstance(parsed, dict) else parsed
            if isinstance(value, str):
                raw_caption = value.strip()
            elif isinstance(value, dict) and task_prompt in value and isinstance(value[task_prompt], str):
                raw_caption = value[task_prompt].strip()
            else:
                raw_caption = str(value).strip()
        except Exception:
            # Fallback for custom prompts or processor versions that do not parse.
            raw_caption = re.sub(r"<[^>]+>", " ", generated_text)
            raw_caption = normalize_caption(raw_caption, strip_boilerplate_prefixes=False, strip_trailing_period=False)

        final_caption = cleanup_caption(raw_caption, self.cleanup)
        return final_caption, raw_caption

    def caption_path(self, image_path: str | Path) -> CaptionRecord:
        p = Path(image_path)
        if not p.exists():
            raise FileNotFoundError(f"Image does not exist: {p}")
        image = load_image_file(p)
        final_caption, raw_caption = self.caption_pil(image)
        return CaptionRecord(
            image=str(p),
            caption=final_caption,
            raw_caption=raw_caption,
            model_name=self.config.model_name,
            model_path=str(self.local_model_path or self.config.model_path),
            task_prompt=self.config.task_prompt,
            seed=self.generation.seed,
            temperature=self.generation.temperature,
            top_p=self.generation.top_p,
            top_k=self.generation.top_k,
            num_beams=self.generation.num_beams,
            max_new_tokens=self.generation.max_new_tokens,
            max_size=self.config.max_size,
            timestamp=iso_timestamp(),
            captionforge_pass="A",
            model_family="florence",
            ensemble_run_index=0,
            image_key=str(p.resolve()),
        )

    def caption_batch(self, batch: BatchCaptionConfig) -> BatchCaptionResult:
        if not batch.input_path.strip():
            raise ValueError("BatchCaptionConfig.input_path is required.")
        result = BatchCaptionResult()
        images = list(iter_image_files(batch.input_path, batch.recursive, batch.filename_glob, batch.extensions))
        if batch.limit and batch.limit > 0:
            images = images[: int(batch.limit)]
        if not images:
            print(f"[JLC Florence Engine] No images found in: {batch.input_path}")
            return result

        output_dir = Path(batch.output_dir) if batch.output_dir.strip() else None
        jsonl_path: Optional[Path] = None
        if batch.write_jsonl:
            base_dir = output_dir or images[0].parent
            jsonl_path = base_dir / (batch.jsonl_filename.strip() or "captions.jsonl")
        also_jsonl_path = Path(batch.also_jsonl_path) if batch.also_jsonl_path.strip() else None

        seen_jsonl_images: set[str] = set()
        if batch.skip_existing_jsonl_images:
            if jsonl_path is not None:
                seen_jsonl_images.update(load_existing_jsonl_images(jsonl_path))
            if also_jsonl_path is not None:
                seen_jsonl_images.update(load_existing_jsonl_images(also_jsonl_path))

        if batch.write_run_config:
            base_dir = output_dir or images[0].parent
            run_config_name = batch.run_config_filename.strip() or f"jlc_florence_caption_run_config_{timestamp()}.json"
            write_run_config_json(base_dir / run_config_name, self.build_run_config(batch), dry_run=batch.dry_run)

        print(f"[JLC Florence Engine] Found {len(images)} image(s).")
        try:
            for index, image_path in enumerate(images, start=1):
                try:
                    txt_path = sidecar_txt_path(image_path, output_dir)
                    source_for_record = str(image_path)
                    if batch.skip_existing_txt and batch.write_txt and txt_path.exists() and not batch.overwrite:
                        print(f"[{index}/{len(images)}] SKIP existing TXT: {txt_path}")
                        result.skipped += 1
                        continue
                    if batch.skip_existing_jsonl_images and (source_for_record in seen_jsonl_images or image_path.name in seen_jsonl_images):
                        print(f"[{index}/{len(images)}] SKIP existing JSONL image: {source_for_record}")
                        result.skipped += 1
                        continue
                    print(f"[{index}/{len(images)}] Captioning: {image_path}")
                    record = self.caption_path(image_path)
                    result.records.append(record)
                    if batch.write_txt:
                        written = write_text_sidecar(txt_path, record.caption, batch.overwrite, batch.backup_existing, batch.dry_run)
                        if not written:
                            result.skipped += 1
                            print(f"[{index}/{len(images)}] SKIP existing TXT after caption: {txt_path}")
                    if jsonl_path is not None:
                        append_jsonl_records(jsonl_path, [record], dry_run=batch.dry_run)
                    if also_jsonl_path is not None:
                        append_jsonl_records(also_jsonl_path, [record], dry_run=batch.dry_run)
                except KeyboardInterrupt:
                    raise
                except Exception as exc:
                    result.failed += 1
                    print(f"[JLC Florence Engine] ERROR on {image_path}: {exc}")
                    error_record = CaptionRecord(
                        image=str(image_path), caption="", raw_caption="", model_name=self.config.model_name,
                        model_path=str(self.local_model_path or self.config.model_path), task_prompt=self.config.task_prompt,
                        seed=self.generation.seed, temperature=self.generation.temperature, top_p=self.generation.top_p,
                        top_k=self.generation.top_k, num_beams=self.generation.num_beams,
                        max_new_tokens=self.generation.max_new_tokens, max_size=self.config.max_size,
                        timestamp=iso_timestamp(), status="error", error=str(exc), captionforge_pass="A",
                        model_family="florence", ensemble_run_index=0, image_key=str(image_path.resolve()),
                    )
                    result.records.append(error_record)
                    if jsonl_path is not None:
                        append_jsonl_records(jsonl_path, [error_record], dry_run=batch.dry_run)
                    if also_jsonl_path is not None:
                        append_jsonl_records(also_jsonl_path, [error_record], dry_run=batch.dry_run)
        finally:
            if self._cache_policy() == "unload_after_run":
                self.unload()
        return result

    def build_run_config(self, batch: Optional[BatchCaptionConfig] = None) -> dict[str, Any]:
        return {
            "timestamp": iso_timestamp(),
            "engine": "JLC Florence Caption Engine",
            "florence_config": asdict(self.config),
            "generation": asdict(self.generation),
            "cleanup": {**asdict(self.cleanup), "replacement_rules": [list(rule) for rule in self.cleanup.replacement_rules]},
            "batch": json_safe(asdict(batch)) if batch is not None else None,
        }


def caption_one_image(image_path: str | Path, florence_config: FlorenceCaptionConfig, generation: Optional[GenerationConfig] = None, cleanup: Optional[CleanupConfig] = None) -> CaptionRecord:
    engine = FlorenceCaptionEngine(florence_config, generation=generation, cleanup=cleanup)
    engine.load()
    try:
        return engine.caption_path(image_path)
    finally:
        cache_policy = getattr(florence_config, "cache_policy", "")
        cache_policy = cache_policy.strip() if isinstance(cache_policy, str) else ""
        if not cache_policy:
            cache_policy = "evict_other_caption_models" if getattr(florence_config, "keep_loaded", True) else "unload_after_run"
        if cache_policy == "unload_after_run":
            engine.unload()
