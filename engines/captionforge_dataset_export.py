"""Optional, non-enlarging training image/caption export shared by both nodes."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from pathlib import Path

from PIL import Image

from .captionforge_source_identity import optional_image_filename


EXPORT_MARKER = ".captionforge-dataset.json"
MARKER_CONTENT = {"type": "captionforge_training_dataset", "version": 1}
EXPORT_DEFAULTS = {
    "enabled": False,
    "output_folder": "",
    "max_size": 0,
    "divisor": 16,
    "image_format": "PNG",
    "jpeg_quality": 95,
    "caption": "short",
}
EXPORT_WIDGETS = {
    "enabled": "Dataset - export image and caption",
    "output_folder": "Dataset - output folder",
    "max_size": "Dataset - max image size",
    "divisor": "Dataset - dimension divisor",
    "image_format": "Dataset - image format",
    "jpeg_quality": "Dataset - JPEG quality",
    "caption": "Dataset - caption",
}


def dataset_export_inputs() -> dict:
    """Append optional widgets so older API workflows remain valid."""
    specs = {
        "enabled": ("BOOLEAN", {"tooltip": "Export a resized image and matching training TXT. Planner owns these controls when connected."}),
        "output_folder": ("STRING", {"tooltip": "Parent folder for training_dataset. Blank uses Output - folder. Originals are never replaced."}),
        "max_size": ("INT", {"min": 0, "max": 8192, "step": 1, "tooltip": "Maximum long edge; never enlarges. 0 follows the configured Validator size (Planner Caption - max image size)."}),
        "divisor": ("INT", {"min": 1, "max": 512, "step": 1, "tooltip": "Round both dimensions DOWN to this multiple after resizing. 1 disables alignment. Images too small for the divisor fail export without enlargement."}),
        "image_format": (["PNG", "JPEG"], {"tooltip": "Format of the exported RGB training image."}),
        "jpeg_quality": ("INT", {"min": 1, "max": 100, "step": 1, "tooltip": "JPEG quality; ignored for PNG."}),
        "caption": (["short", "long", "taggy"], {"tooltip": "Caption written to the matching plain .txt file. Other caption variants remain available."}),
    }
    return {EXPORT_WIDGETS[key]: (kind, {"default": EXPORT_DEFAULTS[key], **options})
            for key, (kind, options) in specs.items()}


def export_settings_from_widgets(widgets: dict) -> dict:
    return {key: widgets.get(name, EXPORT_DEFAULTS[key]) for key, name in EXPORT_WIDGETS.items()}


def normalize_export_settings(values: dict | None) -> dict:
    result = {**EXPORT_DEFAULTS, **(values or {})}
    result["enabled"] = str(result["enabled"]).strip().lower() in {"true", "1", "yes", "on"}
    result["output_folder"] = str(result["output_folder"] or "").strip()
    for key, low, high in (("max_size", 0, 8192), ("divisor", 1, 512), ("jpeg_quality", 1, 100)):
        value = float(result[key])
        if not value.is_integer() or not low <= value <= high:
            raise ValueError(f"Dataset {key} must be an integer from {low} to {high}.")
        result[key] = int(value)
    if result["image_format"] not in {"PNG", "JPEG"} or result["caption"] not in {"short", "long", "taggy"}:
        raise ValueError("Unsupported dataset image format or caption choice.")
    return result


def dataset_root(settings: dict, output_folder: str | Path) -> Path:
    return (Path(settings["output_folder"] or output_folder).expanduser() / "training_dataset").resolve()


def is_dataset_export(path: Path) -> bool:
    """Recognize generated datasets, including after export has been disabled."""
    resolved = path.resolve()
    return any((parent / EXPORT_MARKER).is_file() for parent in (resolved, *resolved.parents))


def prepare_dataset_root(root: Path, input_path: str | Path = "") -> None:
    """Claim only an empty directory or a previously managed export directory."""
    root = root.resolve()
    if input_path and Path(input_path).resolve().is_relative_to(root):
        raise ValueError("Dataset destination contains the input path. Choose another output folder.")
    marker = root / EXPORT_MARKER
    if marker.exists():
        if marker.is_symlink() or json.loads(marker.read_text(encoding="utf-8")) != MARKER_CONTENT:
            raise ValueError("Dataset folder has an invalid CaptionForge marker.")
        return
    if root.exists() and any(root.iterdir()):
        raise ValueError(f"Dataset folder is not empty and is not managed by CaptionForge: {root}")
    root.mkdir(parents=True, exist_ok=True)
    with marker.open("x", encoding="utf-8") as stream:
        json.dump(MARKER_CONTENT, stream)


def export_dimensions(source_size: tuple[int, int], max_size: int, divisor: int) -> tuple[int, int]:
    """Use integer arithmetic so exact multiples never lose a pixel to float error."""
    width, height = source_size
    longest = max(width, height)
    scaled = tuple(edge * max_size // longest if 0 < max_size < longest else edge for edge in source_size)
    size = tuple((edge // divisor) * divisor for edge in scaled)
    if min(size) < 1:
        raise ValueError(f"Image {width}x{height} is too small for divisor {divisor} at max size {max_size}; use a smaller divisor.")
    return size


def resize_for_export(image: Image.Image, max_size: int, divisor: int) -> Image.Image:
    """Calculate dimensions first and resample once; never clamp an edge upward."""
    size = export_dimensions(image.size, max_size, divisor)
    return image.resize(size, Image.Resampling.LANCZOS) if size != image.size else image


def export_paths(root: Path, source: Path, input_root: str | Path, image_key: str, image_format: str) -> tuple[Path, Path, Path]:
    """Keep the original extension in the stem to avoid conversion collisions."""
    optional_name = optional_image_filename(image_key)
    if optional_name:
        relative = Path("optional") / optional_name
    else:
        base = Path(input_root).resolve() if input_root else source.resolve().parent
        if base.is_file():
            base = base.parent
        try:
            relative = Path("files") / source.resolve().relative_to(base)
        except ValueError:
            namespace = hashlib.sha256(str(source.resolve().parent).encode()).hexdigest()[:16]
            relative = Path("external") / namespace / source.name
    target = root / relative.parent / (relative.name + (".png" if image_format == "PNG" else ".jpg"))
    caption = target.with_suffix(".txt")
    receipt = target.with_suffix(".captionforge.json")
    check_export_destinations(root, source, (target, caption, receipt))
    return target, caption, receipt


def check_export_destinations(root: Path, source: Path, paths) -> None:
    for path in paths:
        if path.is_symlink() or not path.resolve().is_relative_to(root.resolve()):
            raise ValueError("Dataset destination escapes its protected folder.")
        if path.resolve() == source.resolve() or (path.exists() and path.samefile(source)):
            raise ValueError("Dataset export would overwrite a source image.")


def _digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def export_pair(*, source: Path, image_key: str, input_root: str | Path, root: Path,
                settings: dict, validator_max_size: int, captions: dict, overwrite: bool,
                prepared_image: Image.Image | None = None, write_variants: bool = False) -> dict:
    """Publish a pair with a receipt written last; incomplete pairs never resume."""
    source = source.resolve()
    if is_dataset_export(source):
        raise ValueError("A generated dataset image cannot be used as its own archive source.")
    prepare_dataset_root(root, input_root)
    target, caption_path, receipt_path = export_paths(root, source, input_root, image_key, settings["image_format"])
    max_size = settings["max_size"] or validator_max_size
    caption = str(captions.get(settings["caption"]) or "").strip()
    if not caption:
        raise ValueError("Selected training caption is empty.")
    variants = {style: str(captions.get(style) or "").strip() for style in ("long", "short", "taggy")} if write_variants else {}
    variant_paths = {style: target.with_name(f"{target.stem}_{style}.txt") for style in variants}
    check_export_destinations(root, source, variant_paths.values())
    stat = source.stat()
    signature = {"source": str(source), "image_key": image_key, "source_size": stat.st_size,
                 "source_mtime_ns": stat.st_mtime_ns, "max_size": max_size,
                 "divisor": settings["divisor"], "format": settings["image_format"],
                 "jpeg_quality": settings["jpeg_quality"], "caption": caption, "variants": variants}
    previous = json.loads(receipt_path.read_text(encoding="utf-8")) if receipt_path.exists() else {}
    owned_variants = set(previous.get("owned_variants", previous.get("signature", {}).get("variants", {})))
    for style, path in variant_paths.items():
        if path.exists() and style not in owned_variants:
            raise FileExistsError(f"Dataset caption variant is untracked: {path}")
    previous_image = previous.get("image") or previous.get("export", {}).get("image")
    if previous_image and Path(previous_image) != target and Path(previous_image).exists():
        raise FileExistsError("This source was exported in another image format. Choose a different Dataset output folder to avoid duplicate training images.")
    if any(path.exists() for path in (target, caption_path, receipt_path)):
        if previous.get("signature", {}).get("source") != str(source) or previous.get("signature", {}).get("image_key") != image_key:
            raise FileExistsError(f"Dataset filename is already occupied by another or untracked source: {target}")
        if not overwrite:
            if (previous.get("signature") == signature and target.is_file() and caption_path.is_file()
                    and previous.get("image_sha256") == _digest(target)
                    and previous.get("caption_sha256") == _digest(caption_path)
                    and all(path.is_file() and previous.get("variant_sha256", {}).get(style) == _digest(path)
                            for style, path in variant_paths.items())):
                return {**previous["export"], "resumed": True}
            raise FileExistsError("Dataset pair is incomplete, changed, or uses different settings; enable overwrite to regenerate it.")
    if prepared_image is None:
        with Image.open(source) as original:
            prepared_image = original.convert("RGB")
    resized = resize_for_export(prepared_image, max_size, settings["divisor"])
    result = {"status": "ok", "image": str(target), "caption": str(caption_path),
              "width": resized.width, "height": resized.height, "caption_style": settings["caption"],
              "max_size": max_size, "divisor": settings["divisor"], "resumed": False,
              "variants": {style: str(path) for style, path in variant_paths.items()}}
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary: list[Path] = []
    try:
        for _ in range(3 + len(variants)):
            descriptor, name = tempfile.mkstemp(prefix=".captionforge-", dir=target.parent)
            os.close(descriptor)
            temporary.append(Path(name))
        options = {"quality": settings["jpeg_quality"], "subsampling": 0} if settings["image_format"] == "JPEG" else {}
        resized.save(temporary[0], format=settings["image_format"], **options)
        temporary[1].write_text(caption + "\n", encoding="utf-8")
        variant_hashes = {}
        for index, (style, text) in enumerate(variants.items(), start=2):
            temporary[index].write_text(text + "\n", encoding="utf-8")
            variant_hashes[style] = _digest(temporary[index])
        receipt = {"signature": signature, "export": result,
                   "image_sha256": _digest(temporary[0]), "caption_sha256": _digest(temporary[1]),
                   "variant_sha256": variant_hashes, "owned_variants": sorted(owned_variants | set(variants))}
        # Reserve ownership before publishing, without claiming untracked files.
        reservation = {"signature": signature, "image": str(target), "owned_variants": receipt["owned_variants"]}
        if previous:
            temporary[-1].write_text(json.dumps(reservation), encoding="utf-8")
            os.replace(temporary[-1], receipt_path)
        else:
            with receipt_path.open("x", encoding="utf-8") as stream:
                json.dump(reservation, stream)
        temporary[-1].write_text(json.dumps(receipt, ensure_ascii=False, indent=2), encoding="utf-8")
        for staging, destination in zip(temporary, (target, caption_path, *variant_paths.values(), receipt_path)):
            os.replace(staging, destination)
    finally:
        for path in temporary:
            path.unlink(missing_ok=True)
    return result
