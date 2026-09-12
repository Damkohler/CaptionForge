"""Canonical source-image identities shared by CaptionForge Pass-A witnesses."""

from __future__ import annotations

from pathlib import Path


OPTIONAL_IMAGE_KEY_PREFIX = "captionforge-optional-image://"


def file_source_identity(image_path: Path, input_root: Path) -> tuple[str, str]:
    """Return the readable stem and portable, collision-safe dataset key."""
    path = Path(image_path)
    root = Path(input_root)
    image = path.stem or "image"
    if root.is_dir():
        try:
            return image, path.relative_to(root).as_posix()
        except ValueError:
            pass
    return image, path.name


def optional_image_identity(index: int) -> tuple[str, str]:
    """Return deterministic display/key values for a filename-less IMAGE tensor."""
    filename = f"comfy_image_{int(index):04d}.png"
    return Path(filename).stem, f"{OPTIONAL_IMAGE_KEY_PREFIX}{filename}"


def optional_image_filename(image_key: object) -> str:
    """Return the synthetic PNG name encoded by an optional-image key, if any."""
    text = str(image_key or "")
    if not text.startswith(OPTIONAL_IMAGE_KEY_PREFIX):
        return ""
    filename = text[len(OPTIONAL_IMAGE_KEY_PREFIX):]
    if not filename or "/" in filename or "\\" in filename:
        return ""
    return filename
