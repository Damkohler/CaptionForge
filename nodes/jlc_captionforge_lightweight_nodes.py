"""Standalone lightweight visual Q&A and caption assembly nodes for CaptionForge.

These nodes reuse CaptionForge's Ollama model choices without entering its
multi-pass caption pipeline. Slots are fixed in Python so that saved wires
remain stable when the active slot count changes.
"""

from __future__ import annotations

import base64
import io
import json
import urllib.error
import urllib.request

from .captionforge_ollama_model_dropdowns import CUSTOM_VALUE, load_ollama_model_dropdowns

MAX_SLOTS = 10
DEFAULT_OLLAMA_URL = "http://127.0.0.1:11434"
DEFAULT_SYSTEM_PROMPT = (
    "Answer the specific question using only visible evidence in the supplied image. "
    "Be concise and literal. If a detail is not reliably visible, answer 'unknown'. "
    "Do not guess, describe unrelated attributes, or add introductory commentary."
)


def _encode_image(image, max_size: int) -> str:
    """Encode one image for Ollama without changing the passthrough tensor."""
    # Import only during execution; importing this module never loads an HF VLM.
    import numpy as np
    from PIL import Image

    if image is None or len(image.shape) != 4 or image.shape[0] != 1:
        raise ValueError(
            "Targeted VLM Query requires a single IMAGE (batch size 1). "
            "Use an image-batch iterator for multiple images."
        )

    pixels = image[0].detach().cpu().clamp(0, 1).numpy()
    pixels = (np.asarray(pixels) * 255.0).round().astype(np.uint8)
    if pixels.ndim != 3 or pixels.shape[2] not in (3, 4):
        raise ValueError("Expected a ComfyUI RGB or RGBA IMAGE tensor.")
    pil = Image.fromarray(pixels).convert("RGB")
    pil.thumbnail((int(max_size), int(max_size)), Image.Resampling.LANCZOS)

    buffer = io.BytesIO()
    pil.save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode("ascii")


def _ask_ollama(
    *,
    url: str,
    model: str,
    question: str,
    system_prompt: str,
    image_base64: str,
    temperature: float,
    max_new_tokens: int,
    timeout_seconds: int,
) -> str:
    """One stateless /api/chat request: no earlier questions or answers included."""
    messages = []
    if system_prompt.strip():
        messages.append({"role": "system", "content": system_prompt.strip()})
    messages.append(
        {"role": "user", "content": question, "images": [image_base64]}
    )
    payload = {
        "model": model,
        "messages": messages,
        "stream": False,
        "options": {
            "temperature": float(temperature),
            "num_predict": int(max_new_tokens),
        },
    }
    request = urllib.request.Request(
        url.rstrip("/") + "/api/chat",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json", "Accept": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=int(timeout_seconds)) as response:
            result = json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")[:400]
        raise RuntimeError(f"Ollama HTTP {exc.code}: {detail or exc.reason}") from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(f"Cannot reach Ollama at {url}: {exc.reason}") from exc

    message = result.get("message") if isinstance(result, dict) else None
    answer = message.get("content") if isinstance(message, dict) else None
    if not isinstance(answer, str) or not answer.strip():
        raise RuntimeError(f"Ollama model '{model}' returned no text answer.")
    return answer.strip()


class CaptionForgeTargetedVLMQuery:
    """Ask up to ten independent image-grounded questions with a local Ollama VLM."""

    CATEGORY = "CaptionForge/Lightweight"
    FUNCTION = "query"
    RETURN_TYPES = ("IMAGE",) + ("STRING",) * MAX_SLOTS
    RETURN_NAMES = ("image",) + tuple(f"answer_{i}" for i in range(1, MAX_SLOTS + 1))
    DESCRIPTION = (
        "Ask 1-10 independent questions of an Ollama vision model, returning "
        "separate answers and the unchanged input image. Uses CaptionForge's "
        "existing Ollama model dropdown configuration."
    )

    @classmethod
    def INPUT_TYPES(cls):
        models = load_ollama_model_dropdowns(__file__)
        required = {
            "image": ("IMAGE", {"tooltip": "Image to inspect; passed to the output unchanged."}),
            "slot_count": (
                "INT",
                {"default": 3, "min": 1, "max": MAX_SLOTS, "step": 1,
                 "tooltip": "Only the first N questions are evaluated. Output pins retain fixed identities."},
            ),
            "model": (
                models["caption_models"],
                {"default": models["caption_default"],
                 "tooltip": "Installed Ollama VLM tag, from CaptionForge's shared model list."},
            ),
            "custom_model_tag": (
                "STRING",
                {"default": "", "tooltip": "Used only when model is set to custom."},
            ),
            "ollama_url": (
                "STRING",
                {"default": DEFAULT_OLLAMA_URL, "tooltip": "URL of the Ollama server."},
            ),
            "system_prompt": (
                "STRING",
                {"multiline": True, "default": DEFAULT_SYSTEM_PROMPT,
                 "tooltip": "Shared instructions applied independently to each question."},
            ),
            "temperature": (
                "FLOAT",
                {"default": 0.2, "min": 0.0, "max": 2.0, "step": 0.05,
                 "tooltip": "Ollama sampling temperature for each answer."},
            ),
            "max_new_tokens": (
                "INT",
                {"default": 256, "min": 16, "max": 4096, "step": 16,
                 "tooltip": "Maximum generated tokens per question (Ollama num_predict)."},
            ),
            "max_image_size": (
                "INT",
                {"default": 1024, "min": 256, "max": 2048, "step": 64,
                 "tooltip": "Resize longest image side for the request only; passthrough stays unchanged."},
            ),
            "timeout_seconds": (
                "INT",
                {"default": 300, "min": 10, "max": 3600, "step": 10,
                 "tooltip": "Timeout for each independent Ollama request."},
            ),
        }
        examples = (
            "Describe the outfit visible in this image.",
            "Describe the person's pose.",
            "What is the person's eye color? If unclear, say unknown.",
        )
        for i in range(1, MAX_SLOTS + 1):
            required[f"question_{i}"] = (
                "STRING",
                {"multiline": True, "default": examples[i - 1] if i <= 3 else "",
                 "tooltip": f"Question {i}; left blank to skip without shifting other answers."},
            )
        return {"required": required}

    def query(
        self,
        image,
        slot_count,
        model,
        custom_model_tag,
        ollama_url,
        system_prompt,
        temperature,
        max_new_tokens,
        max_image_size,
        timeout_seconds,
        **questions,
    ):
        selected_model = (
            custom_model_tag.strip()
            if str(model).strip().lower() == CUSTOM_VALUE
            else str(model).strip()
        )
        if not selected_model:
            raise ValueError("Select an Ollama VLM or enter a custom model tag.")
        url = str(ollama_url).strip().rstrip("/")
        if not url:
            raise ValueError("An Ollama server URL is required.")

        answers = [""] * MAX_SLOTS
        active = [
            (i, str(questions.get(f"question_{i}", "") or "").strip())
            for i in range(1, min(MAX_SLOTS, int(slot_count)) + 1)
        ]
        active = [(i, question) for i, question in active if question]
        if not active:
            return (image, *answers)

        encoded_image = _encode_image(image, max_image_size)
        for i, question in active:
            try:
                answers[i - 1] = _ask_ollama(
                    url=url,
                    model=selected_model,
                    question=question,
                    system_prompt=system_prompt,
                    image_base64=encoded_image,
                    temperature=temperature,
                    max_new_tokens=max_new_tokens,
                    timeout_seconds=timeout_seconds,
                )
            except Exception as exc:
                raise RuntimeError(
                    f"Targeted VLM Query question {i} failed ({selected_model}): {exc}"
                ) from exc
        return (image, *answers)


class CaptionForgeCaptionAssemble:
    """Order manual and connected text chunks with per-section enable switches."""

    CATEGORY = "CaptionForge/Lightweight"
    FUNCTION = "assemble"
    RETURN_TYPES = ("IMAGE", "STRING")
    RETURN_NAMES = ("image", "caption")
    DESCRIPTION = (
        "Assemble up to ten ordered caption sections. Connected STRING inputs "
        "override manual text when nonempty; disabled and empty sections are skipped. "
        "IMAGE passes through unchanged."
    )

    @classmethod
    def INPUT_TYPES(cls):
        required = {
            "image": ("IMAGE", {"tooltip": "Image passthrough, never modified."}),
            "slot_count": (
                "INT",
                {"default": 5, "min": 1, "max": MAX_SLOTS, "step": 1,
                 "tooltip": "Number of ordered sections to consider (1-10)."},
            ),
            "join_mode": (
                ["space", "newline", "double_newline", "comma_space", "custom"],
                {"default": "space", "tooltip": "Separator between nonempty enabled sections."},
            ),
            "custom_delimiter": (
                "STRING", {"default": " ", "tooltip": "Used when join_mode is custom."},
            ),
        }
        optional = {}
        for i in range(1, MAX_SLOTS + 1):
            required[f"enabled_{i}"] = (
                "BOOLEAN", {"default": True, "tooltip": f"Include section {i} in caption."}
            )
            required[f"manual_{i}"] = (
                "STRING",
                {"default": "", "multiline": True,
                 "tooltip": f"Manual section {i}; used when its STRING input is absent or empty."},
            )
            optional[f"text_{i}"] = (
                "STRING", {"tooltip": f"Optional upstream section {i}; overrides manual text if nonempty."}
            )
        return {"required": required, "optional": optional}

    @staticmethod
    def _delimiter(join_mode: str, custom_delimiter: str) -> str:
        return {
            "space": " ",
            "newline": "\n",
            "double_newline": "\n\n",
            "comma_space": ", ",
            "custom": str(custom_delimiter),
        }[join_mode]

    def assemble(self, image, slot_count, join_mode, custom_delimiter, **sections):
        parts = []
        for i in range(1, min(MAX_SLOTS, int(slot_count)) + 1):
            if not sections.get(f"enabled_{i}", True):
                continue
            upstream = sections.get(f"text_{i}")
            manual = sections.get(f"manual_{i}", "")
            upstream = "" if upstream is None else str(upstream).strip()
            manual = "" if manual is None else str(manual).strip()
            chosen = upstream if upstream else manual
            if chosen:
                parts.append(chosen)
        return (image, self._delimiter(join_mode, custom_delimiter).join(parts))


NODE_CLASS_MAPPINGS = {
    "CaptionForgeTargetedVLMQuery": CaptionForgeTargetedVLMQuery,
    "CaptionForgeCaptionAssemble": CaptionForgeCaptionAssemble,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "CaptionForgeTargetedVLMQuery": "CaptionForge Targeted VLM Query",
    "CaptionForgeCaptionAssemble": "CaptionForge Caption Assemble",
}
