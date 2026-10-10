"""Standalone lightweight visual Q&A and caption assembly nodes for CaptionForge.

These nodes reuse CaptionForge's Ollama model choices without entering its
multi-pass caption pipeline. Slots are fixed in Python so that saved wires
remain stable when the active slot count changes.
"""

from __future__ import annotations

import base64
import hashlib
import io
import json
import urllib.error
import urllib.request
from collections import OrderedDict
from threading import RLock

from .captionforge_ollama_model_dropdowns import CUSTOM_VALUE, load_ollama_model_dropdowns

MAX_SLOTS = 10
SLOT_MODES = ("first_n", "all_nonempty")
DEFAULT_OLLAMA_URL = "http://127.0.0.1:11434"
MAX_SEED = 2_147_483_647
MAX_RETRY_TOKENS = 8192
SUCCESS_CACHE_LIMIT = 128

# In-memory, bounded, per-question cache. Only successful text is retained.
# This cache is independent of ComfyUI's node-level execution cache.
_SUCCESS_CACHE: OrderedDict[tuple, str] = OrderedDict()
_SUCCESS_CACHE_LOCK = RLock()


class OllamaEmptyAnswerError(RuntimeError):
    """A complete HTTP response did not contain a final text answer."""


def _slot_indices(slot_mode: str, slot_count: int) -> range:
    """Return stable slot numbers without compacting or reindexing sockets."""
    if slot_mode not in SLOT_MODES:
        raise ValueError(f"Unknown slot_mode: {slot_mode!r}. Expected one of {SLOT_MODES}.")
    if slot_mode == "all_nonempty":
        return range(1, MAX_SLOTS + 1)
    return range(1, min(MAX_SLOTS, max(0, int(slot_count))) + 1)


def _cache_get(key: tuple) -> str | None:
    with _SUCCESS_CACHE_LOCK:
        value = _SUCCESS_CACHE.get(key)
        if value is not None:
            _SUCCESS_CACHE.move_to_end(key)
        return value


def _cache_put(key: tuple, answer: str) -> None:
    with _SUCCESS_CACHE_LOCK:
        _SUCCESS_CACHE[key] = answer
        _SUCCESS_CACHE.move_to_end(key)
        while len(_SUCCESS_CACHE) > SUCCESS_CACHE_LIMIT:
            _SUCCESS_CACHE.popitem(last=False)

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
    seed: int = -1,
    thinking_mode: str = "model_default",
    diagnostics: str = "errors",
) -> str:
    """One stateless Ollama /api/chat request, returning only final-answer text.

    Thinking content is never returned as a caption or logged. Metadata may be
    logged to explain why a model exhausted its generation budget.
    """
    messages = []
    if system_prompt.strip():
        messages.append({"role": "system", "content": system_prompt.strip()})
    messages.append({"role": "user", "content": question, "images": [image_base64]})
    options = {"temperature": float(temperature), "num_predict": int(max_new_tokens)}
    if int(seed) >= 0:
        options["seed"] = int(seed)
    payload = {
        "model": model,
        "messages": messages,
        "stream": False,
        "options": options,
    }
    if thinking_mode != "model_default":
        payload["think"] = thinking_mode == "enabled"

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
    thinking = message.get("thinking") if isinstance(message, dict) else None
    metadata = (
        f"done_reason={result.get('done_reason')!r}, "
        f"eval_count={result.get('eval_count')!r}, "
        f"prompt_eval_count={result.get('prompt_eval_count')!r}, "
        f"thinking_chars={len(thinking) if isinstance(thinking, str) else 0}, "
        f"num_predict={max_new_tokens}"
    ) if isinstance(result, dict) else "response was not a JSON object"
    if not isinstance(answer, str) or not answer.strip():
        raise OllamaEmptyAnswerError(
            f"Ollama model '{model}' returned no final text answer ({metadata})."
        )
    if diagnostics == "verbose":
        print(f"[CaptionForge Targeted VLM] Ollama reply: {metadata}", flush=True)
    return answer.strip()


class CaptionForgeTargetedVLMQuery:
    """Ask up to ten independent image-grounded questions with a local Ollama VLM."""

    CATEGORY = "CaptionForge/Lightweight"
    FUNCTION = "query"
    RETURN_TYPES = ("IMAGE",) + ("STRING",) * MAX_SLOTS
    RETURN_NAMES = ("image",) + tuple(f"answer_{i}" for i in range(1, MAX_SLOTS + 1))
    DESCRIPTION = (
        "Ask filled image-grounded questions using first-N or all-nonempty slots, returning "
        "separate answers and the unchanged input image. Uses CaptionForge's "
        "existing Ollama model dropdown configuration."
    )

    @classmethod
    def IS_CHANGED(cls, force_regenerate=False, **_inputs):
        # A NaN sentinel opts out of ComfyUI's node-level execution cache.
        # Keep it off by default so unchanged workflows remain fast.
        return float("nan") if force_regenerate else None

    @classmethod
    def INPUT_TYPES(cls):
        models = load_ollama_model_dropdowns(__file__)
        required = {
            "image": ("IMAGE", {"tooltip": "Image to inspect; passed to the output unchanged."}),
            "slot_count": (
                "INT",
                {"default": 3, "min": 1, "max": MAX_SLOTS, "step": 1,
                 "tooltip": "In first_n mode, examine questions 1 through N. Ignored by all_nonempty mode. Pins retain their numbers."},
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

        # Append rather than insert: saved workflows serialize widget values in
        # order. Inserting above question_1 would remap existing saved questions.
        required.update({
            "seed": (
                "INT",
                {"default": -1, "min": -1, "max": MAX_SEED, "step": 1,
                 "control_after_generate": True,
                 "tooltip": "-1 leaves Ollama unseeded. Nonnegative seeds are reproducible when the backend permits. ComfyUI may offer seed controls."},
            ),
            "retry_count": (
                "INT",
                {"default": 1, "min": 0, "max": 3, "step": 1,
                 "tooltip": "Additional tries when Ollama returns an empty final answer. Each retry increases the token budget, up to 8192."},
            ),
            "retry_seed_mode": (
                ["increment", "same"],
                {"default": "increment",
                 "tooltip": "For explicit seeds, increment seed on an empty-answer retry or reuse it. -1 stays unseeded."},
            ),
            "error_policy": (
                ["skip_failed", "insert_unknown", "stop_on_error"],
                {"default": "skip_failed",
                 "tooltip": "After retries, omit a failed answer, insert 'unknown', or stop the workflow. Errors never become caption text."},
            ),
            "thinking_mode": (
                ["model_default", "disabled", "enabled"],
                {"default": "model_default",
                 "tooltip": "Controls Ollama's top-level think flag. Model-default omits the flag for compatibility."},
            ),
            "diagnostics": (
                ["errors", "verbose", "off"],
                {"default": "errors",
                 "tooltip": "Console diagnostics; metadata only, never the model's internal thinking text."},
            ),
            "cache_successful_answers": (
                "BOOLEAN",
                {"default": True,
                 "tooltip": "Reuse successful answers per image, question, model and settings during this ComfyUI server session."},
            ),
            "force_regenerate": (
                "BOOLEAN",
                {"default": False,
                 "tooltip": "When true, bypass ComfyUI's node cache and the per-question answer cache on every queue run."},
            ),
            "slot_mode": (
                list(SLOT_MODES),
                {"default": "first_n",
                 "tooltip": "first_n: slots 1 through slot_count. all_nonempty: all ten slots, skipping blank questions. Answers never shift pins."},
            ),
        })
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
        seed=-1,
        retry_count=1,
        retry_seed_mode="increment",
        error_policy="skip_failed",
        thinking_mode="model_default",
        diagnostics="errors",
        cache_successful_answers=True,
        force_regenerate=False,
        slot_mode="first_n",
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

        seed = int(seed)
        if not -1 <= seed <= MAX_SEED:
            raise ValueError(f"Seed must be between -1 and {MAX_SEED}.")
        if retry_seed_mode not in ("increment", "same"):
            raise ValueError(f"Unknown retry_seed_mode: {retry_seed_mode}")
        if error_policy not in ("skip_failed", "insert_unknown", "stop_on_error"):
            raise ValueError(f"Unknown error_policy: {error_policy}")
        if thinking_mode not in ("model_default", "disabled", "enabled"):
            raise ValueError(f"Unknown thinking_mode: {thinking_mode}")

        answers = [""] * MAX_SLOTS
        active = [
            (i, str(questions.get(f"question_{i}", "") or "").strip())
            for i in _slot_indices(slot_mode, slot_count)
        ]
        active = [(i, question) for i, question in active if question]
        if not active:
            return (image, *answers)

        encoded_image = _encode_image(image, max_image_size)
        image_digest = hashlib.sha256(encoded_image.encode("ascii")).hexdigest()
        for i, question in active:
            cache_key = (
                image_digest, url, selected_model, question, str(system_prompt),
                float(temperature), int(max_new_tokens), int(max_image_size),
                seed, int(retry_count), retry_seed_mode, thinking_mode,
            )
            if cache_successful_answers and not force_regenerate:
                cached = _cache_get(cache_key)
                if cached is not None:
                    answers[i - 1] = cached
                    if diagnostics == "verbose":
                        print(f"[CaptionForge Targeted VLM] Question {i}: cached answer.", flush=True)
                    continue

            last_error = None
            for attempt in range(max(0, int(retry_count)) + 1):
                attempt_seed = (
                    ((seed + attempt) % (MAX_SEED + 1))
                    if seed >= 0 and retry_seed_mode == "increment"
                    else seed
                )
                # Thinking models can use the entire small token budget without
                # producing a final response. Increase only for empty-answer retries.
                attempt_tokens = min(int(max_new_tokens) * (attempt + 1), MAX_RETRY_TOKENS)
                try:
                    answer = _ask_ollama(
                        url=url,
                        model=selected_model,
                        question=question,
                        system_prompt=system_prompt,
                        image_base64=encoded_image,
                        temperature=temperature,
                        max_new_tokens=attempt_tokens,
                        timeout_seconds=timeout_seconds,
                        seed=attempt_seed,
                        thinking_mode=thinking_mode,
                        diagnostics=diagnostics,
                    )
                    answers[i - 1] = answer
                    if cache_successful_answers:
                        _cache_put(cache_key, answer)
                    last_error = None
                    break
                except OllamaEmptyAnswerError as exc:
                    last_error = exc
                    if diagnostics != "off":
                        action = "retrying" if attempt < int(retry_count) else "exhausted"
                        print(
                            f"[CaptionForge Targeted VLM] Question {i}, attempt "
                            f"{attempt + 1}/{int(retry_count) + 1}: {exc} [{action}]",
                            flush=True,
                        )
                except Exception as exc:
                    # Fail fast on configuration, HTTP, and transport errors;
                    # extra attempts are reserved for complete-but-empty replies.
                    last_error = exc
                    break

            if last_error is not None:
                message = (
                    f"Targeted VLM Query question {i} failed ({selected_model}): {last_error}"
                )
                if error_policy == "stop_on_error":
                    raise RuntimeError(message) from last_error
                # Even with diagnostics='off', make degraded captions discoverable.
                print(f"[CaptionForge Targeted VLM] WARNING: {message}", flush=True)
                answers[i - 1] = "unknown" if error_policy == "insert_unknown" else ""

        return (image, *answers)

class CaptionForgeCaptionAssemble:
    """Order manual and connected text chunks with per-section enable switches."""

    CATEGORY = "CaptionForge/Lightweight"
    FUNCTION = "assemble"
    RETURN_TYPES = ("IMAGE", "STRING")
    RETURN_NAMES = ("image", "caption")
    DESCRIPTION = (
        "Assemble sections using first-N or all-nonempty slots. Connected STRING inputs "
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
                 "tooltip": "In first_n mode, examine sections 1 through N. Ignored by all_nonempty mode."},
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

        # Append after optional text_1...text_10, which ComfyUI also stores
        # as widgets when unconnected. Putting this in 'required' would shift
        # those persisted widget values in existing saved workflows.
        optional["slot_mode"] = (
            list(SLOT_MODES),
            {"default": "first_n",
             "tooltip": "first_n: sections 1 through slot_count. all_nonempty: all ten slots, using enabled nonblank sections in original order."},
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

    def assemble(self, image, slot_count, join_mode, custom_delimiter, slot_mode="first_n", **sections):
        parts = []
        for i in _slot_indices(slot_mode, slot_count):
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
