"""Scoped handling of expected bitsandbytes activation-cast diagnostics."""

from contextlib import contextmanager
from importlib import metadata
import logging
import re
import threading
import warnings


_CAST_MESSAGES = frozenset(
    f"MatMul8bitLt: inputs will be cast from torch.{dtype} "
    "to float16 during quantization"
    for dtype in ("bfloat16", "float32")
)
_BNB_LOGGER = "bitsandbytes.autograd._functions"
_MINIMUM_BNB_VERSION = (0, 46, 1)


def warn_if_suspicious_8bit_stack(node_name: str) -> None:
    """Warn, without blocking inference, when bnb predates the project floor."""
    try:
        version = metadata.version("bitsandbytes")
    except Exception:
        return
    parts = tuple(int(piece) for piece in re.findall(r"\d+", version)[:3])
    normalized = parts + (0,) * (3 - len(parts))
    if normalized < _MINIMUM_BNB_VERSION:
        warnings.warn(
            f"CaptionForge's {node_name} node detected bitsandbytes {version}; "
            "this older 8-bit inference stack may cause severe slowdowns or compatibility issues. "
            "CaptionForge will continue without modifying packages.",
            RuntimeWarning,
            stacklevel=2,
        )


class _QuantizedCastLogFilter(logging.Filter):
    def __init__(self):
        super().__init__()
        self.thread_id = threading.get_ident()

    def filter(self, record):
        return not (
            record.name == _BNB_LOGGER
            and record.levelno == logging.WARNING
            and record.thread == self.thread_id
            and record.getMessage() in _CAST_MESSAGES
        )


@contextmanager
def quantized_inference_warnings(enabled: bool):
    """Preserve diagnostics except known BF16/FP32 casts during 8-bit inference.

    No filter is installed at import time or for Default mode. catch_warnings
    restores the caller's filters even when generation raises an exception.
    On Python 3.10-3.12 warning filters are process-wide during this brief scope;
    the exact message, category and module keep the suppression narrowly bounded.
    """
    if not enabled:
        yield
        return

    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message=(
                r"\AMatMul8bitLt: inputs will be cast from torch\.(?:bfloat16|float32) "
                r"to float16 during quantization\Z"
            ),
            category=UserWarning,
            module=r"\Abitsandbytes\.autograd\._functions\Z",
        )
        # Recent bitsandbytes versions use logging rather than warnings.warn.
        # Attach to the emitting logger so propagation/ComfyUI handlers remain
        # untouched. Limit this route to the current inference thread as well.
        logger = logging.getLogger(_BNB_LOGGER)
        log_filter = _QuantizedCastLogFilter()
        logger.addFilter(log_filter)
        try:
            yield
        finally:
            logger.removeFilter(log_filter)
