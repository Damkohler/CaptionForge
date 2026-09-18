"""Scoped handling of Joy's expected bitsandbytes activation-cast warning."""

from contextlib import contextmanager
import warnings


@contextmanager
def joy_8bit_inference_warnings(enabled: bool):
    """Preserve all diagnostics except the known BF16 cast during Joy 8-bit inference.

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
                r"\AMatMul8bitLt: inputs will be cast from torch\.bfloat16 "
                r"to float16 during quantization\Z"
            ),
            category=UserWarning,
            module=r"\Abitsandbytes\.autograd\._functions\Z",
        )
        yield
