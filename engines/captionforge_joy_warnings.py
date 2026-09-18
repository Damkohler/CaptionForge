"""Scoped handling of Joy's expected bitsandbytes activation-cast warning."""

from contextlib import contextmanager
import logging
import threading
import warnings


_CAST_MESSAGE = (
    "MatMul8bitLt: inputs will be cast from torch.bfloat16 "
    "to float16 during quantization"
)
_BNB_LOGGER = "bitsandbytes.autograd._functions"


class _JoyCastLogFilter(logging.Filter):
    def __init__(self):
        super().__init__()
        self.thread_id = threading.get_ident()

    def filter(self, record):
        return not (
            record.name == _BNB_LOGGER
            and record.levelno == logging.WARNING
            and record.thread == self.thread_id
            and record.getMessage() == _CAST_MESSAGE
        )


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
        # Recent bitsandbytes versions use logging rather than warnings.warn.
        # Attach to the emitting logger so propagation/ComfyUI handlers remain
        # untouched. Limit this route to the current inference thread as well.
        logger = logging.getLogger(_BNB_LOGGER)
        log_filter = _JoyCastLogFilter()
        logger.addFilter(log_filter)
        try:
            yield
        finally:
            logger.removeFilter(log_filter)
