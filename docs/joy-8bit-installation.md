# Joy 8-bit dtype and clean installation

## Why Balanced (8-bit) retains BF16

`Balanced (8-bit)::bf16` describes quantized weights plus Joy's floating-point
configuration, not an all-INT8 computation. Joy's quantized loader uses
`torch_dtype="auto"` (the checkpoint dtype), excludes the vision tower and
multimodal projector from INT8 conversion, and uses BF16 autocast where supported.
The configuration dtype is also part of the cache key; it does not override
`auto` in the quantized loader. This existing behavior is unchanged.

JoyCaption's upstream ComfyUI implementation uses the same `auto` loading and
BF16 autocast. In bitsandbytes 0.46.1, `MatMul8bitLt.forward` explicitly casts
activations to FP16 for INT8 quantization, while retaining the input dtype for
other computation/output handling. This is an expected internal conversion,
not sufficient evidence that the entire Joy model should switch to FP16.
Changing the vision tower, projector and surrounding floating-point computation
just to remove the warning would change numerical behavior without a quality
validation basis.

References:

- [JoyCaption native BF16 usage](https://github.com/fpgaminer/joycaption/blob/main/README.md)
- [Upstream JoyCaption ComfyUI implementation](https://github.com/fpgaminer/joycaption_comfyui/blob/main/nodes.py)
- [bitsandbytes 0.46.1 implementation](https://github.com/bitsandbytes-foundation/bitsandbytes/blob/0.46.1/bitsandbytes/autograd/_functions.py)

## Warning scope

The old import-time filters were removed. During Balanced (8-bit) generation
only, CaptionForge ignores this exact `UserWarning` or WARNING log record from
`bitsandbytes.autograd._functions`:

```text
MatMul8bitLt: inputs will be cast from torch.bfloat16 to float16 during quantization
```

FP32 cast warnings, different messages, other modules, other warning categories,
processor/loading/cleanup diagnostics and Default-mode warnings remain visible.
The caller's filters are restored even if generation fails. Recent bitsandbytes versions emit this through `logger.warning` instead of
`warnings.warn`; both routes are covered. The logging filter matches the fully
formatted message and originating logger, and applies only to the inference
thread. It is removed on normal exit or failure, without changing logger levels,
handlers, propagation, or pre-existing filters. This does not alter tensors,
quantization parameters or model outputs.

Python 3.10-3.12 warning filters are process-wide while a `catch_warnings` scope
is active. An identical warning from concurrent bitsandbytes work could therefore
also be suppressed during that interval. This is scoped filtering, not a claim
of thread-local warning isolation.

## Installation declarations

`accelerate` was already required by `pyproject.toml`; declaring it there alone
did not cover Manager's requirements-file installation path. CaptionForge now
ships `requirements.txt` with the same runtime dependencies as
`project.dependencies`, including `accelerate` and `bitsandbytes>=0.46.1`.
The `quantization` extra is retained for compatibility with existing install
commands, but bitsandbytes no longer requires opting into an extra.

[ComfyUI Manager's installation code](https://github.com/Comfy-Org/ComfyUI-Manager/blob/main/glob/manager_core.py)
reads `requirements.txt` and installs its entries. Both declarations are kept
explicit so Manager can process individual requirement lines. A regression test
prevents the lists from drifting. The lower bound does not guarantee compatibility
with every future torch/CUDA/bitsandbytes combination.

For an existing installation, run this from CaptionForge's directory using
**the Python interpreter belonging to ComfyUI Desktop's environment**, then
restart ComfyUI:

```powershell
python -m pip install -r requirements.txt
python -m pip check
```

This affects new Registry installs after a release containing these changes is
published; editing the repository does not update an already-published archive.

## Validation

CPU-only focused checks (Python 3.11/3.12; Python 3.10 also needs `tomli`):

```text
python -m unittest discover -s tests -p test_joy_hardening.py -v
```

They cover six repeated caption bursts through both warnings and logging, near-match diagnostics,
Default mode, filter restoration, exceptions, and matching dependency lists.
Generation tests execute the real `caption_pil` method with test doubles for
processor, model and torch; they do not test numerical inference.

Before releasing, validate on a clean supported ComfyUI Desktop environment:

1. Install the updated package through Manager without manually installing
   accelerate/bitsandbytes. Confirm both are present using that environment's
   `python -m pip show accelerate bitsandbytes`, then run `python -m pip check`.
2. Run Joy Balanced (8-bit), two runs over three ordinary test images. Confirm
   six successful captions, normal progress/error diagnostics, and no BF16 cast
   warning bursts. Reuse the loaded model for a subsequent run as well.
3. Check Default mode if sufficient VRAM is available. No warning policy or
   numerical settings should change there.

A real clean Desktop install and CUDA caption generation cannot be established
by the CPU-only checks; they remain release validation steps.
