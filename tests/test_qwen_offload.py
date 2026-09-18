"""Regression tests for Qwen 8-bit low-VRAM CPU offload behavior."""

from __future__ import annotations

import ast
import importlib
import sys
import tempfile
import types
import unittest
from types import SimpleNamespace
from pathlib import Path
from unittest import mock

import torch


ROOT = Path(__file__).resolve().parents[1]


def _install_namespace(name: str, path: Path) -> None:
    if name in sys.modules:
        return
    module = types.ModuleType(name)
    module.__path__ = [str(path)]
    sys.modules[name] = module


_install_namespace("CaptionForge", ROOT)
_install_namespace("CaptionForge.engines", ROOT / "engines")

qwen = importlib.import_module("CaptionForge.engines.jlc_qwen_caption_engine")


class QwenOffloadTests(unittest.TestCase):
    def test_generation_uses_scoped_bnb_diagnostic_handling(self):
        tree = ast.parse(
            (ROOT / "engines/jlc_qwen_caption_engine.py").read_text(encoding="utf-8")
        )
        engine_class = next(
            node
            for node in tree.body
            if isinstance(node, ast.ClassDef) and node.name == "QwenCaptionEngine"
        )
        caption_method = next(
            node
            for node in engine_class.body
            if isinstance(node, ast.FunctionDef) and node.name == "caption_pil"
        )

        scoped_generate_calls = []
        for node in ast.walk(caption_method):
            if not isinstance(node, ast.With):
                continue
            uses_scope = any(
                isinstance(item.context_expr, ast.Call)
                and isinstance(item.context_expr.func, ast.Name)
                and item.context_expr.func.id == "quantized_inference_warnings"
                for item in node.items
            )
            if uses_scope:
                scoped_generate_calls.extend(
                    child
                    for child in ast.walk(node)
                    if isinstance(child, ast.Call)
                    and isinstance(child.func, ast.Attribute)
                    and child.func.attr == "generate"
                )

        self.assertEqual(len(scoped_generate_calls), 1)
        self.assertNotIn("filterwarnings", ast.unparse(caption_method))

    def test_memory_budget_reserves_headroom_and_fp32_cpu_cost(self):
        gib = 1024 ** 3

        with mock.patch.object(qwen.torch.cuda, "is_available", return_value=True), \
             mock.patch.object(qwen.torch.cuda, "current_device", return_value=0), \
             mock.patch.object(qwen.torch.cuda, "mem_get_info", return_value=(10 * gib, 12 * gib)), \
             mock.patch.object(qwen, "_available_system_memory_bytes", return_value=40 * gib):
            budget = qwen._qwen_memory_budget(headroom=0.20)

        # GPU receives 80% of currently free VRAM.
        self.assertEqual(budget[0], int(10 * gib * 0.80))

        # CPU placement is estimated with dtype=int8 but actual bitsandbytes
        # CPU overflow remains FP32, so CPU capacity is divided by four.
        self.assertEqual(budget["cpu"], int(40 * gib * 0.80 / 4.0))

    def test_execution_device_prefers_accelerate_hook(self):
        model = SimpleNamespace(
            _hf_hook=SimpleNamespace(execution_device="cuda:0"),
            hf_device_map={"": "cpu"},
        )

        device = qwen._resolve_model_execution_device(model)

        self.assertEqual(device, torch.device("cuda:0"))

    def test_execution_device_prefers_gpu_from_device_map(self):
        model = SimpleNamespace(
            _hf_hook=None,
            hf_device_map={
                "visual": "cpu",
                "model.layers.0": 0,
                "model.layers.1": "cpu",
            },
        )

        device = qwen._resolve_model_execution_device(model)

        self.assertEqual(device, torch.device("cuda:0"))

    def test_execution_device_falls_back_to_parameter_device(self):
        parameter = torch.nn.Parameter(torch.zeros(1))
        model = torch.nn.Linear(1, 1)
        model.weight = parameter

        device = qwen._resolve_model_execution_device(model)

        self.assertEqual(device, parameter.device)

    def test_device_map_summary(self):
        summary = qwen._summarize_device_map(
            {
                "visual": 0,
                "layer0": 0,
                "layer1": "cpu",
            }
        )

        self.assertIn("cuda:0: 2 module(s)", summary)
        self.assertIn("cpu: 1 module(s)", summary)

    def test_load_uses_supported_8bit_cpu_offload_and_preserves_preload_eviction(self):
        events = []
        captured = {}

        class FakeBitsAndBytesConfig:
            def __init__(self, **kwargs):
                captured["bnb"] = dict(kwargs)

        class FakeProcessor:
            @classmethod
            def from_pretrained(cls, *_args, **_kwargs):
                return cls()

        class FakeModel:
            hf_device_map = {
                "visual": 0,
                "model.layers.0": 0,
                "model.layers.1": "cpu",
            }

            def eval(self):
                return self

        class FakeModelClass:
            @classmethod
            def from_pretrained(cls, *_args, **kwargs):
                events.append("load")
                captured["model_kwargs"] = dict(kwargs)
                return FakeModel()

        fake_transformers = types.ModuleType("transformers")
        fake_transformers.AutoProcessor = FakeProcessor
        fake_transformers.BitsAndBytesConfig = FakeBitsAndBytesConfig

        with tempfile.TemporaryDirectory() as temp_dir:
            local_path = Path(temp_dir)
            config = qwen.QwenCaptionConfig(
                model_path=str(local_path),
                quantization="bnb_8bit",
                device_map="auto",
                keep_loaded=True,
                allow_download=False,
            )
            engine = qwen.QwenCaptionEngine(config)

            def fake_prepare(*_args, **_kwargs):
                events.append("prepare")

            with mock.patch.dict(sys.modules, {"transformers": fake_transformers}), \
                 mock.patch.object(qwen, "get_cached_model", return_value=None), \
                 mock.patch.object(qwen, "prepare_for_model_load", side_effect=fake_prepare), \
                 mock.patch.object(qwen, "register_model"), \
                 mock.patch.object(engine, "_load_model_class", return_value=FakeModelClass), \
                 mock.patch.object(engine, "_detect_model_type", return_value="qwen2_5_vl"), \
                 mock.patch.object(
                     qwen,
                     "_build_qwen_8bit_device_map",
                     return_value=(
                         {
                             "visual": 0,
                             "model.layers.0": 0,
                             "model.layers.1": "cpu",
                         },
                         {0: 8 * 1024**3, "cpu": 8 * 1024**3},
                     ),
                 ):
                engine.load()

        self.assertEqual(events[:2], ["prepare", "load"])
        self.assertTrue(captured["bnb"]["load_in_8bit"])
        self.assertTrue(
            captured["bnb"]["llm_int8_enable_fp32_cpu_offload"]
        )
        self.assertEqual(
            captured["model_kwargs"]["device_map"]["model.layers.1"],
            "cpu",
        )
        self.assertIn("max_memory", captured["model_kwargs"])

    def test_adaptive_mapper_rejects_disk_spill(self):
        class FakeModel:
            _no_split_modules = []

        class FakeModelClass:
            def __new__(cls, *_args, **_kwargs):
                return FakeModel()

        fake_accelerate = SimpleNamespace(
            init_empty_weights=mock.MagicMock(),
            infer_auto_device_map=mock.MagicMock(
                return_value={
                    "visual": 0,
                    "model.layers.0": "cpu",
                    "model.layers.1": "disk",
                }
            ),
        )

        class FakeContext:
            def __enter__(self):
                return None

            def __exit__(self, exc_type, exc, tb):
                return False

        fake_accelerate.init_empty_weights.return_value = FakeContext()

        fake_transformers = SimpleNamespace(
            AutoConfig=SimpleNamespace(
                from_pretrained=mock.MagicMock(return_value=object())
            )
        )

        modules = {
            "accelerate": fake_accelerate,
            "transformers": fake_transformers,
        }

        with mock.patch.dict("sys.modules", modules), \
             mock.patch.object(qwen.torch.cuda, "is_available", return_value=True), \
             mock.patch.object(qwen, "_qwen_memory_budget", return_value={0: 1, "cpu": 1}):
            with self.assertRaisesRegex(RuntimeError, "disk offload"):
                qwen._build_qwen_8bit_device_map(
                    FakeModelClass,
                    qwen.Path("."),
                    True,
                )


if __name__ == "__main__":
    unittest.main()
