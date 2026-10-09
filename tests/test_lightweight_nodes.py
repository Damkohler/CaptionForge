"""CPU-only contract tests for CaptionForge's standalone lightweight nodes."""

from __future__ import annotations

import importlib
import sys
import types
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
for name, directory in (
    ("CaptionForge", ROOT),
    ("CaptionForge.nodes", ROOT / "nodes"),
):
    if name not in sys.modules:
        module = types.ModuleType(name)
        module.__path__ = [str(directory)]
        sys.modules[name] = module

light = importlib.import_module("CaptionForge.nodes.jlc_captionforge_lightweight_nodes")


class CaptionAssemblyTests(unittest.TestCase):
    def test_overrides_fallback_disables_and_orders(self):
        image = object()
        node = light.CaptionForgeCaptionAssemble()
        result = node.assemble(
            image, 5, "space", "",
            manual_1="A portrait of",
            manual_2="a person",
            text_2="a woman",
            manual_3="in",
            enabled_3=False,
            manual_4="a red jacket.",
            text_4="   ",
            manual_5="Bright light.",
        )
        self.assertIs(result[0], image)
        self.assertEqual(result[1], "A portrait of a woman a red jacket. Bright light.")

    def test_disabled_and_inactive_sections_skip(self):
        node = light.CaptionForgeCaptionAssemble()
        result = node.assemble(
            None, 2, "newline", "",
            manual_1="alpha",
            enabled_2=False,
            manual_2="beta",
            manual_3="gamma",
        )
        self.assertEqual(result[1], "alpha")

    def test_original_text_concatenate_separator(self):
        node = light.CaptionForgeCaptionAssemble()
        self.assertEqual(node.assemble(None, 2, "double_newline", "", manual_1="a", manual_2="b")[1], "a\n\nb")

    def test_slot_inputs_have_help_and_fixed_names(self):
        spec = light.CaptionForgeCaptionAssemble.INPUT_TYPES()
        self.assertIn("text_10", spec["optional"])
        self.assertIn("manual_10", spec["required"])
        self.assertIn("enabled_10", spec["required"])
        self.assertTrue(all(len(v) > 1 and "tooltip" in v[1]
                            for side in spec.values() for v in side.values()))


class TargetedQueryTests(unittest.TestCase):
    def test_answer_alignment_and_single_encoding(self):
        image = object()
        node = light.CaptionForgeTargetedVLMQuery()
        with mock.patch.object(light, "_encode_image", return_value="fakeimage") as encode, mock.patch.object(
            light, "_ask_ollama", side_effect=["jacket", "blue eyes"]
        ) as query:
            output = node.query(
                image, 4, "gemma4:26b", "", "http://127.0.0.1:11434",
                "Be literal", 0.2, 100, 1024, 30,
                question_1="Outfit?", question_2="",
                question_3="Eyes?", question_4="",
            )
        self.assertIs(output[0], image)
        self.assertEqual(output[1:5], ("jacket", "", "blue eyes", ""))
        self.assertEqual(output[5:], ("",) * 6)
        encode.assert_called_once_with(image, 1024)
        self.assertEqual(query.call_count, 2)
        self.assertEqual(query.call_args_list[0].kwargs["question"], "Outfit?")
        self.assertEqual(query.call_args_list[1].kwargs["question"], "Eyes?")

    def test_no_active_questions_never_calls_ollama(self):
        with mock.patch.object(light, "_encode_image") as encode:
            image = object()
            outputs = light.CaptionForgeTargetedVLMQuery().query(
                image, 1, "gemma4:26b", "", "http://127.0.0.1:11434",
                "", 0.2, 100, 1024, 30, question_1=""
            )
            encode.assert_not_called()
            self.assertIs(outputs[0], image)
            self.assertEqual(outputs[1:], ("",) * 10)

    def test_query_error_propagates_instead_of_becoming_caption(self):
        with mock.patch.object(light, "_encode_image", return_value="fake"), mock.patch.object(
            light, "_ask_ollama", side_effect=RuntimeError("connection failed")
        ):
            with self.assertRaisesRegex(RuntimeError, "question 1 failed"):
                light.CaptionForgeTargetedVLMQuery().query(
                    object(), 1, "gemma4:26b", "", "http://127.0.0.1:11434",
                    "", 0.2, 100, 1024, 30, question_1="Clothing?"
                )


if __name__ == "__main__":
    unittest.main()
