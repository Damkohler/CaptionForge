"""Regression tests for boundary-safe Pass-A forbidden-phrase cleanup."""

from __future__ import annotations

import importlib
import sys
import types
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def _install_namespace(name: str, path: Path) -> None:
    if name in sys.modules:
        return
    module = types.ModuleType(name)
    module.__path__ = [str(path)]
    sys.modules[name] = module


_install_namespace("CaptionForge", ROOT)
_install_namespace("CaptionForge.engines", ROOT / "engines")
_install_namespace("CaptionForge.nodes", ROOT / "nodes")
_install_namespace("CaptionForge.nodes.caption_nodes", ROOT / "nodes" / "caption_nodes")

if "folder_paths" not in sys.modules:
    folder_paths = types.ModuleType("folder_paths")
    folder_paths.models_dir = str(ROOT / "models")
    folder_paths.get_output_directory = lambda: str(ROOT / "output")
    sys.modules["folder_paths"] = folder_paths

cleanup = importlib.import_module("CaptionForge.engines.captionforge_cleanup")
planner_engine = importlib.import_module("CaptionForge.engines.captionforge_pipeline_planner_engine")
joy = importlib.import_module("CaptionForge.engines.jlc_joy_caption_engine")
qwen = importlib.import_module("CaptionForge.engines.jlc_qwen_caption_engine")
ollama = importlib.import_module(
    "CaptionForge.nodes.caption_nodes.jlc_captionforge_ollama_caption_node"
)


class SharedForbiddenPhraseContractTests(unittest.TestCase):
    def test_pipeline_cleanup_settings_propagate_and_override_standalone_values(self) -> None:
        plan = planner_engine.build_captionforge_pipeline_plan(
            forbidden_phrases="old\nsafety disclaimer",
            replace_pairs="former=>current\nred car=>blue car",
        )
        self.assertEqual(plan["cleanup"]["forbidden_phrases"], ["old", "safety disclaimer"])
        self.assertEqual(
            plan["cleanup"]["replace_pairs"],
            [{"old": "former", "new": "current"}, {"old": "red car", "new": "blue car"}],
        )
        forbidden, pairs = cleanup.resolve_cleanup_settings(
            plan, "standalone forbidden", "standalone old=>standalone new"
        )
        self.assertEqual(forbidden, ["old", "safety disclaimer"])
        self.assertEqual(pairs, [("former", "current"), ("red car", "blue car")])

    def test_standalone_cleanup_settings_survive_without_planner(self) -> None:
        forbidden, pairs = cleanup.resolve_cleanup_settings({}, "old", "former=>current")
        self.assertEqual(forbidden, ["old"])
        self.assertEqual(pairs, [("former", "current")])

    def test_end_to_end_cleanup_keeps_boundaries_and_normalizes_spacing(self) -> None:
        value = cleanup.apply_cleanup_contract(
            "bold holding gold, old, former wording.", ["old"], [("former", "current")]
        )
        self.assertEqual(value, "bold holding gold, current wording.")

    def test_substrings_inside_legitimate_words_do_not_match(self) -> None:
        text = "A bold subject is holding a gold accessory."
        self.assertFalse(cleanup.contains_forbidden_phrase(text, ["old"]))
        self.assertEqual(cleanup.remove_forbidden_phrases(text, ["old"]), text)

    def test_boundary_safe_replacements_preserve_containing_words(self) -> None:
        source = "bold pose, holding a gold prop, old stone wall"
        expected = "bold pose, holding a gold prop, young stone wall"
        self.assertEqual(
            cleanup.replace_phrases(source, [("old", "young")]),
            expected,
        )

    def test_true_word_and_phrase_matches_are_boundary_aware(self) -> None:
        self.assertTrue(cleanup.contains_forbidden_phrase("an old stone wall", ["old"]))
        self.assertTrue(
            cleanup.contains_forbidden_phrase(
                "caption includes safety disclaimer text",
                ["safety disclaimer"],
            )
        )
        self.assertFalse(
            cleanup.contains_forbidden_phrase(
                "caption includes safety disclaimers",
                ["safety disclaimer"],
            )
        )

    def test_joy_and_qwen_replace_pairs_are_boundary_safe_by_default(self) -> None:
        source = "bold pose, holding a gold prop, old stone wall"
        expected = "bold pose, holding a gold prop, young stone wall"
        self.assertTrue(joy.CleanupConfig().replace_whole_words_only)
        self.assertTrue(qwen.CleanupConfig().replace_whole_words_only)
        self.assertEqual(
            joy.apply_replacements(source, [("old", "young")]),
            expected,
        )
        self.assertEqual(
            qwen.apply_replacements(source, [("old", "young")]),
            expected,
        )

    def test_joy_and_qwen_preserve_containing_words_but_remove_true_match(self) -> None:
        source = "bold pose, holding a gold prop, old stone wall"
        expected = "bold pose, holding a gold prop, stone wall"
        self.assertEqual(joy.remove_forbidden_phrases(source, ["old"]), expected)
        self.assertEqual(qwen.remove_forbidden_phrases(source, ["old"]), expected)

    def test_ollama_does_not_drop_paragraph_for_substring_false_positive(self) -> None:
        raw = "A bold figure is holding a gold accessory in dramatic light."
        cleaned, status = ollama._clean_caption(
            raw,
            trigger_word="",
            forbidden_phrases=["old"],
            replacement_rules=[],
        )
        self.assertEqual(status, "ok")
        self.assertEqual(cleaned, raw)

    def test_ollama_replace_pairs_are_boundary_safe(self) -> None:
        raw = "A bold figure is holding a gold prop near an old wall."
        cleaned, status = ollama._clean_caption(
            raw,
            trigger_word="",
            forbidden_phrases=[],
            replacement_rules=[("old", "young")],
        )
        self.assertEqual(status, "ok")
        self.assertEqual(
            cleaned,
            "A bold figure is holding a gold prop near an young wall.",
        )

    def test_ollama_replacement_case_sensitivity_is_preserved(self) -> None:
        raw = "Old wall beside an old wall."
        cleaned, status = ollama._clean_caption(
            raw,
            trigger_word="",
            forbidden_phrases=[],
            replacement_rules=[("old", "young")],
        )
        self.assertEqual(status, "ok")
        self.assertEqual(cleaned, "Old wall beside an young wall.")

    def test_ollama_still_drops_line_for_true_forbidden_match(self) -> None:
        raw = "First safe line.\nAn old line.\nFinal safe line."
        cleaned, status = ollama._clean_caption(
            raw,
            trigger_word="",
            forbidden_phrases=["old"],
            replacement_rules=[],
        )
        self.assertEqual(status, "ok")
        self.assertEqual(cleaned, "First safe line.\nFinal safe line.")

    def test_ollama_run_config_audits_cleanup_settings(self) -> None:
        config = ollama._build_run_config(
            model_tag="example:model",
            ollama_url="http://127.0.0.1:11434",
            system_prompt="system",
            prompt="prompt",
            max_new_tokens=100,
            temperature=0.2,
            top_p=0.9,
            top_k=40,
            repetition_penalty=1.03,
            max_size=1024,
            forbidden_phrases=["old", "safety disclaimer"],
            replacement_rules=[("foo", "bar")],
        )
        self.assertEqual(
            config["cleanup"]["forbidden_phrases"],
            ["old", "safety disclaimer"],
        )
        self.assertEqual(config["cleanup"]["replacement_rules"], [["foo", "bar"]])
        self.assertEqual(
            config["cleanup"]["replacement_match_mode"],
            "whole_word_or_phrase_boundary",
        )
        self.assertFalse(config["cleanup"]["replacement_case_insensitive"])
        self.assertEqual(
            config["cleanup"]["forbidden_match_mode"],
            "whole_word_or_phrase_boundary",
        )
        self.assertEqual(config["cleanup"]["forbidden_action"], "drop_matching_line")

    def test_test01_style_seven_witnesses_all_remain_eligible(self) -> None:
        """Reproduce the Test_01 false-positive signature across seven Pass-A witnesses."""
        witnesses = [
            ("joy", "A subject is holding a prop with both hands."),
            ("joy", "A bold composition with detailed clothing."),
            ("qwen", "The subject is holding an accessory against a plain background."),
            ("ollama", "A bold figure is holding a gold accessory in dramatic light."),
            ("ollama", "The subject is holding a pose with bold styling."),
            ("ollama", "A gold ornament is visible while the subject is holding the garment."),
            ("ollama", "A detailed portrait with a confident pose and studio lighting."),
        ]

        survivors = []
        for family, caption in witnesses:
            if family == "joy":
                cleaned = joy.remove_forbidden_phrases(caption, ["old"])
                status = "ok" if cleaned else "filtered"
            elif family == "qwen":
                cleaned = qwen.remove_forbidden_phrases(caption, ["old"])
                status = "ok" if cleaned else "filtered"
            else:
                cleaned, status = ollama._clean_caption(
                    caption,
                    trigger_word="",
                    forbidden_phrases=["old"],
                    replacement_rules=[],
                )
            if status == "ok" and cleaned:
                survivors.append((family, cleaned))

        self.assertEqual(len(survivors), 7)


if __name__ == "__main__":
    unittest.main()
