from __future__ import annotations

import importlib
import json
import sys
import tempfile
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


# Import submodules without executing CaptionForge/__init__.py, whose purpose is
# full ComfyUI node registration rather than unit-test setup.
_install_namespace("CaptionForge", ROOT)
_install_namespace("CaptionForge.engines", ROOT / "engines")
_install_namespace("CaptionForge.nodes", ROOT / "nodes")

planner_engine = importlib.import_module("CaptionForge.engines.captionforge_pipeline_planner_engine")
planner_node = importlib.import_module("CaptionForge.nodes.jlc_captionforge_pipeline_planner_node")
capstone = importlib.import_module("CaptionForge.nodes.jlc_captionforge_node")


class PlannerContractTests(unittest.TestCase):
    def test_validated_workflow_values_keep_the_same_effective_contract(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            _, plan, _ = planner_node.JLC_CaptionForge_Pipeline_Planner().plan(
                **{
                    "Output - folder": temp_dir,
                    "Output - overwrite outputs": False,
                    "Caption - Joy runs/image": "1",
                    "Caption - Qwen runs/image": "1",
                    "Caption - Ollama runs/image": "Disabled",
                    "Caption - base seed": 3,
                    "Caption - seed mode": "random",
                    "Caption - temperature schedule": "0.80, 0.90",
                    "Caption - top p schedule": "0.70, 0.60",
                    "Caption - top k schedule": "70, 80",
                    "Caption - max image size": 1024,
                    # Existing workflow value; effective runtime was and remains 4096.
                    "Caption - max new tokens": 6000,
                    "Distiller - model": "mistral-small:24b",
                    "Distiller - base seed": 3,
                    "Distiller - seed mode": "random",
                    "Distiller - max caption chars for LLM": 1536,
                    "Distiller - num predict": 3096,
                    "Distiller - temperature": 0.24,
                    "Distiller - top p": 0.90,
                    "Distiller - top k": 60,
                    "Distiller - write prompt JSONL": False,
                    "Distiller - preserve raw response": False,
                    "Validator - model": "gemma4:26b",
                    "Validator - base seed": 3,
                    "Validator - seed mode": "random",
                    "Validator - num predict": 2200,
                    "Validator - temperature": 0.0,
                    "Validator - top p": 0.92,
                    "Validator - top k": 80,
                    "Validator - write prompt JSONL": False,
                    "Validator - preserve raw VLM response": False,
                    "Final - write TXT sidecars": True,
                    "Final - write JSONL": True,
                }
            )

        joy_runs = planner_engine.expand_captionforge_runs(plan, model_key="joy")
        qwen_runs = planner_engine.expand_captionforge_runs(plan, model_key="qwen")
        ollama_runs = planner_engine.expand_captionforge_runs(plan, model_key="ollama")
        self.assertEqual((len(joy_runs), len(qwen_runs), len(ollama_runs)), (1, 1, 0))
        for run in joy_runs + qwen_runs:
            self.assertEqual(run.temperature, 0.80)
            self.assertEqual(run.top_p, 0.70)
            self.assertEqual(run.top_k, 70)
            self.assertEqual(run.max_new_tokens, 4096)
            self.assertEqual(run.max_size, 1024)

        self.assertEqual(plan["distiller"]["model"], "mistral-small:24b")
        self.assertEqual(plan["distiller"]["num_predict"], 3096)
        self.assertEqual(plan["distiller"]["temperature"], 0.24)
        self.assertEqual(plan["distiller"]["top_p"], 0.90)
        self.assertEqual(plan["distiller"]["top_k"], 60)
        self.assertEqual(plan["validator"]["model"], "gemma4:26b")
        self.assertEqual(plan["validator"]["num_predict"], 2200)
        self.assertEqual(plan["validator"]["temperature"], 0.0)
        self.assertEqual(plan["validator"]["top_p"], 0.92)
        self.assertEqual(plan["validator"]["top_k"], 80)
        self.assertEqual(capstone._resolve_fat_draft_max_caption_chars(plan, 1536), 1536)
        self.assertFalse(any(capstone._resolve_stage_audit_settings(plan, False, False).values()))
        self.assertTrue(plan["final"]["write_txt_sidecars"])
        self.assertTrue(plan["final"]["write_jsonl"])

    def test_planner_zero_values_survive_and_removed_controls_are_absent(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            _, plan, _ = planner_node.JLC_CaptionForge_Pipeline_Planner().plan(
                **{
                    "Output - folder": temp_dir,
                    "Output - overwrite outputs": False,
                    "Caption - Joy runs/image": "1",
                    "Caption - Qwen runs/image": "Disabled",
                    "Caption - Ollama runs/image": "Disabled",
                    "Caption - base seed": 0,
                    "Caption - max image size": 0,
                    "Caption - max new tokens": 9999,
                    "Distiller - base seed": 0,
                    "Distiller - max caption chars for LLM": 0,
                    "Distiller - top p": 0,
                    "Distiller - top k": 0,
                    "Validator - base seed": 0,
                    "Validator - top p": 0,
                    "Validator - top k": 0,
                    # Old serialized inputs are accepted as harmless extras.
                    "Distiller - strategy": "by_model_then_global",
                    "Final - caption style": "comma",
                }
            )

        self.assertEqual(plan["shared"]["base_seed"], 0)
        self.assertEqual(plan["shared"]["max_size"], 0)
        self.assertEqual(plan["shared"]["max_new_tokens"], 4096)
        self.assertEqual(plan["distiller"]["base_seed"], 0)
        self.assertEqual(plan["distiller"]["max_caption_chars_for_llm"], 0)
        self.assertEqual(plan["distiller"]["top_p"], 0.0)
        self.assertEqual(plan["distiller"]["top_k"], 0)
        self.assertEqual(plan["validator"]["base_seed"], 0)
        self.assertEqual(plan["validator"]["top_p"], 0.0)
        self.assertEqual(plan["validator"]["top_k"], 0)
        self.assertNotIn("strategy", plan["distiller"])
        self.assertNotIn("caption_style", plan["final"])

    def test_removed_widgets_and_pass_a_token_contract(self) -> None:
        required = planner_node.JLC_CaptionForge_Pipeline_Planner.INPUT_TYPES()["required"]
        self.assertNotIn("Distiller - strategy", required)
        self.assertNotIn("Final - caption style", required)
        token_spec = required["Caption - max new tokens"][1]
        self.assertEqual(token_spec["default"], 4096)
        self.assertEqual(token_spec["max"], 4096)

        default_plan = planner_engine.build_captionforge_pipeline_plan(
            output_dir="unused",
            joy_runs_per_image=1,
            qwen_runs_per_image=0,
        )
        clamped_plan = planner_engine.build_captionforge_pipeline_plan(
            output_dir="unused",
            joy_runs_per_image=1,
            qwen_runs_per_image=0,
            max_new_tokens=12000,
        )
        self.assertEqual(default_plan["shared"]["max_new_tokens"], 4096)
        self.assertEqual(clamped_plan["shared"]["max_new_tokens"], 4096)


class CapstoneResolutionTests(unittest.TestCase):
    def test_normalization_preserves_nested_zero_values(self) -> None:
        normalized = planner_engine.normalize_captionforge_pipeline_plan(
            json.dumps(
                {
                    "captionforge_config_type": "captionforge_pipeline_plan",
                    "distiller": {"base_seed": 0, "top_p": 0, "top_k": 0},
                }
            )
        )
        self.assertEqual(normalized["distiller"], {"base_seed": 0, "top_p": 0, "top_k": 0})
        self.assertEqual(capstone._resolve_setting(normalized, 0.88, "distiller.top_p"), 0)

    def test_planner_owns_max_caption_chars_and_canonical_wins(self) -> None:
        plan = {
            "captionforge_config_type": "captionforge_pipeline_plan",
            "distiller": {"max_caption_chars_for_llm": 0},
            "pass_b": {"max_caption_chars_for_llm": 111},
            "pass_b_distiller": {"max_caption_chars_for_llm": 222},
        }
        self.assertEqual(capstone._resolve_fat_draft_max_caption_chars(plan, 333), 0)
        self.assertEqual(capstone._resolve_fat_draft_max_caption_chars({}, 333), 333)

        legacy_plan = {
            "captionforge_config_type": "captionforge_pipeline_plan",
            "pass_b_distiller": {"max_caption_chars_for_llm": 222},
        }
        self.assertEqual(capstone._resolve_fat_draft_max_caption_chars(legacy_plan, 333), 222)

    def test_planner_audit_controls_are_independent_and_override_widgets(self) -> None:
        plan = {
            "captionforge_config_type": "captionforge_pipeline_plan",
            "distiller": {"write_prompt_jsonl": True, "preserve_raw_response": False},
            "validator": {"write_prompt_jsonl": False, "preserve_raw_vlm_response": True},
        }
        resolved = capstone._resolve_stage_audit_settings(plan, False, True)
        self.assertEqual(
            resolved,
            {
                "fat_write_prompts": True,
                "fat_preserve_raw": False,
                "val_write_prompts": False,
                "val_preserve_raw": True,
                "fmt_write_prompts": False,
                "fmt_preserve_raw": True,
            },
        )

        standalone = capstone._resolve_stage_audit_settings({}, True, False)
        self.assertTrue(standalone["fat_write_prompts"])
        self.assertTrue(standalone["val_write_prompts"])
        self.assertTrue(standalone["fmt_write_prompts"])
        self.assertFalse(standalone["fat_preserve_raw"])
        self.assertFalse(standalone["val_preserve_raw"])
        self.assertFalse(standalone["fmt_preserve_raw"])

    def test_canonical_audit_namespace_wins_over_legacy(self) -> None:
        plan = {
            "captionforge_config_type": "captionforge_pipeline_plan",
            "distiller": {"write_prompt_jsonl": False},
            "pass_b": {"write_prompt_jsonl": True},
            "pass_b_distiller": {"write_prompt_jsonl": True},
            "validator": {"preserve_raw_vlm_response": False},
            "pass_c": {"preserve_raw_vlm_response": True},
        }
        resolved = capstone._resolve_stage_audit_settings(plan, True, True)
        self.assertFalse(resolved["fat_write_prompts"])
        self.assertFalse(resolved["val_preserve_raw"])

    def test_stale_style_and_strategy_keys_do_not_control_active_behavior(self) -> None:
        stale_plan = {
            "captionforge_config_type": "captionforge_pipeline_plan",
            "distiller": {"strategy": "by_model_then_global"},
            "final": {"caption_style": "comma"},
        }
        export_format = capstone._resolve_setting(
            stale_plan,
            "natural",
            "final.txt_export_format",
            default="natural",
        )
        self.assertEqual(export_format, "natural")
        self.assertEqual(capstone._resolve_fat_draft_max_caption_chars(stale_plan, 1536), 1536)


class OverwriteContractTests(unittest.TestCase):
    def test_raw_response_preserves_or_replaces_deterministic_file(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            raw_dir = Path(temp_dir)
            expected_path = raw_dir / "image__01_fat_draft_raw.json"
            first_path = capstone._save_raw_response(
                raw_dir,
                "image",
                "01_fat_draft_raw",
                {"response": "first"},
            )
            self.assertEqual(first_path, str(expected_path))
            first = expected_path.read_text(encoding="utf-8")

            skipped_path = capstone._save_raw_response(
                raw_dir,
                "image",
                "01_fat_draft_raw",
                {"response": "second"},
                overwrite=False,
            )
            self.assertEqual(skipped_path, "")
            self.assertEqual(expected_path.read_text(encoding="utf-8"), first)

            replaced_path = capstone._save_raw_response(
                raw_dir,
                "image",
                "01_fat_draft_raw",
                {"response": "second"},
                overwrite=True,
            )
            self.assertEqual(replaced_path, str(expected_path))
            self.assertIn("second", expected_path.read_text(encoding="utf-8"))

    def test_jsonl_append_and_reset_semantics(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            keys = (
                "fat_draft_jsonl",
                "fat_draft_prompt_jsonl",
                "validator_jsonl",
                "validator_prompt_jsonl",
                "taggy_jsonl",
                "taggy_prompt_jsonl",
                "final_jsonl",
            )
            paths = {key: str(root / f"{key}.jsonl") for key in keys}
            final_path = Path(paths["final_jsonl"])
            capstone._write_jsonl(final_path, [{"run": 1}], append=True)
            capstone._write_jsonl(final_path, [{"run": 2}], append=True)
            self.assertEqual(len(final_path.read_text(encoding="utf-8").splitlines()), 2)

            capstone._reset_outputs(paths, overwrite=False)
            self.assertTrue(final_path.exists())
            capstone._reset_outputs(paths, overwrite=True)
            self.assertFalse(final_path.exists())

    def test_final_sidecars_are_preserved_when_overwrite_is_false(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            image_path = Path(temp_dir) / "image.png"
            capstone._write_final_txt_sidecars(
                image_path,
                "original long",
                "original short",
                "original taggy",
                overwrite=True,
            )
            capstone._write_final_txt_sidecars(
                image_path,
                "replacement long",
                "replacement short",
                "replacement taggy",
                overwrite=False,
            )
            self.assertEqual((Path(temp_dir) / "image_long.txt").read_text(encoding="utf-8"), "original long\n")
            self.assertEqual((Path(temp_dir) / "image_short.txt").read_text(encoding="utf-8"), "original short\n")
            self.assertEqual((Path(temp_dir) / "image_taggy.txt").read_text(encoding="utf-8"), "original taggy\n")


if __name__ == "__main__":
    unittest.main()
