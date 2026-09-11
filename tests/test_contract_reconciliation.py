from __future__ import annotations

import importlib
import json
import struct
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest import mock

from PIL import Image

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


def _embedded_png_workflow(path: Path) -> dict:
    data = path.read_bytes()
    offset = 8
    while offset < len(data):
        length = struct.unpack(">I", data[offset : offset + 4])[0]
        chunk_type = data[offset + 4 : offset + 8]
        chunk_data = data[offset + 8 : offset + 8 + length]
        if chunk_type == b"tEXt":
            keyword, separator, text = chunk_data.partition(b"\0")
            if separator and keyword == b"workflow":
                return json.loads(text.decode("utf-8"))
        offset += 12 + length
    raise AssertionError(f"No workflow tEXt chunk in {path}")


class PlannerContractTests(unittest.TestCase):
    def test_fresh_defaults_freeze_pass_a_and_downstream_parity(self) -> None:
        planner_types = planner_node.JLC_CaptionForge_Pipeline_Planner.INPUT_TYPES()
        capstone_types = capstone.JLC_CaptionForge.INPUT_TYPES()
        planner_required = planner_types["required"]
        capstone_required = capstone_types["required"]
        capstone_optional = capstone_types["optional"]

        self.assertEqual(
            {
                name: planner_required[name][1]["default"]
                for name in (
                    "Caption - Joy runs/image",
                    "Caption - Qwen runs/image",
                    "Caption - Ollama runs/image",
                )
            },
            {
                "Caption - Joy runs/image": "2",
                "Caption - Qwen runs/image": "1",
                "Caption - Ollama runs/image": "1",
            },
        )

        shared_controls = (
            ("Ollama - URL", "Ollama - URL"),
            ("Ollama - keep loaded", "Ollama - keep loaded"),
            ("Ollama - request timeout seconds", "Ollama - request timeout seconds"),
            ("Distiller - model", "Fat Draft - model"),
            ("Distiller - custom Ollama model", "Fat Draft - custom Ollama model"),
            ("Distiller - prompt", "Fat Draft - prompt"),
            ("Distiller - max caption chars for LLM", "Fat Draft - max caption chars"),
            ("Distiller - num predict", "Fat Draft - max new tokens"),
            ("Distiller - temperature", "Fat Draft - temperature"),
            ("Distiller - top p", "Fat Draft - top p"),
            ("Distiller - top k", "Fat Draft - top k"),
            ("Validator - model", "Validator - model"),
            ("Validator - custom Ollama model", "Validator - custom Ollama model"),
            ("Validator - system prompt", "Validator - system prompt"),
            ("Validator - prompt", "Validator - prompt"),
            ("Validator - num predict", "Validator - max new tokens"),
            ("Validator - temperature", "Validator - temperature"),
            ("Validator - top p", "Validator - top p"),
            ("Validator - top k", "Validator - top k"),
            ("Formatter - model", "Formatter - model"),
            ("Formatter - custom Ollama model", "Formatter - custom Ollama model"),
            ("Formatter - prompt", "Formatter - prompt"),
            ("Formatter - num predict", "Formatter - max new tokens"),
            ("Formatter - temperature", "Formatter - temperature"),
            ("Formatter - top p", "Formatter - top p"),
            ("Formatter - top k", "Formatter - top k"),
        )
        for planner_name, capstone_name in shared_controls:
            with self.subTest(control=planner_name):
                self.assertEqual(
                    planner_required[planner_name][1]["default"],
                    capstone_required[capstone_name][1]["default"],
                )

        for planner_name, capstone_name in (
            ("Distiller - seed", "Distiller seed"),
            ("Validator - seed", "Validator seed"),
            ("Formatter - seed", "Formatter seed"),
        ):
            with self.subTest(control=planner_name):
                self.assertEqual(planner_required[planner_name][1]["default"], -1)
                self.assertNotIn("default", capstone_optional[capstone_name][1])
                self.assertIsNone(capstone._normalize_optional_seed(None))
                self.assertIsNone(
                    capstone._normalize_optional_seed(
                        planner_required[planner_name][1]["default"]
                    )
                )

        for prefix, raw_name in (
            ("Distiller", "Distiller - preserve raw response"),
            ("Validator", "Validator - preserve raw VLM response"),
            ("Formatter", "Formatter - preserve raw response"),
        ):
            with self.subTest(control=f"{prefix} audit defaults"):
                self.assertEqual(
                    planner_required[f"{prefix} - write prompt JSONL"][1]["default"],
                    capstone_required["Audit - write prompt JSONL"][1]["default"],
                )
                self.assertEqual(
                    planner_required[raw_name][1]["default"],
                    capstone_required["Audit - preserve raw responses"][1]["default"],
                )

        self.assertNotIn("exactly three concise sentences", capstone.DEFAULT_TAGGY_FORMATTER_INSTRUCTIONS)
        self.assertIn("at most 90 words", capstone.DEFAULT_TAGGY_FORMATTER_INSTRUCTIONS)

        engine_plan = planner_engine.build_captionforge_pipeline_plan(output_dir="unused")
        self.assertEqual(
            tuple(
                len(planner_engine.expand_captionforge_runs(engine_plan, model_key=family))
                for family in ("joy", "qwen", "ollama")
            ),
            (2, 1, 1),
        )
        self.assertEqual(engine_plan["distiller"]["seed"], -1)
        self.assertEqual(engine_plan["validator"]["seed"], -1)
        self.assertEqual(engine_plan["formatter"]["seed"], -1)

    def test_planner_resolves_custom_models_for_all_downstream_passes(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            _, plan, _ = planner_node.JLC_CaptionForge_Pipeline_Planner().plan(
                **{
                    "Output - folder": temp_dir,
                    "Output - overwrite outputs": False,
                    "Caption - Joy runs/image": "1",
                    "Caption - Qwen runs/image": "Disabled",
                    "Distiller - model": "custom",
                    "Distiller - custom Ollama model": "custom-distiller",
                    "Validator - model": "custom",
                    "Validator - custom Ollama model": "custom-validator",
                    "Formatter - model": "custom",
                    "Formatter - custom Ollama model": "custom-formatter",
                }
            )

        self.assertEqual(plan["distiller"]["model"], "custom-distiller")
        self.assertEqual(plan["validator"]["model"], "custom-validator")
        self.assertEqual(plan["formatter"]["model"], "custom-formatter")

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
                    "Ollama - URL": "http://127.0.0.1:11435",
                    "Ollama - keep loaded": False,
                    "Ollama - request timeout seconds": 321,
                    "Distiller - model": "mistral-small:24b",
                    "Distiller - prompt": "planner distiller instructions",
                    "Distiller - seed": 3,
                    "Distiller - max caption chars for LLM": 1536,
                    "Distiller - num predict": 3096,
                    "Distiller - temperature": 0.24,
                    "Distiller - top p": 0.90,
                    "Distiller - top k": 60,
                    "Distiller - write prompt JSONL": False,
                    "Distiller - preserve raw response": False,
                    "Validator - model": "gemma4:26b",
                    "Validator - system prompt": "planner validator system",
                    "Validator - prompt": "planner validator instructions",
                    "Validator - seed": 3,
                    "Validator - num predict": 2200,
                    "Validator - temperature": 0.0,
                    "Validator - top p": 0.92,
                    "Validator - top k": 80,
                    "Validator - write prompt JSONL": False,
                    "Validator - preserve raw VLM response": False,
                    "Formatter - model": "custom",
                    "Formatter - custom Ollama model": "planner-formatter",
                    "Formatter - prompt": "planner formatter instructions",
                    "Formatter - seed": 3,
                    "Formatter - num predict": 1234,
                    "Formatter - temperature": 0.34,
                    "Formatter - top p": 0.76,
                    "Formatter - top k": 43,
                    "Formatter - write prompt JSONL": False,
                    "Formatter - preserve raw response": False,
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
        self.assertEqual(plan["distiller"]["prompt"], "planner distiller instructions")
        self.assertEqual(plan["distiller"]["num_predict"], 3096)
        self.assertEqual(plan["distiller"]["temperature"], 0.24)
        self.assertEqual(plan["distiller"]["top_p"], 0.90)
        self.assertEqual(plan["distiller"]["top_k"], 60)
        self.assertEqual(plan["distiller"]["seed"], 3)
        self.assertEqual(plan["validator"]["model"], "gemma4:26b")
        self.assertEqual(plan["validator"]["system_prompt"], "planner validator system")
        self.assertEqual(plan["validator"]["prompt"], "planner validator instructions")
        self.assertEqual(plan["pass_c"]["system_prompt"], "planner validator system")
        self.assertEqual(plan["pass_c"]["prompt"], "planner validator instructions")
        self.assertEqual(plan["pass_c_vlm_validator"]["system_prompt"], "planner validator system")
        self.assertEqual(plan["pass_c_vlm_validator"]["prompt"], "planner validator instructions")
        self.assertEqual(plan["validator"]["num_predict"], 2200)
        self.assertEqual(plan["validator"]["temperature"], 0.0)
        self.assertEqual(plan["validator"]["top_p"], 0.92)
        self.assertEqual(plan["validator"]["top_k"], 80)
        self.assertEqual(plan["validator"]["seed"], 3)
        self.assertEqual(plan["ollama"]["url"], "http://127.0.0.1:11435")
        self.assertFalse(plan["ollama"]["keep_loaded"])
        self.assertEqual(plan["ollama"]["request_timeout_seconds"], 321)
        self.assertEqual(plan["formatter"]["model"], "planner-formatter")
        self.assertEqual(plan["formatter"]["prompt"], "planner formatter instructions")
        self.assertEqual(plan["formatter"]["seed"], 3)
        self.assertEqual(plan["formatter"]["num_predict"], 1234)
        self.assertEqual(plan["formatter"]["temperature"], 0.34)
        self.assertEqual(plan["formatter"]["top_p"], 0.76)
        self.assertEqual(plan["formatter"]["top_k"], 43)
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
                    "Distiller - seed": 0,
                    "Distiller - max caption chars for LLM": 0,
                    "Distiller - top p": 0,
                    "Distiller - top k": 0,
                    "Validator - seed": 0,
                    "Validator - top p": 0,
                    "Validator - top k": 0,
                    "Formatter - seed": 0,
                    # Old serialized inputs are accepted as harmless extras.
                    "Distiller - strategy": "by_model_then_global",
                    "Final - caption style": "comma",
                }
            )

        self.assertEqual(plan["shared"]["base_seed"], 0)
        self.assertEqual(plan["shared"]["max_size"], 0)
        self.assertEqual(plan["shared"]["max_new_tokens"], 4096)
        self.assertEqual(plan["distiller"]["seed"], 0)
        self.assertEqual(plan["distiller"]["max_caption_chars_for_llm"], 0)
        self.assertEqual(plan["distiller"]["top_p"], 0.0)
        self.assertEqual(plan["distiller"]["top_k"], 0)
        self.assertEqual(plan["validator"]["seed"], 0)
        self.assertEqual(plan["validator"]["top_p"], 0.0)
        self.assertEqual(plan["validator"]["top_k"], 0)
        self.assertEqual(plan["formatter"]["seed"], 0)
        self.assertNotIn("strategy", plan["distiller"])
        self.assertNotIn("caption_style", plan["final"])

    def test_removed_widgets_and_pass_a_token_contract(self) -> None:
        required = planner_node.JLC_CaptionForge_Pipeline_Planner.INPUT_TYPES()["required"]
        self.assertNotIn("Distiller - strategy", required)
        self.assertNotIn("Final - caption style", required)
        self.assertNotIn("Distiller - seed mode", required)
        self.assertNotIn("Validator - seed mode", required)
        self.assertIn("Formatter - seed", required)
        self.assertEqual(
            required["Validator - system prompt"][1]["default"],
            capstone.DEFAULT_VALIDATOR_SYSTEM_PROMPT,
        )
        self.assertEqual(
            required["Validator - prompt"][1]["default"],
            capstone.DEFAULT_VALIDATOR_INSTRUCTIONS,
        )
        self.assertEqual(
            required["Distiller - prompt"][1]["default"],
            capstone.DEFAULT_FAT_DRAFT_INSTRUCTIONS,
        )
        self.assertEqual(
            required["Formatter - prompt"][1]["default"],
            capstone.DEFAULT_TAGGY_FORMATTER_INSTRUCTIONS,
        )
        self.assertIn("Ollama num_predict", required["Distiller - num predict"][1]["tooltip"])
        self.assertIn("Ollama num_predict", required["Validator - num predict"][1]["tooltip"])
        self.assertIn("Ollama num_predict", required["Formatter - num predict"][1]["tooltip"])
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
        self.assertEqual(
            default_plan["validator"]["system_prompt"],
            capstone.DEFAULT_VALIDATOR_SYSTEM_PROMPT,
        )
        self.assertEqual(
            default_plan["validator"]["prompt"],
            capstone.DEFAULT_VALIDATOR_INSTRUCTIONS,
        )
        self.assertEqual(default_plan["distiller"]["prompt"], capstone.DEFAULT_FAT_DRAFT_INSTRUCTIONS)
        self.assertEqual(default_plan["formatter"]["prompt"], capstone.DEFAULT_TAGGY_FORMATTER_INSTRUCTIONS)
        self.assertEqual(default_plan["ollama"]["url"], capstone.DEFAULT_OLLAMA_URL)


class WorkflowAssetContractTests(unittest.TestCase):
    def test_planner_downstream_controls_match_ui_api_and_png_workflows(self) -> None:
        ui = json.loads((ROOT / "assets" / "workflows" / "CaptionForge_FullWorkflow.json").read_text(encoding="utf-8"))
        api = json.loads((ROOT / "assets" / "workflows" / "CaptionForge_FullWorkflow_API.json").read_text(encoding="utf-8"))
        png = _embedded_png_workflow(ROOT / "assets" / "workflows" / "CaptionForge_FullWorkflow.png")

        prompt_defaults = {
            "Distiller - prompt": capstone.DEFAULT_FAT_DRAFT_INSTRUCTIONS,
            "Validator - system prompt": capstone.DEFAULT_VALIDATOR_SYSTEM_PROMPT,
            "Validator - prompt": capstone.DEFAULT_VALIDATOR_INSTRUCTIONS,
            "Formatter - prompt": capstone.DEFAULT_TAGGY_FORMATTER_INSTRUCTIONS,
        }
        required_names = list(
            planner_node.JLC_CaptionForge_Pipeline_Planner.INPUT_TYPES()["required"]
        )
        ui_planner = next(
            node for node in ui["nodes"]
            if node["type"] == "JLC_CaptionForge_Pipeline_Planner"
        )
        api_planner = next(
            node for node in api.values()
            if node["class_type"] == "JLC_CaptionForge_Pipeline_Planner"
        )
        for name in required_names:
            self.assertEqual(api_planner["inputs"][name], ui_planner["widgets_values_named"][name])

        for workflow in (ui, png):
            node = next(
                node for node in workflow["nodes"]
                if node["type"] == "JLC_CaptionForge_Pipeline_Planner"
            )
            self.assertEqual(len(node["widgets_values"]), len(required_names))
            for index, name in enumerate(required_names):
                self.assertEqual(node["widgets_values"][index], node["widgets_values_named"][name])
            for name, value in prompt_defaults.items():
                self.assertEqual(node["widgets_values_named"][name], value)
            formatter_prompt = node["widgets_values_named"]["Formatter - prompt"]
            self.assertIn("SHORT:", formatter_prompt)
            self.assertIn("TAGGY:", formatter_prompt)

    def test_formatter_prompt_matches_ui_api_and_png_workflows(self) -> None:
        ui = json.loads((ROOT / "assets" / "workflows" / "CaptionForge_FullWorkflow.json").read_text(encoding="utf-8"))
        api = json.loads((ROOT / "assets" / "workflows" / "CaptionForge_FullWorkflow_API.json").read_text(encoding="utf-8"))
        png = _embedded_png_workflow(ROOT / "assets" / "workflows" / "CaptionForge_FullWorkflow.png")

        expected = capstone.DEFAULT_TAGGY_FORMATTER_INSTRUCTIONS
        api_capstone = next(
            node for node in api.values() if node["class_type"] == "JLC_CaptionForge"
        )
        api_prompt = api_capstone["inputs"]["Formatter - prompt"]
        self.assertEqual(api_prompt, expected)

        for workflow in (ui, png):
            node = next(node for node in workflow["nodes"] if node["type"] == "JLC_CaptionForge")
            self.assertEqual(node["widgets_values_named"]["Formatter - prompt"], expected)

    def test_canonical_nodes_match_source_defaults_and_current_widget_order(self) -> None:
        ui = json.loads((ROOT / "assets" / "workflows" / "CaptionForge_FullWorkflow.json").read_text(encoding="utf-8"))
        api = json.loads((ROOT / "assets" / "workflows" / "CaptionForge_FullWorkflow_API.json").read_text(encoding="utf-8"))
        png = _embedded_png_workflow(ROOT / "assets" / "workflows" / "CaptionForge_FullWorkflow.png")
        self.assertEqual(png, ui)

        contracts = (
            (
                "JLC_CaptionForge_Pipeline_Planner",
                planner_node.JLC_CaptionForge_Pipeline_Planner,
                (
                    "Caption - Joy runs/image",
                    "Caption - Qwen runs/image",
                    "Caption - Ollama runs/image",
                    "Ollama - URL",
                    "Ollama - keep loaded",
                    "Ollama - request timeout seconds",
                    "Distiller - model",
                    "Distiller - custom Ollama model",
                    "Distiller - prompt",
                    "Distiller - seed",
                    "Distiller - max caption chars for LLM",
                    "Distiller - num predict",
                    "Distiller - temperature",
                    "Distiller - top p",
                    "Distiller - top k",
                    "Distiller - write prompt JSONL",
                    "Distiller - preserve raw response",
                    "Validator - model",
                    "Validator - custom Ollama model",
                    "Validator - system prompt",
                    "Validator - prompt",
                    "Validator - seed",
                    "Validator - num predict",
                    "Validator - temperature",
                    "Validator - top p",
                    "Validator - top k",
                    "Validator - write prompt JSONL",
                    "Validator - preserve raw VLM response",
                    "Formatter - model",
                    "Formatter - custom Ollama model",
                    "Formatter - prompt",
                    "Formatter - seed",
                    "Formatter - num predict",
                    "Formatter - temperature",
                    "Formatter - top p",
                    "Formatter - top k",
                    "Formatter - write prompt JSONL",
                    "Formatter - preserve raw response",
                ),
            ),
            (
                "JLC_CaptionForge",
                capstone.JLC_CaptionForge,
                (
                    "Ollama - URL",
                    "Ollama - keep loaded",
                    "Ollama - request timeout seconds",
                    "Fat Draft - model",
                    "Fat Draft - custom Ollama model",
                    "Fat Draft - prompt",
                    "Fat Draft - max caption chars",
                    "Fat Draft - max new tokens",
                    "Fat Draft - temperature",
                    "Fat Draft - top p",
                    "Fat Draft - top k",
                    "Validator - model",
                    "Validator - custom Ollama model",
                    "Validator - system prompt",
                    "Validator - prompt",
                    "Validator - max new tokens",
                    "Validator - temperature",
                    "Validator - top p",
                    "Validator - top k",
                    "Formatter - model",
                    "Formatter - custom Ollama model",
                    "Formatter - prompt",
                    "Formatter - max new tokens",
                    "Formatter - temperature",
                    "Formatter - top p",
                    "Formatter - top k",
                    "Audit - write prompt JSONL",
                    "Audit - preserve raw responses",
                ),
            ),
        )
        for class_type, node_class, default_names in contracts:
            required = node_class.INPUT_TYPES()["required"]
            names = list(required)
            ui_node = next(node for node in ui["nodes"] if node["type"] == class_type)
            api_node = next(node for node in api.values() if node["class_type"] == class_type)
            self.assertEqual(len(ui_node["widgets_values"]), len(names))
            for index, name in enumerate(names):
                with self.subTest(class_type=class_type, index=index, control=name):
                    self.assertEqual(
                        ui_node["widgets_values"][index],
                        ui_node["widgets_values_named"][name],
                    )
                    self.assertEqual(
                        api_node["inputs"][name],
                        ui_node["widgets_values_named"][name],
                    )
            for name in default_names:
                with self.subTest(class_type=class_type, default=name):
                    self.assertEqual(
                        ui_node["widgets_values_named"][name],
                        required[name][1]["default"],
                    )

    def test_canonical_workflows_have_only_public_safe_paths(self) -> None:
        ui = json.loads((ROOT / "assets" / "workflows" / "CaptionForge_FullWorkflow.json").read_text(encoding="utf-8"))
        api = json.loads((ROOT / "assets" / "workflows" / "CaptionForge_FullWorkflow_API.json").read_text(encoding="utf-8"))
        png = _embedded_png_workflow(ROOT / "assets" / "workflows" / "CaptionForge_FullWorkflow.png")
        for name, artifact in (("ui", ui), ("api", api), ("png", png)):
            serialized = json.dumps(artifact).lower()
            with self.subTest(artifact=name):
                for forbidden in ("c:\\\\users\\\\josel", ".tests\\\\", "release_1.0.1_test01", "smoke_test"):
                    self.assertNotIn(forbidden, serialized)


class CapstoneResolutionTests(unittest.TestCase):
    def _capture_downstream_calls(self, *, plan: dict | None = None) -> tuple[list[dict], dict]:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            Image.new("RGB", (2, 2), "white").save(root / "image.png")
            caption_path = root / "captions.jsonl"
            caption_path.write_text(
                json.dumps(
                    {
                        "image": "image.png",
                        "image_key": "image",
                        "caption": "source caption",
                        "model_family": "joy",
                        "status": "ok",
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            kwargs = {
                "Input - captions JSONL": str(caption_path),
                "Input - image path": temp_dir,
                "Output - folder": str(root / "output"),
                "Output - run name": "validator-contract",
                "Output - overwrite outputs": True,
                "Ollama - URL": "http://127.0.0.1:11431",
                "Ollama - keep loaded": True,
                "Ollama - request timeout seconds": 111,
                "Fat Draft - model": "Custom",
                "Fat Draft - custom Ollama model": "local-distiller",
                "Fat Draft - prompt": "local distiller instructions",
                "Fat Draft - max new tokens": 666,
                "Fat Draft - temperature": 0.6,
                "Fat Draft - top p": 0.61,
                "Fat Draft - top k": 16,
                "Distiller seed": 18,
                "Validator - model": "Custom",
                "Validator - custom Ollama model": "local-vlm",
                "Validator - system prompt": "local validator system",
                "Validator - prompt": "local validator instructions",
                "Validator - max new tokens": 777,
                "Validator - temperature": 0.7,
                "Validator - top p": 0.71,
                "Validator - top k": 17,
                "Validator seed": 19,
                "Formatter - model": "Custom",
                "Formatter - custom Ollama model": "local-formatter",
                "Formatter - prompt": "local formatter instructions",
                "Formatter - max new tokens": 999,
                "Formatter - temperature": 0.9,
                "Formatter - top p": 0.91,
                "Formatter - top k": 19,
                "Formatter seed": 20,
            }
            if plan is not None:
                kwargs["pipeline_plan"] = plan

            with mock.patch.object(
                capstone, "_evict_python_models_before_ollama_if_needed"
            ), mock.patch.object(
                capstone,
                "_ollama_generate_text",
                return_value=("generated text", {"ok": True}),
            ) as text_calls, mock.patch.object(
                capstone,
                "_ollama_chat_image",
                return_value=("validated caption", {"ok": True}),
            ) as validator_call:
                capstone.JLC_CaptionForge().forge(**kwargs)

            return (
                [dict(call.kwargs) for call in text_calls.call_args_list],
                dict(validator_call.call_args.kwargs),
            )

    def test_standalone_distiller_and_formatter_widgets_reach_production_calls(self) -> None:
        text_calls, _ = self._capture_downstream_calls()
        distiller_call, formatter_call = text_calls

        self.assertEqual(distiller_call["model"], "local-distiller")
        self.assertTrue(distiller_call["prompt"].startswith("local distiller instructions"))
        self.assertEqual(distiller_call["num_predict"], 666)
        self.assertEqual(distiller_call["temperature"], 0.6)
        self.assertEqual(distiller_call["top_p"], 0.61)
        self.assertEqual(distiller_call["top_k"], 16)
        self.assertEqual(distiller_call["seed"], 18)

        self.assertEqual(formatter_call["model"], "local-formatter")
        self.assertTrue(formatter_call["prompt"].startswith("local formatter instructions"))
        self.assertEqual(formatter_call["num_predict"], 999)
        self.assertEqual(formatter_call["temperature"], 0.9)
        self.assertEqual(formatter_call["top_p"], 0.91)
        self.assertEqual(formatter_call["top_k"], 19)
        self.assertEqual(formatter_call["seed"], 20)

        for call in text_calls:
            self.assertEqual(call["ollama_url"], "http://127.0.0.1:11431")
            self.assertTrue(call["keep_loaded"])
            self.assertEqual(call["timeout"], 111.0)

    def test_standalone_validator_widgets_reach_production_call(self) -> None:
        _, call = self._capture_downstream_calls()
        self.assertEqual(call["model"], "local-vlm")
        self.assertEqual(call["system_prompt"], "local validator system")
        self.assertTrue(call["user_prompt"].startswith("local validator instructions"))
        self.assertEqual(call["num_predict"], 777)
        self.assertEqual(call["temperature"], 0.7)
        self.assertEqual(call["top_p"], 0.71)
        self.assertEqual(call["top_k"], 17)
        self.assertEqual(call["seed"], 19)
        self.assertEqual(call["ollama_url"], "http://127.0.0.1:11431")
        self.assertTrue(call["keep_loaded"])
        self.assertEqual(call["timeout"], 111.0)

    def test_planner_downstream_settings_override_widgets_in_production_calls(self) -> None:
        text_calls, call = self._capture_downstream_calls(
            plan={
                "captionforge_config_type": "captionforge_pipeline_plan",
                "ollama": {
                    "url": "http://127.0.0.1:11432",
                    "keep_loaded": False,
                    "request_timeout_seconds": 222,
                },
                "distiller": {
                    "model": "planner-distiller",
                    "prompt": "planner distiller instructions",
                    "seed": 21,
                    "num_predict": 667,
                    "temperature": 0.16,
                    "top_p": 0.62,
                    "top_k": 26,
                },
                "validator": {
                    "model": "planner-vlm",
                    "system_prompt": "planner validator system",
                    "prompt": "planner validator instructions",
                    "seed": 23,
                    "num_predict": 888,
                    "temperature": 0.2,
                    "top_p": 0.82,
                    "top_k": 28,
                },
                "formatter": {
                    "model": "planner-formatter",
                    "prompt": "planner formatter instructions",
                    "seed": 25,
                    "num_predict": 889,
                    "temperature": 0.4,
                    "top_p": 0.84,
                    "top_k": 48,
                },
            }
        )
        distiller_call, formatter_call = text_calls

        self.assertEqual(distiller_call["model"], "planner-distiller")
        self.assertTrue(distiller_call["prompt"].startswith("planner distiller instructions"))
        self.assertEqual(distiller_call["num_predict"], 667)
        self.assertEqual(distiller_call["temperature"], 0.16)
        self.assertEqual(distiller_call["top_p"], 0.62)
        self.assertEqual(distiller_call["top_k"], 26)
        self.assertEqual(distiller_call["seed"], 21)

        self.assertEqual(call["model"], "planner-vlm")
        self.assertEqual(call["system_prompt"], "planner validator system")
        self.assertTrue(call["user_prompt"].startswith("planner validator instructions"))
        self.assertEqual(call["num_predict"], 888)
        self.assertEqual(call["temperature"], 0.2)
        self.assertEqual(call["top_p"], 0.82)
        self.assertEqual(call["top_k"], 28)
        self.assertEqual(call["seed"], 23)

        self.assertEqual(formatter_call["model"], "planner-formatter")
        self.assertTrue(formatter_call["prompt"].startswith("planner formatter instructions"))
        self.assertEqual(formatter_call["num_predict"], 889)
        self.assertEqual(formatter_call["temperature"], 0.4)
        self.assertEqual(formatter_call["top_p"], 0.84)
        self.assertEqual(formatter_call["top_k"], 48)
        self.assertEqual(formatter_call["seed"], 25)

        for downstream_call in [*text_calls, call]:
            self.assertEqual(downstream_call["ollama_url"], "http://127.0.0.1:11432")
            self.assertFalse(downstream_call["keep_loaded"])
            self.assertEqual(downstream_call["timeout"], 222.0)

    def test_formatter_derivatives_parse_dual_and_legacy_responses(self) -> None:
        short, taggy = capstone._parse_formatter_derivatives(
            "SHORT: A concise natural caption.\n"
            "TAGGY: subject, blue dress, soft lighting."
        )
        self.assertEqual(short, "A concise natural caption.")
        self.assertEqual(taggy, "subject, blue dress, soft lighting.")

        legacy_short, legacy_taggy = capstone._parse_formatter_derivatives(
            "subject, blue dress, soft lighting"
        )
        self.assertEqual(legacy_short, "")
        self.assertEqual(legacy_taggy, "subject, blue dress, soft lighting")

    def test_short_limit_and_deterministic_fallback_obey_word_cap(self) -> None:
        overlong_sentence = " ".join(f"word{i}" for i in range(120)) + "."
        ai_short = capstone._limit_ai_short_caption(overlong_sentence, max_words=90)
        fallback = capstone._compact_lora_short_caption(overlong_sentence, "")
        self.assertEqual(len(ai_short.removesuffix("…").split()), 90)
        self.assertEqual(len(fallback.split()), 90)

    def test_taggy_compaction_removes_terminal_sentence_punctuation(self) -> None:
        self.assertEqual(
            capstone._compact_taggy_caption("subject, blue dress, soft lighting."),
            "subject, blue dress, soft lighting",
        )

    def test_normalization_preserves_nested_zero_values(self) -> None:
        normalized = planner_engine.normalize_captionforge_pipeline_plan(
            json.dumps(
                {
                    "captionforge_config_type": "captionforge_pipeline_plan",
                    "distiller": {"seed": 0, "top_p": 0, "top_k": 0},
                }
            )
        )
        self.assertEqual(normalized["distiller"], {"seed": 0, "top_p": 0, "top_k": 0})
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
            "formatter": {"write_prompt_jsonl": True, "preserve_raw_response": False},
        }
        resolved = capstone._resolve_stage_audit_settings(plan, False, True)
        self.assertEqual(
            resolved,
            {
                "fat_write_prompts": True,
                "fat_preserve_raw": False,
                "val_write_prompts": False,
                "val_preserve_raw": True,
                "fmt_write_prompts": True,
                "fmt_preserve_raw": False,
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
            "formatter": {"write_prompt_jsonl": False},
            "pass_d": {"write_prompt_jsonl": True},
        }
        resolved = capstone._resolve_stage_audit_settings(plan, True, True)
        self.assertFalse(resolved["fat_write_prompts"])
        self.assertFalse(resolved["val_preserve_raw"])
        self.assertFalse(resolved["fmt_write_prompts"])

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
