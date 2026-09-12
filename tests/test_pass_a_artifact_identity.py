"""Focused tests for Pass-A source identity and artifact persistence."""

from __future__ import annotations

import importlib
import json
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


_install_namespace("CaptionForge", ROOT)
_install_namespace("CaptionForge.engines", ROOT / "engines")
_install_namespace("CaptionForge.nodes", ROOT / "nodes")
_install_namespace("CaptionForge.nodes.caption_nodes", ROOT / "nodes" / "caption_nodes")

if "folder_paths" not in sys.modules:
    folder_paths = types.ModuleType("folder_paths")
    folder_paths.models_dir = str(ROOT / "models")
    folder_paths.get_output_directory = lambda: str(ROOT / "output")
    sys.modules["folder_paths"] = folder_paths
elif not hasattr(sys.modules["folder_paths"], "models_dir"):
    sys.modules["folder_paths"].models_dir = str(ROOT / "models")

planner_engine = importlib.import_module("CaptionForge.engines.captionforge_pipeline_planner_engine")
source_identity = importlib.import_module("CaptionForge.engines.captionforge_source_identity")
orchestrator = importlib.import_module("CaptionForge.nodes.jlc_captionforge_node")
joy = importlib.import_module("CaptionForge.nodes.caption_nodes.jlc_captionforge_joy_caption_node")
qwen = importlib.import_module("CaptionForge.nodes.caption_nodes.jlc_captionforge_qwen_caption_node")
ollama = importlib.import_module("CaptionForge.nodes.caption_nodes.jlc_captionforge_ollama_caption_node")


FILE_SOURCE_IDENTITIES = {
    "_DSC2094-Edit-2.jpg": "_DSC2094-Edit-2",
    "ChatGPT Image May 26, 2026, 08_21_31 AM.png": "ChatGPT Image May 26, 2026, 08_21_31 AM",
    "Set A/Sub Folder/_DSC2094-Edit-2.jpg": "_DSC2094-Edit-2",
    "Set A/photo.jpg": "photo",
    "Set A/photo.png": "photo",
    "Set B/photo.jpg": "photo",
    "comfy_image_0000.png": "comfy_image_0000",
}
SOURCE_IDENTITIES = {
    **FILE_SOURCE_IDENTITIES,
    "captionforge-optional-image://comfy_image_0000.png": "comfy_image_0000",
}


def _required_defaults(node_class) -> dict[str, object]:
    return {
        name: spec[1]["default"]
        for name, spec in node_class.INPUT_TYPES()["required"].items()
    }


class PassAArtifactIdentityTests(unittest.TestCase):
    def test_each_witness_writes_collision_safe_raw_identity_without_txt_sidecars(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            input_dir = root / "inputs"
            output_dir = root / "outputs"
            input_dir.mkdir()
            for relative_name in FILE_SOURCE_IDENTITIES:
                image_path = input_dir / Path(*relative_name.split("/"))
                image_path.parent.mkdir(parents=True, exist_ok=True)
                Image.new("RGB", (2, 2), "white").save(image_path)
            optional_pil = Image.new("RGB", (2, 2), "black")

            plan = planner_engine.build_captionforge_pipeline_plan(
                output_dir=str(output_dir),
                input_path=str(input_dir),
                run_name="captionforge_run",
                joy_runs_per_image=1,
                qwen_runs_per_image=1,
                ollama_runs_per_image=1,
            )

            joy_engine = mock.Mock()
            joy_engine.config = types.SimpleNamespace(max_size=0)
            joy_engine.local_model_path = ""
            joy_engine.caption_pil.return_value = ("joy caption", "joy raw")
            joy_engine.build_run_config.return_value = {}
            joy_kwargs = _required_defaults(joy.JLC_CaptionForgeJoy)
            joy_kwargs["pipeline_plan"] = plan
            with mock.patch.object(joy, "JoyCaptionEngine", return_value=joy_engine), mock.patch.object(
                joy, "_tensor_to_pil", return_value=[optional_pil]
            ):
                joy.JLC_CaptionForgeJoy().caption(**joy_kwargs)

            qwen_engine = mock.Mock()
            qwen_engine.config = types.SimpleNamespace(max_size=0)
            qwen_engine.local_model_path = ""
            qwen_engine.caption_pil.return_value = ("qwen caption", "qwen raw")
            qwen_engine.build_run_config.return_value = {}
            qwen_kwargs = _required_defaults(qwen.JLC_CaptionForgeQwen)
            qwen_kwargs["pipeline_plan"] = plan
            with mock.patch.object(qwen, "QwenCaptionEngine", return_value=qwen_engine), mock.patch.object(
                qwen, "_tensor_to_pil", return_value=[optional_pil]
            ):
                qwen.JLC_CaptionForgeQwen().caption(**qwen_kwargs)

            ollama_kwargs = _required_defaults(ollama.JLC_CaptionForgeOllamaCaption)
            ollama_kwargs["pipeline_plan"] = plan
            with mock.patch.object(ollama, "_ensure_ollama_model"), mock.patch.object(
                ollama, "_persist_caption_model_if_possible"
            ), mock.patch.object(
                ollama, "_evict_python_models_before_ollama_if_needed"
            ), mock.patch.object(
                ollama, "_ollama_generate_caption", return_value="ollama caption"
            ), mock.patch.object(
                ollama, "_tensor_to_pil", return_value=[optional_pil]
            ):
                ollama.JLC_CaptionForgeOllamaCaption().caption(**ollama_kwargs)

            raw_path = Path(plan["paths"]["caption_jsonl"])
            records = [json.loads(line) for line in raw_path.read_text(encoding="utf-8").splitlines()]
            self.assertEqual(len(records), len(SOURCE_IDENTITIES) * 3)
            for family in ("joy", "qwen", "ollama"):
                family_records = [record for record in records if record["model_family"] == family]
                actual = {record["image_key"]: record["image"] for record in family_records}
                self.assertEqual(actual, SOURCE_IDENTITIES)

            grouped = orchestrator._group_records_by_image(records)
            self.assertEqual(set(grouped), set(SOURCE_IDENTITIES))
            self.assertTrue(all(len(group) == 3 for group in grouped.values()))
            self.assertEqual(list(root.rglob("*.txt")), [])

            opt_images_dir = root / "opt_images"
            opt_images_dir.mkdir()
            optional_path = opt_images_dir / "comfy_image_0000.png"
            optional_pil.save(optional_path)
            optional_key = source_identity.optional_image_identity(0)[1]
            self.assertEqual(optional_key, "captionforge-optional-image://comfy_image_0000.png")
            self.assertEqual(
                orchestrator._resolve_image_path_for_group(
                    grouped[optional_key],
                    [input_dir, opt_images_dir],
                    optional_image_root=opt_images_dir,
                ),
                optional_path,
            )
            self.assertEqual(
                orchestrator._resolve_image_path_for_group(
                    grouped["Set A/Sub Folder/_DSC2094-Edit-2.jpg"],
                    [input_dir, opt_images_dir],
                    optional_image_root=opt_images_dir,
                ),
                input_dir / "Set A" / "Sub Folder" / "_DSC2094-Edit-2.jpg",
            )
            self.assertEqual(
                orchestrator._resolve_image_path_for_group(
                    grouped["Set A/photo.jpg"],
                    [input_dir, opt_images_dir],
                    optional_image_root=opt_images_dir,
                ),
                input_dir / "Set A" / "photo.jpg",
            )
            empty_dataset = root / "empty-inputs"
            empty_dataset.mkdir()
            self.assertEqual(
                orchestrator._resolve_image_path_for_group(
                    [{"image": "comfy_image_0000", "image_key": "comfy_image_0000"}],
                    [empty_dataset, opt_images_dir],
                    optional_image_root=opt_images_dir,
                ),
                optional_path,
            )
            self.assertEqual(
                orchestrator._resolve_image_path_for_group(
                    [{"image": "photo", "image_key": r"Set A\photo.jpg"}],
                    [input_dir, opt_images_dir],
                    optional_image_root=opt_images_dir,
                ),
                input_dir / "Set A" / "photo.jpg",
            )

    def test_raw_response_filenames_do_not_collapse_relative_keys(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            raw_dir = Path(temp_dir)
            first = orchestrator._save_raw_response(raw_dir, "Set A/photo.jpg", "stage", {"ok": 1})
            second = orchestrator._save_raw_response(raw_dir, "Set B/photo.jpg", "stage", {"ok": 2})
            extension_variant = orchestrator._save_raw_response(raw_dir, "Set A/photo.png", "stage", {"ok": 3})
            self.assertEqual(len({first, second, extension_variant}), 3)
            self.assertTrue(all(Path(path).exists() for path in (first, second, extension_variant)))


if __name__ == "__main__":
    unittest.main()
