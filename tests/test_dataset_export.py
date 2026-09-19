"""CPU checks for paired exports, source protection, and Planner ownership."""

from __future__ import annotations

import importlib
import io
import json
import sys
import tempfile
import types
import unittest
import urllib.error
from pathlib import Path
from unittest import mock

from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
for name, path in (("CaptionForge", ROOT), ("CaptionForge.engines", ROOT / "engines"),
                   ("CaptionForge.nodes", ROOT / "nodes"),
                   ("CaptionForge.nodes.caption_nodes", ROOT / "nodes" / "caption_nodes")):
    if name not in sys.modules:
        module = types.ModuleType(name)
        module.__path__ = [str(path)]
        sys.modules[name] = module
if "folder_paths" not in sys.modules:
    sys.modules["folder_paths"] = types.ModuleType("folder_paths")
sys.modules["folder_paths"].models_dir = str(ROOT / "models")
sys.modules["folder_paths"].get_output_directory = lambda: str(ROOT / "output")

exporter = importlib.import_module("CaptionForge.engines.captionforge_dataset_export")
planner = importlib.import_module("CaptionForge.nodes.jlc_captionforge_pipeline_planner_node")
orchestrator = importlib.import_module("CaptionForge.nodes.jlc_captionforge_node")


class DatasetExportTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.source = self.root / "portraits" / "photo.jpg"
        self.source.parent.mkdir()
        Image.new("RGB", (900, 600), "orange").save(self.source)
        self.original_bytes = self.source.read_bytes()
        self.settings = exporter.normalize_export_settings({"enabled": True, "max_size": 256})

    def pair(self, **overrides):
        arguments = dict(source=self.source, image_key="portraits/photo.jpg", input_root=self.root,
                         root=self.root / "training_dataset", settings=self.settings,
                         validator_max_size=1536, captions={"short": "A portrait.", "long": "Long portrait.", "taggy": "portrait"},
                         overwrite=False)
        arguments.update(overrides)
        return exporter.export_pair(**arguments)

    def test_downscale_and_divisibility_never_enlarge(self):
        result = self.pair()
        with Image.open(result["image"]) as image:
            self.assertEqual(image.size, (256, 160))
        self.assertEqual(Path(result["caption"]).read_text().strip(), "A portrait.")
        self.assertEqual(self.source.read_bytes(), self.original_bytes)
        self.assertFalse(self.source.with_suffix(".txt").exists())

    def test_small_image_and_divisor_one(self):
        image = Image.new("RGB", (99, 65))
        self.assertEqual(exporter.resize_for_export(image, 1536, 1).size, (99, 65))
        self.assertEqual(exporter.resize_for_export(image, 1536, 16).size, (96, 64))
        with self.assertRaises(ValueError):
            exporter.resize_for_export(Image.new("RGB", (10, 9)), 1536, 16)

    def test_fractional_divisor_rejected(self):
        with self.assertRaises(ValueError):
            exporter.normalize_export_settings({"divisor": 16.5})

    def test_original_extensions_and_folders_cannot_collide(self):
        first = self.pair()
        second_source = self.source.with_suffix(".png")
        Image.new("RGB", (900, 600)).save(second_source)
        second = self.pair(source=second_source, image_key="portraits/photo.png")
        self.assertNotEqual(first["image"], second["image"])
        self.assertNotEqual(first["caption"], second["caption"])
        self.assertEqual(Path(first["image"]).relative_to(self.root).as_posix(), "training_dataset/files/portraits/photo.jpg.png")

    def test_resume_and_changed_settings(self):
        first = self.pair()
        original_mtime = Path(first["image"]).stat().st_mtime_ns
        self.assertTrue(self.pair()["resumed"])
        self.assertEqual(Path(first["image"]).stat().st_mtime_ns, original_mtime)
        with self.assertRaises(FileExistsError):
            self.pair(settings={**self.settings, "divisor": 32})
        regenerated = self.pair(settings={**self.settings, "divisor": 32}, overwrite=True)
        self.assertFalse(regenerated["resumed"])

    def test_foreign_folder_and_foreign_file_are_protected(self):
        destination = self.root / "training_dataset"
        destination.mkdir()
        original = destination / "archive.png"
        original.write_bytes(b"archive")
        with self.assertRaises(ValueError):
            self.pair(overwrite=True)
        self.assertEqual(original.read_bytes(), b"archive")
        original.unlink()
        exporter.prepare_dataset_root(destination)
        target, _, _ = exporter.export_paths(destination, self.source, self.root, "portraits/photo.jpg", "PNG")
        target.parent.mkdir(parents=True)
        target.write_bytes(b"untracked")
        with self.assertRaises(FileExistsError):
            self.pair(overwrite=True)
        self.assertEqual(target.read_bytes(), b"untracked")

    def test_destination_containing_input_is_rejected(self):
        with self.assertRaises(ValueError):
            exporter.prepare_dataset_root(self.root, self.source)

    def test_generated_images_excluded_from_every_witness_scan(self):
        self.pair()
        for family in ("joy", "qwen", "ollama"):
            module = importlib.import_module(f"CaptionForge.nodes.caption_nodes.jlc_captionforge_{family}_caption_node")
            paths = module._iter_input_path_images(str(self.root), True, "*")
            self.assertEqual([entry[2] for entry in paths], [self.source])

    def test_explicit_generated_dataset_root_is_valid_witness_input(self):
        exported = self.pair()
        dataset_root = self.root / "training_dataset"
        exported_image = Path(exported["image"])

        for family in ("joy", "qwen", "ollama"):
            module = importlib.import_module(f"CaptionForge.nodes.caption_nodes.jlc_captionforge_{family}_caption_node")

            # A generated dataset discovered under a broader source root remains excluded.
            parent_scan = module._iter_input_path_images(str(self.root), True, "*")
            self.assertEqual([entry[2] for entry in parent_scan], [self.source])

            # But explicitly selecting that generated dataset is an intentional user action
            # and must make its images available for recaptioning/reprocessing.
            explicit_scan = module._iter_input_path_images(str(dataset_root), True, "*")
            self.assertEqual([entry[2] for entry in explicit_scan], [exported_image])

    def test_jpeg_and_optional_image_namespace(self):
        result = self.pair(image_key="captionforge-optional-image://comfy_image_0000.png",
                           settings={**self.settings, "image_format": "JPEG", "caption": "taggy"})
        self.assertIn("optional", Path(result["image"]).parts)
        with Image.open(result["image"]) as image:
            self.assertEqual(image.format, "JPEG")
        self.assertEqual(Path(result["caption"]).read_text().strip(), "portrait")

    def run_pipeline(self, *, plan=None, overwrite=True, validator_side_effect=None, **extra):
        caption_path = self.root / "witness.jsonl"
        caption_path.write_text(json.dumps({"image": str(self.source), "image_key": "portraits/photo.jpg",
                                            "caption": "A portrait.", "model_family": "joy", "status": "ok"}) + "\n")
        kwargs = {"Input - captions JSONL": str(caption_path), "Input - image path": str(self.root),
                  "Output - folder": str(self.root), "Output - run name": "prototype",
                  "Output - overwrite outputs": overwrite, "Validator - max image size": 256,
                  "Dataset - export image and caption": True, "pipeline_plan": plan}
        kwargs.update(extra)
        with mock.patch.object(orchestrator, "_evict_python_models_before_ollama_if_needed"), \
             mock.patch.object(orchestrator, "_ollama_generate_text", return_value=("SHORT: A portrait.\nTAGGY: portrait", {})) as text, \
             mock.patch.object(orchestrator, "_ollama_chat_image", return_value=("A long portrait.", {}), side_effect=validator_side_effect) as validator:
            result = orchestrator.JLC_CaptionForge().forge(**kwargs)
        return json.loads(result[3])["records"][0], text.call_count, validator.call_count

    def test_standalone_export_and_resume_without_model_calls(self):
        record, _, _ = self.run_pipeline()
        self.assertEqual(record["status"], "ok", record)
        self.assertEqual(record["dataset_export"]["width"], 256)
        self.assertTrue(Path(record["outputs"]["short"]).is_relative_to(self.root / "training_dataset"))
        self.assertFalse(self.source.with_name("photo_short.txt").exists())
        resumed, text_calls, validator_calls = self.run_pipeline(overwrite=False)
        self.assertEqual((text_calls, validator_calls), (0, 0))
        self.assertTrue(resumed["dataset_export"]["resumed"])

    def test_planner_controls_override_every_local_setting_including_blank_folder(self):
        _, plan, _ = planner.JLC_CaptionForge_Pipeline_Planner().plan(**{
            "Input - image path": str(self.root), "Output - folder": str(self.root),
            "Output - run name": "prototype", "Caption - max image size": 384,
            "Dataset - export image and caption": True, "Dataset - dimension divisor": 32,
            "Dataset - caption": "long", "Dataset - image format": "JPEG", "Dataset - JPEG quality": 88,
        })
        plan["paths"]["a_raw_captions_jsonl"] = str(self.root / "witness.jsonl")
        # The canonical plan may expose more than one compatible ledger key.
        for key in plan["paths"]:
            if "caption" in key.lower() and "jsonl" in key.lower():
                plan["paths"][key] = str(self.root / "witness.jsonl")
        record, _, _ = self.run_pipeline(plan=plan, **{
            "Dataset - export image and caption": False,
            "Dataset - output folder": str(self.root / "wrong"), "Dataset - max image size": 128,
            "Dataset - dimension divisor": 1, "Dataset - caption": "taggy",
            "Dataset - image format": "PNG", "Dataset - JPEG quality": 1,
        })
        self.assertEqual(record["status"], "ok", record)
        exported = record["dataset_export"]
        self.assertTrue(Path(exported["image"]).is_relative_to(self.root / "training_dataset"))
        self.assertEqual((exported["width"], exported["height"]), (384, 256))
        self.assertEqual(exported["caption_style"], "long")
        self.assertTrue(exported["image"].endswith(".jpg"))
        self.assertFalse((self.root / "wrong").exists())
        receipt = json.loads(Path(exported["image"]).with_suffix(".captionforge.json").read_text())
        self.assertEqual(receipt["signature"]["jpeg_quality"], 88)

    def test_old_or_disabled_planner_cannot_enable_local_export(self):
        record, _, _ = self.run_pipeline(plan={"captionforge_config_type": "captionforge_pipeline_plan"})
        self.assertEqual(record["status"], "ok", record)
        self.assertNotIn("dataset_export", record)
        self.assertFalse((self.root / "training_dataset").exists())

    def test_export_backfills_completed_captions_without_model_calls(self):
        first, _, _ = self.run_pipeline(**{"Dataset - export image and caption": False})
        self.assertNotIn("dataset_export", first)
        record, text_calls, validator_calls = self.run_pipeline(overwrite=False)
        self.assertEqual(record["dataset_export"]["status"], "ok")
        self.assertEqual((text_calls, validator_calls), (0, 0))

    def test_failed_export_preserves_captions_and_can_retry_without_models(self):
        record, _, _ = self.run_pipeline(**{"Dataset - dimension divisor": 512})
        self.assertEqual(record["status"], "error")
        self.assertEqual(record["error_stage"], "dataset_export")
        self.assertEqual(record["long"], "A long portrait.")
        recovered, text_calls, validator_calls = self.run_pipeline(overwrite=False)
        self.assertEqual(recovered["status"], "ok", recovered)
        self.assertEqual((text_calls, validator_calls), (0, 0))

    def test_validator_retry_does_not_reduce_training_export(self):
        error = urllib.error.HTTPError("http://localhost/api/chat", 413, "Too large", {}, io.BytesIO())
        record, _, calls = self.run_pipeline(
            validator_side_effect=[error, ("A long portrait.", {})],
            **{"Validator - max image size": 768},
        )
        self.assertEqual(calls, 2)
        self.assertEqual((record["dataset_export"]["width"], record["dataset_export"]["height"]), (768, 512))

    def test_interrupted_publication_requires_overwrite_and_recovers(self):
        replace = exporter.os.replace
        counter = 0

        def interrupted(source, target):
            nonlocal counter
            counter += 1
            if counter == 2:
                raise OSError("Interrupted caption publication")
            return replace(source, target)

        with mock.patch.object(exporter.os, "replace", side_effect=interrupted):
            with self.assertRaises(OSError):
                self.pair()
        with self.assertRaises(FileExistsError):
            self.pair()
        self.assertEqual(self.pair(overwrite=True)["status"], "ok")
        self.assertEqual(self.source.read_bytes(), self.original_bytes)

    def test_hardlink_to_source_cannot_be_overwritten(self):
        destination = self.root / "training_dataset"
        exporter.prepare_dataset_root(destination)
        target, _, _ = exporter.export_paths(destination, self.source, self.root, "portraits/photo.jpg", "PNG")
        target.parent.mkdir(parents=True)
        target.hardlink_to(self.source)
        with self.assertRaises(ValueError):
            self.pair(overwrite=True)
        self.assertEqual(self.source.read_bytes(), self.original_bytes)

    def test_caption_variant_hardlinks_and_untracked_files_are_protected(self):
        destination = self.root / "training_dataset"
        exporter.prepare_dataset_root(destination)
        target, _, _ = exporter.export_paths(destination, self.source, self.root, "portraits/photo.jpg", "PNG")
        target.parent.mkdir(parents=True)
        variant = target.with_name(target.stem + "_long.txt")
        variant.hardlink_to(self.source)
        with self.assertRaises(ValueError):
            self.pair(overwrite=True, write_variants=True)
        self.assertEqual(self.source.read_bytes(), self.original_bytes)
        variant.unlink()
        variant.write_text("My existing caption")
        with self.assertRaises(FileExistsError):
            self.pair(overwrite=True, write_variants=True)
        self.assertEqual(variant.read_text(), "My existing caption")

    def test_format_change_cannot_leave_duplicate_training_images(self):
        original = self.pair()
        with self.assertRaises(FileExistsError):
            self.pair(settings={**self.settings, "image_format": "JPEG"}, overwrite=True)
        self.assertTrue(Path(original["image"]).exists())
        self.assertFalse(Path(original["image"]).with_suffix(".jpg").exists())

    def test_changed_variant_cannot_silently_resume(self):
        first = self.pair(write_variants=True)
        Path(first["variants"]["long"]).write_text("Changed caption")
        with self.assertRaises(FileExistsError):
            self.pair(write_variants=True)
        self.assertEqual(self.pair(write_variants=True, overwrite=True)["status"], "ok")

    def test_canonical_workflows_serialize_dataset_widgets(self):
        directory = ROOT / "assets" / "workflows"
        ui = json.loads((directory / "CaptionForge_FullWorkflow_Rel_v1.0.2.json").read_text(encoding="utf-8"))
        api = json.loads((directory / "CaptionForge_FullWorkflow_API_Rel_v1.0.2.json").read_text(encoding="utf-8"))
        for name, cls in (("JLC_CaptionForge", orchestrator.JLC_CaptionForge),
                          ("JLC_CaptionForge_Pipeline_Planner", planner.JLC_CaptionForge_Pipeline_Planner)):
            node = next(item for item in ui["nodes"] if item["type"] == name)
            api_node = next(item for item in api.values() if item["class_type"] == name)
            names = list(cls.INPUT_TYPES()["required"]) + list(exporter.dataset_export_inputs())
            self.assertEqual(node["widgets_values"], [node["widgets_values_named"][key] for key in names])
            for key in names:
                self.assertEqual(api_node["inputs"][key], node["widgets_values_named"][key])


if __name__ == "__main__":
    unittest.main()

