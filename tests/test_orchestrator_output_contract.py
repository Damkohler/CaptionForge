"""Focused tests for the CaptionForge Orchestrator output contract."""

from __future__ import annotations

import base64
import importlib
import io
import json
import sys
import tempfile
import types
import unittest
from contextlib import redirect_stdout
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

if "folder_paths" not in sys.modules:
    folder_paths = types.ModuleType("folder_paths")
    folder_paths.get_output_directory = lambda: str(ROOT / "output")
    sys.modules["folder_paths"] = folder_paths

planner_engine = importlib.import_module("CaptionForge.engines.captionforge_pipeline_planner_engine")
orchestrator = importlib.import_module("CaptionForge.nodes.jlc_captionforge_node")


class ValidatorImagePreparationTests(unittest.TestCase):
    def _prepare(self, size: tuple[int, int], max_size: int) -> tuple[tuple[int, int], str, Path]:
        temp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(temp_dir.cleanup)
        image_path = Path(temp_dir.name) / "source.jpg"
        Image.new("RGB", size, "white").save(image_path, format="JPEG", quality=95)

        output = io.StringIO()
        with redirect_stdout(output):
            encoded = orchestrator._pil_to_base64_png(image_path, max_size=max_size)
        with Image.open(io.BytesIO(base64.b64decode(encoded))) as transmitted:
            transmitted_size = transmitted.size
            transmitted_format = transmitted.format

        self.assertEqual(transmitted_format, "PNG")
        return transmitted_size, output.getvalue(), image_path

    def test_large_image_is_reduced_to_longest_side_maximum(self) -> None:
        transmitted_size, log, image_path = self._prepare((400, 200), max_size=100)

        self.assertEqual(transmitted_size, (100, 50))
        with Image.open(image_path) as source:
            self.assertEqual(source.size, (400, 200))
        self.assertIn("source=400x200", log)
        self.assertIn("validator=100x50", log)
        self.assertIn("encoded_payload=", log)
        self.assertIn("MiB", log)

    def test_resize_preserves_aspect_ratio(self) -> None:
        transmitted_size, _, _ = self._prepare((300, 500), max_size=100)

        self.assertEqual(transmitted_size, (60, 100))
        self.assertEqual(transmitted_size[0] / transmitted_size[1], 300 / 500)

    def test_image_below_maximum_is_not_enlarged(self) -> None:
        transmitted_size, _, _ = self._prepare((80, 40), max_size=100)

        self.assertEqual(transmitted_size, (80, 40))

    def test_zero_maximum_disables_resizing(self) -> None:
        transmitted_size, _, _ = self._prepare((400, 200), max_size=0)

        self.assertEqual(transmitted_size, (400, 200))


class OrchestratorOutputContractTests(unittest.TestCase):
    def test_public_output_surface_is_exactly_five_named_strings(self) -> None:
        self.assertEqual(
            orchestrator.JLC_CaptionForge.RETURN_NAMES,
            ("long_captions", "short_captions", "taggy_captions", "final_records", "status"),
        )
        self.assertEqual(orchestrator.JLC_CaptionForge.RETURN_TYPES, ("STRING",) * 5)

    def _run(self, *, planner_connected: bool) -> tuple[tuple[str, ...], Path]:
        temp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(temp_dir.cleanup)
        root = Path(temp_dir.name)
        image_names = ("one.png", "two.png")
        for name in image_names:
            Image.new("RGB", (2, 2), "white").save(root / name)

        caption_path = root / "captions.jsonl"
        caption_path.write_text(
            "\n".join(
                json.dumps(
                    {
                        "image": name,
                        "image_key": name,
                        "caption": f"Pass-A caption for {name}",
                        "model_family": "joy",
                        "status": "ok",
                    }
                )
                for name in image_names
            )
            + "\n",
            encoding="utf-8",
        )

        kwargs = {
            "Input - captions JSONL": str(caption_path),
            "Input - image path": str(root),
            "Output - folder": str(root / "standalone-output"),
            "Output - run name": "output-contract",
            "Output - overwrite outputs": True,
            "Final - write TXT sidecars": True,
            "Final - write JSONL": True,
        }
        if planner_connected:
            plan = planner_engine.build_captionforge_pipeline_plan(
                output_dir=str(root / "planned-output"),
                input_path=str(root),
                run_name="output-contract-planned",
                final_write_txt_sidecars=True,
                final_write_jsonl=True,
            )
            plan["paths"]["caption_jsonl"] = str(caption_path)
            plan["paths"]["pass_a_jsonl"] = str(caption_path)
            kwargs["pipeline_plan"] = plan

        text_results = iter(
            (
                "fat draft one",
                "SHORT: Short one.\nTAGGY: tag one, detail one",
                "fat draft two",
                "SHORT: Short two.\nTAGGY: tag two, detail two",
            )
        )
        validator_results = iter(("Long one.", "Long two."))

        def fake_text(**_kwargs):
            return next(text_results), {"ok": True}

        def fake_image(**_kwargs):
            return next(validator_results), {"ok": True}

        with mock.patch.object(
            orchestrator, "_evict_python_models_before_ollama_if_needed"
        ), mock.patch.object(
            orchestrator, "_ollama_generate_text", side_effect=fake_text
        ), mock.patch.object(
            orchestrator, "_ollama_chat_image", side_effect=fake_image
        ):
            result = orchestrator.JLC_CaptionForge().forge(**kwargs)

        return result, root

    def _assert_semantic_outputs(self, *, planner_connected: bool) -> None:
        result, root = self._run(planner_connected=planner_connected)
        self.assertEqual(len(result), 5)
        long_captions, short_captions, taggy_captions, final_records_json, status = result

        self.assertEqual(long_captions.split("\n\n"), ["Long one.", "Long two."])
        self.assertEqual(short_captions.split("\n\n"), ["Short one.", "Short two."])
        self.assertEqual(
            taggy_captions.split("\n\n"),
            ["tag one, detail one", "tag two, detail two"],
        )

        payload = json.loads(final_records_json)
        self.assertEqual(set(payload), {"records", "run_outputs"})
        self.assertEqual(len(payload["records"]), 2)
        self.assertEqual(
            [(record["long"], record["short"], record["taggy"]) for record in payload["records"]],
            [
                ("Long one.", "Short one.", "tag one, detail one"),
                ("Long two.", "Short two.", "tag two, detail two"),
            ],
        )

        for image_name, record in zip(("one.png", "two.png"), payload["records"], strict=True):
            self.assertEqual(record["image_key"], image_name)
            self.assertEqual(set(record["outputs"]), {"long", "short", "taggy"})
            self.assertEqual(len(record["sidecar_paths"]), 3)
            for path in record["outputs"].values():
                self.assertTrue(Path(path).is_file())
                self.assertEqual(Path(path).parent, root)

        run_outputs = payload["run_outputs"]
        for key in (
            "caption_jsonl",
            "pass_a_jsonl",
            "fat_draft_jsonl",
            "validator_jsonl",
            "taggy_jsonl",
            "final_jsonl",
            "output_paths_json",
            "image_root",
            "image_roots",
        ):
            self.assertIn(key, run_outputs)
        self.assertEqual(run_outputs["planner_connected"], planner_connected)

        self.assertIsInstance(status, str)
        self.assertIn("CaptionForge Orchestrator", status)
        self.assertIn(f"planner_connected={planner_connected}", status)
        self.assertIn("images=2", status)
        self.assertIn("final_ok=2 final_failed=0", status)

    def test_standalone_returns_aligned_semantic_products_and_records(self) -> None:
        self._assert_semantic_outputs(planner_connected=False)

    def test_planner_connected_returns_the_same_semantic_products(self) -> None:
        self._assert_semantic_outputs(planner_connected=True)

    def test_failed_item_keeps_an_empty_aligned_semantic_position(self) -> None:
        temp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(temp_dir.cleanup)
        root = Path(temp_dir.name)
        image_names = ("one.png", "two.png", "three.png")
        for name in image_names:
            Image.new("RGB", (2, 2), "white").save(root / name)

        caption_path = root / "captions.jsonl"
        source_records = (
            {
                "image": "one.png",
                "image_key": "one.png",
                "caption": "usable caption one",
                "model_family": "joy",
                "status": "ok",
            },
            {
                "image": "two.png",
                "image_key": "two.png",
                "caption": "FAILED: exception text must never become a caption",
                "model_family": "joy",
                "status": "error",
            },
            {
                "image": "three.png",
                "image_key": "three.png",
                "caption": "usable caption three",
                "model_family": "joy",
                "status": "ok",
            },
        )
        caption_path.write_text(
            "\n".join(json.dumps(record) for record in source_records) + "\n",
            encoding="utf-8",
        )

        text_results = iter(
            (
                "fat draft one",
                "SHORT: Short one.\nTAGGY: tag one",
                "fat draft three",
                "SHORT: Short three.\nTAGGY: tag three",
            )
        )
        validator_results = iter(("Long one.", "Long three."))

        with mock.patch.object(
            orchestrator, "_evict_python_models_before_ollama_if_needed"
        ), mock.patch.object(
            orchestrator,
            "_ollama_generate_text",
            side_effect=lambda **_kwargs: (next(text_results), {"ok": True}),
        ), mock.patch.object(
            orchestrator,
            "_ollama_chat_image",
            side_effect=lambda **_kwargs: (next(validator_results), {"ok": True}),
        ):
            result = orchestrator.JLC_CaptionForge().forge(
                **{
                    "Input - captions JSONL": str(caption_path),
                    "Input - image path": str(root),
                    "Output - folder": str(root / "output"),
                    "Output - run name": "failure-output-contract",
                    "Output - overwrite outputs": True,
                    "Final - write TXT sidecars": True,
                    "Final - write JSONL": True,
                }
            )

        long_captions, short_captions, taggy_captions, final_records_json, status = result
        semantic_entries = (
            long_captions.split("\n\n"),
            short_captions.split("\n\n"),
            taggy_captions.split("\n\n"),
        )
        self.assertEqual(semantic_entries[0], ["Long one.", "", "Long three."])
        self.assertEqual(semantic_entries[1], ["Short one.", "", "Short three."])
        self.assertEqual(semantic_entries[2], ["tag one", "", "tag three"])
        for entries in semantic_entries:
            self.assertEqual(len(entries), 3)
            self.assertEqual(entries[1], "")
            joined = " ".join(entries).lower()
            for diagnostic in ("failed", "failure", "error", "exception", "traceback"):
                self.assertNotIn(diagnostic, joined)

        records = json.loads(final_records_json)["records"]
        self.assertEqual(len(records), 3)
        self.assertEqual(records[1]["status"], "error")
        self.assertEqual(records[1]["error"], "no_usable_captions_selected")
        self.assertEqual(
            (records[1]["long"], records[1]["short"], records[1]["taggy"]),
            ("", "", ""),
        )
        self.assertIn("final_ok=2 final_failed=1", status)


if __name__ == "__main__":
    unittest.main()
