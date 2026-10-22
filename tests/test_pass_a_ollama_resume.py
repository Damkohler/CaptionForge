"""Ollama Pass A restart tests. All model and network work is mocked."""

from __future__ import annotations

import copy
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
for name, path in (
    ("CaptionForge", ROOT), ("CaptionForge.engines", ROOT / "engines"),
    ("CaptionForge.nodes", ROOT / "nodes"),
    ("CaptionForge.nodes.caption_nodes", ROOT / "nodes" / "caption_nodes"),
):
    if name not in sys.modules:
        module = types.ModuleType(name)
        module.__path__ = [str(path)]
        sys.modules[name] = module
if "folder_paths" not in sys.modules:
    module = types.ModuleType("folder_paths")
    module.models_dir = str(ROOT / "models")
    module.get_output_directory = lambda: str(ROOT / "output")
    sys.modules["folder_paths"] = module

ollama = importlib.import_module("CaptionForge.nodes.caption_nodes.jlc_captionforge_ollama_caption_node")
planner = importlib.import_module("CaptionForge.engines.captionforge_pipeline_planner_engine")
planner_node = importlib.import_module("CaptionForge.nodes.jlc_captionforge_pipeline_planner_node")
orchestrator = importlib.import_module("CaptionForge.nodes.jlc_captionforge_node")
resume = importlib.import_module("CaptionForge.engines.captionforge_pass_a_resume")


class OllamaResumeTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.inputs = self.root / "inputs"
        self.inputs.mkdir()
        for name in ("a/photo.png", "b/photo.png", "c.png", "d.png"):
            path = self.inputs / name
            path.parent.mkdir(parents=True, exist_ok=True)
            Image.new("RGB", (2, 2), "white").save(path)
        self.image_bytes = {path: path.read_bytes() for path in self.inputs.rglob("*.png")}
        self.plan = planner.build_captionforge_pipeline_plan(
            output_dir=str(self.root / "output"), input_path=str(self.inputs),
            run_name="resume", overwrite_outputs=False,
            joy_runs_per_image=0, qwen_runs_per_image=0, ollama_runs_per_image=2,
            base_seed=123, seed_mode="increment",
        )
        self.path = Path(self.plan["paths"]["caption_jsonl"])
        self.kwargs = {
            name: spec[1]["default"]
            for name, spec in ollama.JLC_CaptionForgeOllamaCaption.INPUT_TYPES()["required"].items()
        }
        self.kwargs.update(pipeline_plan=self.plan, model="custom", custom_model_tag="witness-a:tag")
        for name in ("_ensure_ollama_model", "_persist_caption_model_if_possible",
                     "_evict_python_models_before_ollama_if_needed"):
            setattr(self, name, self.enterContext(mock.patch.object(ollama, name)))
        self.network = self.enterContext(mock.patch.object(ollama, "_http_json", side_effect=AssertionError("Live HTTP forbidden")))
        self.generate = self.enterContext(mock.patch.object(ollama, "_ollama_generate_caption", return_value="visible caption"))

    def tearDown(self):
        for path, original in self.image_bytes.items():
            self.assertEqual(path.read_bytes(), original)
        self.network.assert_not_called()

    def run_node(self, **changes):
        return ollama.JLC_CaptionForgeOllamaCaption().caption(**{**self.kwargs, **changes})

    def records(self):
        return resume.read_pass_a_jsonl(self.path)

    def write_records(self, records, *, newline=True):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_bytes(("\n".join(json.dumps(record) for record in records) + ("\n" if newline else "")).encode("utf-8"))

    def crash_after(self, count):
        self.generate.side_effect = ["visible caption"] * count + [RuntimeError("simulated restart")]
        with self.assertRaisesRegex(RuntimeError, "simulated restart"):
            self.run_node()
        self.generate.side_effect = None
        self.generate.reset_mock()

    def test_clean_first_execution_and_successful_resume(self):
        result = self.run_node()
        self.assertEqual(self.generate.call_count, 8)
        self.assertEqual(len(self.records()), 8)
        original = self.path.read_bytes()
        self.generate.reset_mock()
        self._ensure_ollama_model.reset_mock()
        self._persist_caption_model_if_possible.reset_mock()
        self._evict_python_models_before_ollama_if_needed.reset_mock()
        self.assertEqual(self.run_node(), result)
        self.generate.assert_not_called()
        self._ensure_ollama_model.assert_not_called()
        self._persist_caption_model_if_possible.assert_not_called()
        self._evict_python_models_before_ollama_if_needed.assert_not_called()
        self.assertEqual(self.path.read_bytes(), original)

    def test_restart_after_half_images(self):
        self.crash_after(4)
        prefix = self.path.read_bytes()
        self.run_node()
        self.assertEqual(self.generate.call_count, 4)
        self.assertTrue(self.path.read_bytes().startswith(prefix))
        self.assertEqual(len(self.records()), 8)

    def test_restart_midway_through_image(self):
        self.crash_after(3)
        self.assertEqual(len(self.records()), 3)
        self.run_node()
        self.assertEqual(self.generate.call_count, 5)
        self.assertEqual(self.generate.call_args_list[0].kwargs["seed"], 124)
        identities = [(r["image_key"], r["ensemble_run_index"]) for r in self.records()]
        self.assertEqual(len(set(identities)), 8)

    def test_distinct_model_nodes_resume_independently(self):
        self.run_node()
        self.generate.reset_mock()
        self.run_node(custom_model_tag="witness-b:tag")
        self.assertEqual(self.generate.call_count, 8)
        self.generate.reset_mock()
        self.run_node()
        self.run_node(custom_model_tag="witness-b:tag")
        self.generate.assert_not_called()
        self.assertEqual({r["model_name"] for r in self.records()}, {"witness-a:tag", "witness-b:tag"})
        self.assertEqual(len(self.records()), 16)

    def test_same_filename_in_recursive_folders(self):
        self.crash_after(1)
        self.run_node()
        records = self.records()
        self.assertEqual({r["image_key"] for r in records}, {"a/photo.png", "b/photo.png", "c.png", "d.png"})
        self.assertEqual(sum(r["image_key"] == "a/photo.png" for r in records), 2)
        self.assertEqual(sum(r["image_key"] == "b/photo.png" for r in records), 2)

    def test_failed_empty_and_missing_operations_retried(self):
        self.run_node()
        records = self.records()
        records[0]["status"] = "error"
        records[1]["caption"] = "  "
        records.pop(2)
        self.write_records(records)
        prefix = self.path.read_bytes()
        self.generate.reset_mock()
        self.run_node()
        self.assertEqual(self.generate.call_count, 3)
        self.assertTrue(self.path.read_bytes().startswith(prefix))
        self.assertEqual(len(self.records()), 10)

    def test_every_persisted_operation_field_invalidates_completion(self):
        self.run_node()
        original = self.records()
        changes = {
            "prompt": "changed prompt", "system_prompt": "changed system",
            "model_name": "other:tag", "seed": 999,
            "temperature": 0.01, "top_p": 0.01, "top_k": 1,
            "repetition_penalty": 1.9, "max_new_tokens": 16, "max_size": 64,
        }
        for field, value in changes.items():
            with self.subTest(field=field):
                records = copy.deepcopy(original)
                records[0][field] = value
                self.write_records(records)
                self.generate.reset_mock()
                self.run_node()
                self.assertEqual(self.generate.call_count, 1)

    def test_changed_node_prompt_model_and_parameter(self):
        self.run_node()
        original = self.records()
        for changes in (
            {"custom_prompt_mode": True, "custom_prompt": "changed"},
            {"system_prompt": "changed"}, {"custom_model_tag": "changed:tag"},
            {"repetition_penalty": 1.9},
        ):
            with self.subTest(changes=changes):
                self.write_records(original)
                self.generate.reset_mock()
                self.run_node(**changes)
                self.assertEqual(self.generate.call_count, 8)

    def test_cleanup_metadata_and_legacy_cleanup_effect(self):
        self.run_node()
        original = self.records()
        for legacy in (False, True):
            with self.subTest(legacy=legacy):
                records = copy.deepcopy(original)
                if legacy:
                    for record in records:
                        record.pop("resume_cleanup")
                self.write_records(records)
                self.generate.reset_mock()
                changed = copy.deepcopy(self.plan)
                changed["shared"]["trigger_word"] = "new-trigger"
                self.run_node(pipeline_plan=changed)
                self.assertEqual(self.generate.call_count, 8)

    def test_v102_without_new_metadata_resumes_and_stays_unchanged(self):
        self.run_node()
        legacy = self.records()
        for record in legacy:
            record.pop("resume_cleanup")
        self.write_records(legacy)
        original = self.path.read_bytes()
        self.generate.reset_mock()
        self.run_node()
        self.generate.assert_not_called()
        self.assertEqual(self.path.read_bytes(), original)

    def test_incomplete_identity_does_not_skip(self):
        self.run_node()
        records = self.records()
        records[0].pop("max_size")
        self.write_records(records)
        self.generate.reset_mock()
        self.run_node()
        self.assertEqual(self.generate.call_count, 1)

    def test_other_stages_and_missing_pass_do_not_qualify(self):
        self.run_node()
        original = self.records()
        for stage in ("B", "C", "D", None):
            with self.subTest(stage=stage):
                records = copy.deepcopy(original)
                if stage is None:
                    records[0].pop("captionforge_pass")
                else:
                    records[0]["captionforge_pass"] = stage
                self.assertIsNone(resume.ollama_operation_key(records[0]))
                self.write_records(records)
                self.generate.reset_mock()
                self.run_node()
                self.assertEqual(self.generate.call_count, 1)

    def test_legacy_without_usable_raw_caption_is_regenerated(self):
        self.run_node()
        original = self.records()
        for raw_caption in (None, "", "  ", 123):
            with self.subTest(raw_caption=raw_caption):
                records = copy.deepcopy(original)
                records[0].pop("resume_cleanup")
                if raw_caption is None:
                    records[0].pop("raw_caption")
                else:
                    records[0]["raw_caption"] = raw_caption
                self.write_records(records)
                self.generate.reset_mock()
                self.run_node()
                self.assertEqual(self.generate.call_count, 1)

    def test_truncated_tail_backed_up_before_recovery(self):
        self.crash_after(3)
        prefix = self.path.read_bytes()
        damaged = prefix + b'{"image_key":"b/photo.png","caption":"partial\xe2\x82'
        self.path.write_bytes(damaged)
        self.run_node()
        self.assertEqual(self.generate.call_count, 5)
        self.assertEqual(len(self.records()), 8)
        self.assertTrue(self.path.read_bytes().startswith(prefix))
        backups = list(self.path.parent.glob(self.path.name + ".*.bak"))
        self.assertEqual(len(backups), 1)
        self.assertEqual(backups[0].read_bytes(), damaged)

    def test_complete_final_object_without_newline_recovered(self):
        self.crash_after(3)
        records = self.records()
        self.write_records(records, newline=False)
        original = self.path.read_bytes()
        self.run_node()
        self.assertEqual(self.generate.call_count, 5)
        self.assertTrue(self.path.read_bytes().startswith(original + b"\n"))
        self.assertEqual(next(self.path.parent.glob(self.path.name + ".*.bak")).read_bytes(), original)

    def test_truncated_final_record_with_newline_is_backed_up(self):
        self.crash_after(2)
        prefix = self.path.read_bytes()
        original = prefix + b'{"caption":\n'
        self.path.write_bytes(original)
        self.run_node()
        self.assertEqual(self.generate.call_count, 6)
        self.assertEqual(len(self.records()), 8)
        self.assertTrue(self.path.read_bytes().startswith(prefix))
        self.assertEqual(next(self.path.parent.glob(self.path.name + ".*.bak")).read_bytes(), original)

    def test_non_object_record_fails_closed(self):
        for content in (b'[]\n', b'{}\nnull\n{}\n'):
            with self.subTest(content=content):
                self.path.parent.mkdir(parents=True, exist_ok=True)
                self.path.write_bytes(content)
                with self.assertRaisesRegex(RuntimeError, "Invalid Pass A JSONL object"):
                    self.run_node()
                self.assertEqual(self.path.read_bytes(), content)
        self.generate.assert_not_called()

    def test_changed_ledger_during_repair_is_not_overwritten(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        original = b'{}\n{"partial"'
        changed = original + b' concurrent append'
        self.path.write_bytes(original)
        with mock.patch.object(Path, "read_bytes", side_effect=[original, changed]):
            with self.assertRaisesRegex(RuntimeError, "changed during recovery"):
                resume.read_pass_a_jsonl(self.path, repair_tail=True)
        self.assertEqual(self.path.read_bytes(), original)
        self.assertEqual(next(self.path.parent.glob(self.path.name + ".*.bak")).read_bytes(), original)

    def test_middle_corruption_fails_closed_before_model_work(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        original = b'{"status":"ok"}\n{broken}\n{"status":"ok"}\n'
        self.path.write_bytes(original)
        with self.assertRaisesRegex(RuntimeError, r":2 \(byte 16\)"):
            self.run_node()
        self.generate.assert_not_called()
        self._ensure_ollama_model.assert_not_called()
        self.assertEqual(self.path.read_bytes(), original)
        self.assertEqual(list(self.path.parent.glob("*.bak")), [])
        with self.assertRaisesRegex(RuntimeError, ":2"):
            planner_node._reset_pass_a_jsonl_for_overwrite(self.plan, overwrite_outputs=False)
        with self.assertRaisesRegex(RuntimeError, ":2"):
            resume.read_pass_a_jsonl(self.path)

    def test_backup_failure_leaves_original_and_prevents_inference(self):
        self.crash_after(1)
        original = self.path.read_bytes() + b'{"partial"'
        self.path.write_bytes(original)
        real_open = Path.open
        def fail_backup(path, *args, **kwargs):
            if path.suffix == ".bak":
                raise PermissionError("backup denied")
            return real_open(path, *args, **kwargs)
        with mock.patch.object(Path, "open", fail_backup):
            with self.assertRaisesRegex(PermissionError, "backup denied"):
                self.run_node()
        self.assertEqual(self.path.read_bytes(), original)
        self.generate.assert_not_called()

    def test_planner_repairs_before_any_witness_appends(self):
        self.crash_after(2)
        original = self.path.read_bytes()
        self.path.write_bytes(original + b'{"partial"')
        planner_node._reset_pass_a_jsonl_for_overwrite(self.plan, overwrite_outputs=False)
        self.assertEqual(self.path.read_bytes(), original)
        self.assertEqual(len(self.records()), 2)

    def test_overwrite_true_preserves_planner_fresh_run_behavior(self):
        self.run_node()
        fresh = copy.deepcopy(self.plan)
        fresh["shared"]["overwrite_outputs"] = True
        planner_node._reset_pass_a_jsonl_for_overwrite(fresh, overwrite_outputs=True)
        self.assertFalse(self.path.exists())
        self.generate.reset_mock()
        self.run_node(pipeline_plan=fresh)
        self.assertEqual(self.generate.call_count, 8)
        self.assertEqual(len(self.records()), 8)
        self.assertEqual(list(self.path.parent.glob("*.bak")), [])

    def test_standalone_always_generates_and_remains_file_silent(self):
        with mock.patch.object(ollama, "_tensor_to_pil", return_value=[Image.new("RGB", (2, 2))]):
            first = self.run_node(pipeline_plan=None)
            second = self.run_node(pipeline_plan=None)
        self.assertEqual(first, second)
        self.assertEqual(self.generate.call_count, 2)
        self.assertFalse(self.path.exists())

    def test_orchestrator_deduplicates_before_limits_without_merging_runs_or_models(self):
        self.run_node()
        records = self.records()
        first = records[0]
        other_model = {**first, "model_name": "other:tag", "model_path": "other:tag"}
        repeated_run = {**first, "ensemble_run_index": 2}
        duplicates = [first, {**first, "timestamp": "later"}, records[1], other_model, repeated_run]
        selected = orchestrator._select_caption_records(
            duplicates, include_families="ollama", max_per_family=4, max_total=4,
        )
        self.assertEqual(selected, [duplicates[1], records[1], other_model, repeated_run])
        missing_identity = {**first}
        missing_identity.pop("ensemble_run_index")
        self.assertEqual(len(resume.deduplicate_ollama_completions([missing_identity] * 2)), 2)

    def test_legacy_and_new_duplicates_collapse_to_latest_success(self):
        self.run_node()
        current = self.records()[0]
        legacy = {**current}
        legacy.pop("resume_cleanup")
        updated = {**current, "caption": "updated cleanup caption"}
        failed = {**current, "status": "error"}
        selected = orchestrator._select_caption_records(
            [legacy, current, updated, failed], include_families="all", max_per_family=0, max_total=0,
        )
        self.assertEqual(selected, [updated])

    def test_orchestrator_receives_unique_evidence_and_tolerates_read_only_tail(self):
        self.run_node()
        records = self.records()
        self.write_records([records[0], {**records[0], "timestamp": "later"}, records[1]])
        original = self.path.read_bytes() + b'{"partial"'
        self.path.write_bytes(original)
        with mock.patch.object(orchestrator, "_evict_python_models_before_ollama_if_needed"), mock.patch.object(
            orchestrator, "_ollama_generate_text", return_value=("SHORT: short\nTAGGY: tag", {})
        ) as text, mock.patch.object(orchestrator, "_ollama_chat_image", return_value=("long caption", {})):
            result = orchestrator.JLC_CaptionForge().forge(**{
                "Input - captions JSONL": str(self.path), "Input - image path": str(self.inputs),
                "Output - folder": str(self.root / "orchestrator"), "Output - run name": "test",
                "Final - write TXT sidecars": False,
            })
        self.assertEqual(json.loads(result[3])["records"][0]["status"], "ok")
        distiller_prompt = text.call_args_list[0].kwargs["prompt"]
        self.assertEqual(distiller_prompt.count("family=ollama"), 2)
        self.assertEqual(self.path.read_bytes(), original)
        self.assertEqual(list(self.path.parent.glob("*.bak")), [])


if __name__ == "__main__":
    unittest.main()
