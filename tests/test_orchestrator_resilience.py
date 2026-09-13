"""Focused resilience tests for the CaptionForge Orchestrator."""

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

orchestrator = importlib.import_module("CaptionForge.nodes.jlc_captionforge_node")


def _http_error(status: int) -> urllib.error.HTTPError:
    return urllib.error.HTTPError(
        "http://127.0.0.1:11434/api/chat",
        status,
        "Request Entity Too Large" if status == 413 else "HTTP failure",
        {},
        io.BytesIO(b"error"),
    )


def _wrapped_http_error(status: int) -> RuntimeError:
    error = RuntimeError(f"HTTP {status} from Ollama")
    error.__cause__ = _http_error(status)
    return error


def _completed_record(image_key: str, *, status: str = "ok") -> dict:
    complete = status == "ok"
    return {
        "captionforge_pass": "D_FINAL_EXPORT",
        "image_key": image_key,
        "status": status,
        "final_caption": "Existing long." if complete else "",
        "long": "Existing long." if complete else "",
        "short": "Existing short." if complete else "",
        "taggy": "existing, tags" if complete else "",
    }


class OrchestratorResilienceTests(unittest.TestCase):
    def _run(
        self,
        image_keys: tuple[str, ...],
        *,
        overwrite: bool = True,
        max_size: int | None = None,
        existing_final_records: tuple[dict, ...] = (),
        text_side_effect=None,
        validator_side_effect=None,
    ) -> tuple[tuple[str, ...], Path, mock.Mock, mock.Mock, mock.Mock]:
        temp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(temp_dir.cleanup)
        root = Path(temp_dir.name)
        for image_key in image_keys:
            Image.new("RGB", (4, 2), "white").save(root / f"{image_key}.png")

        caption_path = root / "captions.jsonl"
        caption_path.write_text(
            "\n".join(
                json.dumps(
                    {
                        "image": f"{image_key}.png",
                        "image_key": image_key,
                        "caption": f"source caption for {image_key}",
                        "model_family": "joy",
                        "status": "ok",
                    }
                )
                for image_key in image_keys
            )
            + "\n",
            encoding="utf-8",
        )

        output_dir = root / "output"
        run_name = "resilience"
        plan: dict = {}
        if max_size is not None:
            plan = {
                "captionforge_config_type": "captionforge_pipeline_plan",
                "caption_settings": {"max_size": max_size},
            }
        paths = orchestrator._derive_paths(output_dir, run_name, plan)
        final_path = Path(paths["final_jsonl"])
        if existing_final_records:
            orchestrator._write_jsonl(final_path, list(existing_final_records), append=True)

        if text_side_effect is None:
            text_side_effect = lambda **_kwargs: (
                "SHORT: Generated short.\nTAGGY: generated, tags",
                {"ok": True},
            )
        if validator_side_effect is None:
            validator_side_effect = lambda **_kwargs: ("Generated long caption.", {"ok": True})

        kwargs = {
            "Input - captions JSONL": str(caption_path),
            "Input - image path": str(root),
            "Output - folder": str(output_dir),
            "Output - run name": run_name,
            "Output - overwrite outputs": overwrite,
            "Final - write TXT sidecars": False,
            "Final - write JSONL": True,
        }
        if plan:
            kwargs["pipeline_plan"] = plan

        original_prepare = orchestrator._pil_to_base64_png
        with mock.patch.object(
            orchestrator, "_evict_python_models_before_ollama_if_needed"
        ), mock.patch.object(
            orchestrator, "_ollama_generate_text", side_effect=text_side_effect
        ) as text_mock, mock.patch.object(
            orchestrator, "_ollama_chat_image", side_effect=validator_side_effect
        ) as validator_mock, mock.patch.object(
            orchestrator, "_pil_to_base64_png", wraps=original_prepare
        ) as prepare_mock:
            result = orchestrator.JLC_CaptionForge().forge(**kwargs)

        return result, final_path, text_mock, validator_mock, prepare_mock

    def test_middle_image_failure_is_isolated_and_later_image_completes(self) -> None:
        def text_response(**kwargs):
            if "source caption for middle" in kwargs["prompt"]:
                raise RuntimeError("synthetic distiller failure")
            return "SHORT: Generated short.\nTAGGY: generated, tags", {"ok": True}

        result, _, _, validator_mock, _ = self._run(
            ("first", "middle", "third"),
            text_side_effect=text_response,
        )

        payload = json.loads(result[3])
        records = payload["records"]
        self.assertEqual([record["status"] for record in records], ["ok", "error", "ok"])
        self.assertEqual(records[1]["image_key"], "middle")
        self.assertEqual(records[1]["error_stage"], "distiller")
        self.assertEqual(records[1]["error_type"], "RuntimeError")
        self.assertEqual(records[1]["error_message"], "synthetic distiller failure")
        self.assertEqual(validator_mock.call_count, 2)
        self.assertIn("final_ok=2 final_failed=1", result[4])

    def test_resume_skips_only_latest_complete_final_records(self) -> None:
        existing = (
            _completed_record("a", status="error"),
            _completed_record("a", status="ok"),
            _completed_record("b", status="ok"),
            _completed_record("b", status="error"),
        )
        result, final_path, text_mock, validator_mock, _ = self._run(
            ("a", "b", "c"),
            overwrite=False,
            existing_final_records=existing,
        )

        self.assertEqual(text_mock.call_count, 4)
        self.assertEqual(validator_mock.call_count, 2)
        self.assertIn("resume_skipped=1", result[4])
        self.assertEqual(json.loads(result[3])["run_outputs"]["resume_skipped"], 1)
        ledger = orchestrator._read_jsonl(final_path)
        successful_a = [
            record
            for record in ledger
            if record.get("image_key") == "a" and record.get("status") == "ok"
        ]
        self.assertEqual(len(successful_a), 1)
        self.assertEqual(ledger[-2]["image_key"], "b")
        self.assertEqual(ledger[-1]["image_key"], "c")

    def test_resume_completion_uses_latest_usable_record_and_ignores_bad_line(self) -> None:
        temp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(temp_dir.cleanup)
        ledger_path = Path(temp_dir.name) / "final.jsonl"
        partial_b = _completed_record("b")
        partial_b["short"] = ""
        records = (
            _completed_record("a", status="error"),
            _completed_record("a"),
            _completed_record("b"),
            partial_b,
            _completed_record("c"),
            _completed_record("c", status="error"),
        )
        orchestrator._write_jsonl(ledger_path, list(records), append=True)
        with ledger_path.open("a", encoding="utf-8") as ledger:
            ledger.write("{interrupted")

        completed = orchestrator._completed_final_records(ledger_path)

        self.assertEqual(set(completed), {"a"})

    def test_overwrite_true_ignores_existing_completion_and_runs_fresh(self) -> None:
        result, final_path, text_mock, validator_mock, _ = self._run(
            ("a", "b", "c"),
            overwrite=True,
            existing_final_records=(_completed_record("a"),),
        )

        self.assertEqual(text_mock.call_count, 6)
        self.assertEqual(validator_mock.call_count, 3)
        self.assertIn("resume_skipped=0", result[4])
        self.assertEqual(len(orchestrator._read_jsonl(final_path)), 3)

    def test_validator_413_retries_1024_then_succeeds_at_768(self) -> None:
        result, _, _, validator_mock, prepare_mock = self._run(
            ("image",),
            max_size=1024,
            validator_side_effect=(_wrapped_http_error(413), ("Recovered long caption.", {"ok": True})),
        )

        self.assertEqual(result[4].count("final_ok=1"), 1)
        self.assertEqual(validator_mock.call_count, 2)
        self.assertEqual(
            [call.kwargs["max_size"] for call in prepare_mock.call_args_list],
            [1024, 768],
        )

    def test_unrelated_validator_http_error_is_not_retried(self) -> None:
        result, _, _, validator_mock, prepare_mock = self._run(
            ("image",),
            max_size=1024,
            validator_side_effect=_wrapped_http_error(500),
        )

        record = json.loads(result[3])["records"][0]
        self.assertEqual(record["status"], "error")
        self.assertEqual(record["error_stage"], "validator")
        self.assertEqual(validator_mock.call_count, 1)
        self.assertEqual(len(prepare_mock.call_args_list), 1)

    def test_exhausted_413_retries_fail_one_image_and_continue(self) -> None:
        result, _, _, validator_mock, prepare_mock = self._run(
            ("first", "second"),
            max_size=768,
            validator_side_effect=(
                _wrapped_http_error(413),
                _wrapped_http_error(413),
                ("Second image completed.", {"ok": True}),
            ),
        )

        records = json.loads(result[3])["records"]
        self.assertEqual([record["status"] for record in records], ["error", "ok"])
        self.assertEqual(records[0]["error_stage"], "validator")
        self.assertEqual(records[1]["image_key"], "second")
        self.assertEqual(validator_mock.call_count, 3)
        self.assertEqual(
            [call.kwargs["max_size"] for call in prepare_mock.call_args_list],
            [768, 512, 768],
        )
        self.assertIn("final_ok=1 final_failed=1", result[4])

    def test_validator_retry_ladders_are_strictly_smaller(self) -> None:
        self.assertEqual(orchestrator._validator_image_retry_caps(0), (0, 1024, 768, 512))
        self.assertEqual(orchestrator._validator_image_retry_caps(1536), (1536, 1024, 768, 512))
        self.assertEqual(orchestrator._validator_image_retry_caps(1024), (1024, 768, 512))
        self.assertEqual(orchestrator._validator_image_retry_caps(768), (768, 512))
        self.assertEqual(orchestrator._validator_image_retry_caps(512), (512,))

    def test_explicit_user_abort_is_not_isolated(self) -> None:
        with self.assertRaises(KeyboardInterrupt):
            self._run(("image",), validator_side_effect=KeyboardInterrupt())


if __name__ == "__main__":
    unittest.main()
