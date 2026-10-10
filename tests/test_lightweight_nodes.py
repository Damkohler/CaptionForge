"""CPU-only contract tests for CaptionForge's standalone lightweight nodes."""

from __future__ import annotations

import importlib
import io
import json
import math
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


    def test_all_nonempty_assembles_distant_sections_in_order(self):
        image = object()
        node = light.CaptionForgeCaptionAssemble()
        result = node.assemble(
            image, 1, "newline", "",
            slot_mode="all_nonempty",
            manual_1="Character: JessJenn.",
            manual_2="   ",
            manual_5="Fallback pose.",
            text_5="Hands behind head.",
            manual_6="Fallback outfit.",
            text_6="Black crop top.",
            enabled_7=False,
            manual_7="This should be excluded.",
            manual_9="Photorealistic.",
            manual_10="",
        )
        self.assertIs(result[0], image)
        self.assertEqual(
            result[1],
            "Character: JessJenn.\nHands behind head.\nBlack crop top.\nPhotorealistic.",
        )

    def test_all_nonempty_empty_connected_input_uses_manual_fallback(self):
        node = light.CaptionForgeCaptionAssemble()
        result = node.assemble(
            None, 1, "space", "", slot_mode="all_nonempty",
            manual_4="standing", text_4="  ", manual_8="outside",
        )
        self.assertEqual(result[1], "standing outside")

    def test_first_n_still_excludes_later_sections(self):
        node = light.CaptionForgeCaptionAssemble()
        result = node.assemble(
            None, 2, "space", "", slot_mode="first_n",
            manual_1="first", manual_3="third", text_10="last",
        )
        self.assertEqual(result[1], "first")

    def test_slot_mode_appended_and_socket_names_unchanged(self):
        spec = light.CaptionForgeCaptionAssemble.INPUT_TYPES()
        required = list(spec["required"])
        self.assertEqual(required[-1], "slot_mode")
        self.assertLess(required.index("manual_10"), required.index("slot_mode"))
        self.assertEqual(spec["required"]["slot_mode"][1]["default"], "first_n")
        self.assertEqual(list(spec["optional"]), [f"text_{i}" for i in range(1, 11)])
        self.assertEqual(light.CaptionForgeCaptionAssemble.RETURN_NAMES, ("image", "caption"))

    def test_all_nonempty_with_no_content_is_empty_string(self):
        result = light.CaptionForgeCaptionAssemble().assemble(
            object(), 1, "custom", "++", slot_mode="all_nonempty",
        )
        self.assertEqual(result[1], "")



class TargetedQueryTests(unittest.TestCase):
    def setUp(self):
        with light._SUCCESS_CACHE_LOCK:
            light._SUCCESS_CACHE.clear()

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
                    "", 0.2, 100, 1024, 30,
                    retry_count=0, error_policy="stop_on_error",
                    question_1="Clothing?"
                )


    def test_comfy_force_regenerate_cache_invalidation(self):
        self.assertIsNone(light.CaptionForgeTargetedVLMQuery.IS_CHANGED(force_regenerate=False))
        self.assertTrue(math.isnan(light.CaptionForgeTargetedVLMQuery.IS_CHANGED(force_regenerate=True)))

    def test_seed_controls_appended_after_old_question_widgets(self):
        spec = light.CaptionForgeTargetedVLMQuery.INPUT_TYPES()["required"]
        fields = list(spec)
        self.assertLess(fields.index("question_10"), fields.index("seed"))
        self.assertEqual(spec["seed"][1]["default"], -1)
        self.assertTrue(spec["seed"][1]["control_after_generate"])
        self.assertIn("force_regenerate", fields)

    def test_ollama_payload_default_and_explicit_seed_thinking(self):
        def call(**extra):
            result = json.dumps({
                "message": {"content": "blue jeans"},
                "done_reason": "stop",
                "eval_count": 12,
            }).encode("utf-8")
            with mock.patch.object(light.urllib.request, "urlopen", return_value=io.BytesIO(result)) as open_url:
                answer = light._ask_ollama(
                    url="http://127.0.0.1:11434", model="gemma4:26b",
                    question="Outfit?", system_prompt="", image_base64="img",
                    temperature=0.3, max_new_tokens=512, timeout_seconds=30,
                    **extra,
                )
                request = open_url.call_args.args[0]
                return answer, json.loads(request.data)

        answer, payload = call()
        self.assertEqual(answer, "blue jeans")
        self.assertNotIn("seed", payload["options"])
        self.assertNotIn("think", payload)
        answer, payload = call(seed=42, thinking_mode="disabled")
        self.assertEqual(payload["options"]["seed"], 42)
        self.assertIs(payload["think"], False)

    def test_empty_reply_reports_metadata_not_thinking_text(self):
        result = json.dumps({
            "message": {"content": "", "thinking": "HIDDEN_NOT_A_CAPTION"},
            "done_reason": "length", "eval_count": 512,
        }).encode("utf-8")
        with mock.patch.object(light.urllib.request, "urlopen", return_value=io.BytesIO(result)):
            with self.assertRaises(light.OllamaEmptyAnswerError) as err:
                light._ask_ollama(
                    url="http://127.0.0.1:11434", model="gemma4:26b",
                    question="Pose?", system_prompt="", image_base64="img",
                    temperature=0.2, max_new_tokens=512, timeout_seconds=30,
                )
        self.assertIn("done_reason='length'", str(err.exception))
        self.assertIn("eval_count=512", str(err.exception))
        self.assertIn("thinking_chars=", str(err.exception))
        self.assertNotIn("HIDDEN_NOT_A_CAPTION", str(err.exception))

    def test_retries_only_empty_question_with_incremented_seed_and_token_budget(self):
        failed = light.OllamaEmptyAnswerError("empty")
        with mock.patch.object(light, "_encode_image", return_value="fake"), mock.patch.object(
            light, "_ask_ollama", side_effect=["blue outfit", failed, "hands raised"]
        ) as ask:
            output = light.CaptionForgeTargetedVLMQuery().query(
                object(), 2, "gemma4:26b", "", "http://127.0.0.1:11434",
                "", 0.2, 128, 1024, 30, seed=42, retry_count=1,
                question_1="Clothes?", question_2="Pose?",
            )
        self.assertEqual(output[1:3], ("blue outfit", "hands raised"))
        self.assertEqual(ask.call_count, 3)
        self.assertEqual([c.kwargs["seed"] for c in ask.call_args_list], [42, 42, 43])
        self.assertEqual([c.kwargs["max_new_tokens"] for c in ask.call_args_list], [128, 128, 256])

    def test_skip_failed_and_insert_unknown_do_not_abort(self):
        for policy, expected in (("skip_failed", ""), ("insert_unknown", "unknown")):
            with self.subTest(policy=policy):
                with mock.patch.object(light, "_encode_image", return_value="fake"), mock.patch.object(
                    light, "_ask_ollama",
                    side_effect=[
                        light.OllamaEmptyAnswerError("empty attempt 1"),
                        light.OllamaEmptyAnswerError("empty attempt 2"),
                        "pose okay",
                    ],
                ) as ask:
                    output = light.CaptionForgeTargetedVLMQuery().query(
                        object(), 2, "gemma4:26b", "", "http://127.0.0.1:11434",
                        "", 0.2, 128, 1024, 30,
                        retry_count=1, error_policy=policy,
                        cache_successful_answers=False,
                        question_1="Eyes?", question_2="Pose?",
                    )
                self.assertEqual(output[1], expected)
                self.assertEqual(output[2], "pose okay")
                self.assertEqual(ask.call_count, 3)

    def test_success_cache_preserves_completed_questions_and_regenerate_bypasses(self):
        node = light.CaptionForgeTargetedVLMQuery()
        def run(count, **changes):
            return node.query(
                object(), count, "gemma4:26b", "", "http://127.0.0.1:11434",
                "", 0.2, 128, 1024, 30,
                question_1="Outfit?", question_2="Pose?", **changes,
            )
        with mock.patch.object(light, "_encode_image", return_value="fake"), mock.patch.object(
            light, "_ask_ollama", side_effect=["outfit", "pose", "outfit new"]
        ) as ask:
            first = run(1)
            second = run(2)
            third = run(1, force_regenerate=True)
        self.assertEqual(first[1], "outfit")
        self.assertEqual(second[1:3], ("outfit", "pose"))
        self.assertEqual(third[1], "outfit new")
        self.assertEqual(ask.call_count, 3)

    def test_explicit_seed_change_causes_new_query(self):
        node = light.CaptionForgeTargetedVLMQuery()
        with mock.patch.object(light, "_encode_image", return_value="fake"), mock.patch.object(
            light, "_ask_ollama", side_effect=["red", "blue"]
        ) as ask:
            one = node.query(object(), 1, "gemma4:26b", "", "http://127.0.0.1:11434",
                             "", 0.5, 128, 1024, 30, seed=10, question_1="Color?")
            two = node.query(object(), 1, "gemma4:26b", "", "http://127.0.0.1:11434",
                             "", 0.5, 128, 1024, 30, seed=11, question_1="Color?")
        self.assertEqual((one[1], two[1]), ("red", "blue"))
        self.assertEqual(ask.call_count, 2)


    def test_all_nonempty_honors_sparse_questions_and_fixed_answer_pins(self):
        image = object()
        with mock.patch.object(light, "_encode_image", return_value="img") as encoder, mock.patch.object(
            light, "_ask_ollama", side_effect=["pose", "outfit", "background"]
        ) as ask:
            output = light.CaptionForgeTargetedVLMQuery().query(
                image, 1, "gemma4:26b", "", "http://127.0.0.1:11434",
                "", 0.2, 128, 1024, 30, slot_mode="all_nonempty",
                question_1="  ", question_2="", question_5="Pose?",
                question_6="Outfit?", question_7="Background?",
            )
        self.assertIs(output[0], image)
        self.assertEqual(len(output), 11)
        self.assertEqual(output[1:5], ("", "", "", ""))
        self.assertEqual(output[5:8], ("pose", "outfit", "background"))
        self.assertEqual(output[8:], ("", "", ""))
        self.assertEqual([c.kwargs["question"] for c in ask.call_args_list],
                         ["Pose?", "Outfit?", "Background?"])
        encoder.assert_called_once_with(image, 1024)

    def test_first_n_ignores_nonempty_question_above_limit(self):
        with mock.patch.object(light, "_encode_image") as encoder:
            output = light.CaptionForgeTargetedVLMQuery().query(
                object(), 4, "gemma4:26b", "", "http://127.0.0.1:11434",
                "", 0.2, 128, 1024, 30, slot_mode="first_n",
                question_5="Pose?",
            )
        self.assertEqual(output[1:], ("",) * 10)
        encoder.assert_not_called()

    def test_all_nonempty_blank_questions_do_not_load_image(self):
        with mock.patch.object(light, "_encode_image") as encoder:
            output = light.CaptionForgeTargetedVLMQuery().query(
                object(), 1, "gemma4:26b", "", "http://127.0.0.1:11434",
                "", 0.2, 128, 1024, 30, slot_mode="all_nonempty",
                question_5="   ", question_10="",
            )
        self.assertEqual(output[1:], ("",) * 10)
        encoder.assert_not_called()

    def test_query_slot_mode_appended_without_reindexing_questions(self):
        spec = light.CaptionForgeTargetedVLMQuery.INPUT_TYPES()["required"]
        self.assertEqual(list(spec)[-1], "slot_mode")
        self.assertLess(list(spec).index("force_regenerate"), list(spec).index("slot_mode"))
        self.assertEqual(spec["slot_mode"][1]["default"], "first_n")
        self.assertEqual(light.CaptionForgeTargetedVLMQuery.RETURN_NAMES[5], "answer_5")
        self.assertEqual(light.CaptionForgeTargetedVLMQuery.RETURN_NAMES[10], "answer_10")

    def test_invalid_slot_mode_fails_closed_before_network_work(self):
        for node in (light.CaptionForgeCaptionAssemble(), light.CaptionForgeTargetedVLMQuery()):
            with self.subTest(node=node.__class__.__name__):
                with self.assertRaisesRegex(ValueError, "Unknown slot_mode"):
                    if isinstance(node, light.CaptionForgeCaptionAssemble):
                        node.assemble(None, 1, "space", "", slot_mode="does-not-exist")
                    else:
                        with mock.patch.object(light, "_encode_image") as encoder:
                            node.query(
                                object(), 1, "gemma4:26b", "", "http://127.0.0.1:11434",
                                "", 0.2, 128, 1024, 30,
                                slot_mode="does-not-exist", question_5="Pose?",
                            )
                            encoder.assert_not_called()


if __name__ == "__main__":
    unittest.main()
