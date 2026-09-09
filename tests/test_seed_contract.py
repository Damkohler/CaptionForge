from __future__ import annotations

import importlib
import json
import random
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
    folder_paths.get_output_directory = lambda: str(ROOT / "output")
    sys.modules["folder_paths"] = folder_paths

planner_engine = importlib.import_module("CaptionForge.engines.captionforge_pipeline_planner_engine")
planner_node = importlib.import_module("CaptionForge.nodes.jlc_captionforge_pipeline_planner_node")
capstone = importlib.import_module("CaptionForge.nodes.jlc_captionforge_node")
ollama_caption = importlib.import_module("CaptionForge.nodes.caption_nodes.jlc_captionforge_ollama_caption_node")


class PassASeedContractTests(unittest.TestCase):
    def _schedule(self, base_seed: int, seed_mode: str, runs: int = 3) -> list[int | None]:
        plan = planner_engine.build_captionforge_pipeline_plan(
            output_dir="unused",
            joy_runs_per_image=runs,
            qwen_runs_per_image=0,
            ollama_runs_per_image=0,
            base_seed=base_seed,
            seed_mode=seed_mode,
        )
        return [run.seed for run in planner_engine.expand_captionforge_runs(plan, model_key="joy")]

    def test_fixed_increment_and_decrement(self) -> None:
        self.assertEqual(self._schedule(100, "fixed"), [100, 100, 100])
        self.assertEqual(self._schedule(100, "increment"), [100, 101, 102])
        self.assertEqual(self._schedule(100, "decrement"), [100, 99, 98])

    def test_uint32_wraparound(self) -> None:
        maximum = planner_engine.MAX_SEED_32
        self.assertEqual(self._schedule(maximum - 1, "increment"), [maximum - 1, maximum, 0])
        self.assertEqual(self._schedule(0, "decrement"), [0, maximum, maximum - 1])

    def test_literal_zero_survives(self) -> None:
        self.assertEqual(self._schedule(0, "fixed"), [0, 0, 0])
        self.assertEqual(self._schedule(0, "increment"), [0, 1, 2])
        self.assertEqual(self._schedule(0, "random"), self._schedule(0, "random"))
        self.assertTrue(all(seed is not None for seed in self._schedule(0, "random")))

    def test_hash_random_has_a_stable_golden_schedule(self) -> None:
        expected = [90425488, 1231494799, 4241276224, 2828971892]
        self.assertEqual(self._schedule(123, "random", 4), expected)
        self.assertEqual(self._schedule(123, "random", 4), expected)
        self.assertEqual(len(set(expected)), len(expected))

    def test_hash_random_is_independent_of_global_random_state(self) -> None:
        state = random.getstate()
        try:
            random.seed(1)
            first = self._schedule(321, "random", 5)
            for _ in range(100):
                random.random()
            random.seed(999999)
            second = self._schedule(321, "random", 5)
        finally:
            random.setstate(state)
        self.assertEqual(first, second)

    def test_schedule_is_image_ordinal_independent_and_reusable(self) -> None:
        expected = self._schedule(100, "increment", 3)
        per_image = [self._schedule(100, "increment", 3) for _ in range(4)]
        self.assertEqual(per_image, [expected, expected, expected, expected])

    def test_minus_one_is_unseeded_for_every_mode(self) -> None:
        for mode in ("fixed", "increment", "decrement", "random"):
            with self.subTest(mode=mode):
                self.assertEqual(self._schedule(-1, mode), [None, None, None])

    def test_joy_qwen_and_ollama_share_the_planner_schedule(self) -> None:
        plan = planner_engine.build_captionforge_pipeline_plan(
            output_dir="unused",
            joy_runs_per_image=3,
            qwen_runs_per_image=3,
            ollama_runs_per_image=3,
            base_seed=77,
            seed_mode="random",
        )
        schedules = [
            [run.seed for run in planner_engine.expand_captionforge_runs(plan, model_key=family)]
            for family in ("joy", "qwen", "ollama")
        ]
        self.assertEqual(schedules[0], schedules[1])
        self.assertEqual(schedules[1], schedules[2])


class OllamaPassAUnsetSeedTests(unittest.TestCase):
    def test_unset_seed_is_omitted_from_ollama_options(self) -> None:
        options = ollama_caption._ollama_options(
            max_new_tokens=10,
            temperature=0.5,
            top_p=0.9,
            top_k=20,
            repetition_penalty=1.0,
            seed=None,
        )
        self.assertNotIn("seed", options)

    def test_planned_unset_seed_reaches_caption_call_without_int_none(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            Image.new("RGB", (2, 2), "white").save(root / "image.png")
            plan = planner_engine.build_captionforge_pipeline_plan(
                output_dir=temp_dir,
                input_path=temp_dir,
                joy_runs_per_image=0,
                qwen_runs_per_image=0,
                ollama_runs_per_image=1,
                base_seed=-1,
                seed_mode="random",
            )
            required = ollama_caption.JLC_CaptionForgeOllamaCaption.INPUT_TYPES()["required"]
            kwargs = {name: spec[1]["default"] for name, spec in required.items()}
            kwargs["pipeline_plan"] = plan

            with mock.patch.object(ollama_caption, "_ensure_ollama_model"), mock.patch.object(
                ollama_caption, "_persist_caption_model_if_possible"
            ), mock.patch.object(
                ollama_caption, "_evict_python_models_before_ollama_if_needed"
            ), mock.patch.object(
                ollama_caption, "_ollama_generate_caption", return_value="caption"
            ) as generate:
                ollama_caption.JLC_CaptionForgeOllamaCaption().caption(**kwargs)

            self.assertIsNone(generate.call_args.kwargs["seed"])


class StageSeedContractTests(unittest.TestCase):
    def _run_capstone(
        self,
        *,
        standalone_seeds: tuple[int | None, int | None, int | None],
        plan_seeds: tuple[int, int, int] | None = None,
    ) -> tuple[list[int | None], list[int | None]]:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            for name in ("one.png", "two.png"):
                Image.new("RGB", (2, 2), "white").save(root / name)
            caption_path = root / "captions.jsonl"
            caption_path.write_text(
                "\n".join(
                    json.dumps(
                        {
                            "image": name,
                            "image_key": Path(name).stem,
                            "caption": f"caption for {name}",
                            "model_family": "joy",
                            "status": "ok",
                        }
                    )
                    for name in ("one.png", "two.png")
                )
                + "\n",
                encoding="utf-8",
            )

            kwargs = {
                "Input - captions JSONL": str(caption_path),
                "Input - image path": temp_dir,
                "Output - folder": str(root / "output"),
                "Output - run name": "seed-test",
                "Output - overwrite outputs": True,
                "Distiller seed": standalone_seeds[0],
                "Validator seed": standalone_seeds[1],
                "Formatter seed": standalone_seeds[2],
            }
            if plan_seeds is not None:
                kwargs["pipeline_plan"] = {
                    "captionforge_config_type": "captionforge_pipeline_plan",
                    "distiller": {"seed": plan_seeds[0]},
                    "validator": {"seed": plan_seeds[1]},
                    "formatter": {"seed": plan_seeds[2]},
                }

            text_seeds: list[int | None] = []
            image_seeds: list[int | None] = []

            def fake_text(**call_kwargs):
                text_seeds.append(call_kwargs["seed"])
                return "generated text", {"ok": True}

            def fake_image(**call_kwargs):
                image_seeds.append(call_kwargs["seed"])
                return "validated natural caption", {"ok": True}

            with mock.patch.object(capstone, "_evict_python_models_before_ollama_if_needed"), mock.patch.object(
                capstone, "_ollama_generate_text", side_effect=fake_text
            ), mock.patch.object(capstone, "_ollama_chat_image", side_effect=fake_image):
                capstone.JLC_CaptionForge().forge(**kwargs)

            return text_seeds, image_seeds

    def test_standalone_exact_stage_seeds_repeat_for_every_image(self) -> None:
        text_seeds, image_seeds = self._run_capstone(standalone_seeds=(200, 300, 400))
        self.assertEqual(text_seeds, [200, 400, 200, 400])
        self.assertEqual(image_seeds, [300, 300])

    def test_planner_overrides_all_standalone_stage_seed_inputs(self) -> None:
        text_seeds, image_seeds = self._run_capstone(
            standalone_seeds=(1, 2, 3),
            plan_seeds=(200, 300, 400),
        )
        self.assertEqual(text_seeds, [200, 400, 200, 400])
        self.assertEqual(image_seeds, [300, 300])

    def test_literal_zero_survives_all_stage_paths(self) -> None:
        text_seeds, image_seeds = self._run_capstone(
            standalone_seeds=(9, 9, 9),
            plan_seeds=(0, 0, 0),
        )
        self.assertEqual(text_seeds, [0, 0, 0, 0])
        self.assertEqual(image_seeds, [0, 0])

    def test_omitted_standalone_seeds_are_unseeded_and_do_not_crash(self) -> None:
        text_seeds, image_seeds = self._run_capstone(standalone_seeds=(None, None, None))
        self.assertEqual(text_seeds, [None, None, None, None])
        self.assertEqual(image_seeds, [None, None])


class SeedOwnershipSchemaTests(unittest.TestCase):
    def test_planner_preserves_three_exact_stage_seeds_without_modes(self) -> None:
        plan = planner_engine.build_captionforge_pipeline_plan(
            output_dir="unused",
            joy_runs_per_image=1,
            qwen_runs_per_image=0,
            distiller_seed=200,
            validator_seed=300,
            formatter_seed=400,
        )
        self.assertEqual(plan["distiller"]["seed"], 200)
        self.assertEqual(plan["validator"]["seed"], 300)
        self.assertEqual(plan["formatter"]["seed"], 400)
        self.assertNotIn("seed_mode", plan["distiller"])
        self.assertNotIn("seed_mode", plan["validator"])
        self.assertNotIn("seed_mode", plan["formatter"])

    def test_planner_input_schema_has_fixed_stage_seeds_only(self) -> None:
        required = planner_node.JLC_CaptionForge_Pipeline_Planner.INPUT_TYPES()["required"]
        self.assertIn("Distiller - seed", required)
        self.assertIn("Validator - seed", required)
        self.assertIn("Formatter - seed", required)
        self.assertNotIn("Distiller - seed mode", required)
        self.assertNotIn("Validator - seed mode", required)
        self.assertNotIn("Formatter - seed mode", required)

    def test_capstone_exposes_only_optional_stage_seed_inputs(self) -> None:
        input_types = capstone.JLC_CaptionForge.INPUT_TYPES()
        required = input_types["required"]
        optional = input_types["optional"]
        self.assertIn("Distiller seed", optional)
        self.assertIn("Validator seed", optional)
        self.assertIn("Formatter seed", optional)
        for name in ("Distiller seed", "Validator seed", "Formatter seed"):
            self.assertTrue(optional[name][1]["forceInput"])
        for old_name in (
            "Fat Draft - base seed",
            "Fat Draft - seed mode",
            "Validator - base seed",
            "Validator - seed mode",
            "Formatter - base seed",
            "Formatter - seed mode",
        ):
            self.assertNotIn(old_name, required)
            self.assertNotIn(old_name, optional)


if __name__ == "__main__":
    unittest.main()
