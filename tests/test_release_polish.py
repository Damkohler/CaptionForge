"""CPU-only release metadata, module documentation, and node-help checks."""

from __future__ import annotations

import ast
import importlib
import json
import re
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
elif not hasattr(sys.modules["folder_paths"], "models_dir"):
    sys.modules["folder_paths"].models_dir = str(ROOT / "models")

version_module = importlib.import_module("CaptionForge.captionforge_version")
planner = importlib.import_module("CaptionForge.nodes.jlc_captionforge_pipeline_planner_node")
capstone = importlib.import_module("CaptionForge.nodes.jlc_captionforge_node")
template_options = importlib.import_module("CaptionForge.nodes.jlc_captionforge_template_options")
joy = importlib.import_module("CaptionForge.nodes.caption_nodes.jlc_captionforge_joy_caption_node")
qwen = importlib.import_module("CaptionForge.nodes.caption_nodes.jlc_captionforge_qwen_caption_node")
ollama = importlib.import_module("CaptionForge.nodes.caption_nodes.jlc_captionforge_ollama_caption_node")


class ReleaseMetadataTests(unittest.TestCase):
    def test_current_release_versions_are_1_0_2(self) -> None:
        self.assertEqual(version_module.CAPTIONFORGE_VERSION, "1.0.2")
        pyproject = (ROOT / "pyproject.toml").read_text(encoding="utf-8")
        match = re.search(r'^version\s*=\s*"([^"]+)"', pyproject, re.MULTILINE)
        self.assertIsNotNone(match)
        self.assertEqual(match.group(1), version_module.CAPTIONFORGE_VERSION)

        config = json.loads(
            (ROOT / "config" / "captionforge_ollama_models.json").read_text(encoding="utf-8")
        )
        self.assertEqual(config["_meta"]["version"], version_module.CAPTIONFORGE_VERSION)
        self.assertEqual(capstone.CAPTIONFORGE_NODE_VERSION, version_module.CAPTIONFORGE_VERSION)

    def test_anchor_tooltips_describe_frozen_1_x_behavior(self) -> None:
        expected = (
            "Optional persistent caption/training anchor. In CaptionForge 1.x, a non-empty "
            "anchor is preserved in the final caption variants rather than treated as image "
            "evidence that the Validator may remove."
        )
        self.assertEqual(
            planner.JLC_CaptionForge_Pipeline_Planner.INPUT_TYPES()["required"]
            ["LoRA - user caption anchor"][1]["tooltip"],
            expected,
        )
        self.assertEqual(
            capstone.JLC_CaptionForge.INPUT_TYPES()["required"]
            ["LoRA - user caption anchor"][1]["tooltip"],
            expected,
        )

    def test_package_discovery_is_explicit_and_production_only(self) -> None:
        pyproject = (ROOT / "pyproject.toml").read_text(encoding="utf-8")
        self.assertNotIn("[tool.setuptools.packages.find]", pyproject)
        for package in (
            "CaptionForge",
            "CaptionForge.engines",
            "CaptionForge.nodes",
            "CaptionForge.nodes.caption_nodes",
        ):
            self.assertIn(f'"{package}"', pyproject)
        for excluded in (".backups", ".tests", ".bak", ".deprecated", "experimental"):
            self.assertNotIn(f'"{excluded}', pyproject)

    def test_active_python_modules_have_module_docstrings(self) -> None:
        files = [ROOT / "__init__.py", ROOT / "captionforge_version.py"]
        files.extend((ROOT / "engines").glob("*.py"))
        files.extend((ROOT / "nodes").glob("*.py"))
        files.extend((ROOT / "nodes" / "caption_nodes").glob("*.py"))
        files.extend((ROOT / "tests").glob("*.py"))
        for path in files:
            with self.subTest(path=path.relative_to(ROOT)):
                tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
                self.assertTrue(ast.get_docstring(tree))


class ActiveNodeHelpTests(unittest.TestCase):
    def test_every_active_input_has_plain_help_text(self) -> None:
        node_classes = (
            planner.JLC_CaptionForge_Pipeline_Planner,
            capstone.JLC_CaptionForge,
            template_options.JLC_CaptionForgeExtraOptions,
            joy.JLC_CaptionForgeJoy,
            qwen.JLC_CaptionForgeQwen,
            ollama.JLC_CaptionForgeOllamaCaption,
        )
        for node_class in node_classes:
            inputs = node_class.INPUT_TYPES()
            for section in ("required", "optional"):
                for name, spec in inputs.get(section, {}).items():
                    with self.subTest(node=node_class.__name__, input=name):
                        self.assertGreaterEqual(len(spec), 2)
                        metadata = spec[1]
                        self.assertIsInstance(metadata, dict)
                        self.assertTrue(str(metadata.get("tooltip", "")).strip())


if __name__ == "__main__":
    unittest.main()
