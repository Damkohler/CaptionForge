"""CPU-only warning and installation-contract tests; no model downloads required."""
import ast
from contextlib import nullcontext
import importlib.util
from pathlib import Path
import types
import unittest
from unittest.mock import Mock
import warnings

try:
    import tomllib
except ModuleNotFoundError:
    import tomli as tomllib

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    'joy_warnings', ROOT / 'engines/captionforge_joy_warnings.py'
)
joy_warnings = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(joy_warnings)
scope = joy_warnings.joy_8bit_inference_warnings
MESSAGE = 'MatMul8bitLt: inputs will be cast from torch.bfloat16 to float16 during quantization'
MODULE = 'bitsandbytes.autograd._functions'


def emit(message=MESSAGE, category=UserWarning, module=MODULE):
    warnings.warn_explicit(message, category, filename='mock_bnb.py', lineno=1, module=module)


class WarningScopeTests(unittest.TestCase):
    def test_import_does_not_change_filters(self):
        before = list(warnings.filters)
        SPEC.loader.exec_module(joy_warnings)
        self.assertEqual(warnings.filters, before)

    def test_six_caption_bursts_and_useful_diagnostics(self):
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always')
            before = list(warnings.filters)
            for _ in range(6):
                with scope(True):
                    for _ in range(100):
                        emit()
                    emit('Unrelated bitsandbytes warning')
                self.assertEqual(warnings.filters, before)
            self.assertEqual(len(caught), 6)
            self.assertTrue(all(str(w.message) == 'Unrelated bitsandbytes warning' for w in caught))

    def test_near_matches_remain_visible(self):
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always')
            with scope(True):
                emit(MESSAGE.replace('bfloat16', 'float32'))
                emit(MESSAGE + ' extra context')
                emit('prefix ' + MESSAGE)
                emit(module='another_engine')
                emit(category=RuntimeWarning)
                emit(module=MODULE + '.other')
            self.assertEqual(len(caught), 6)

    def test_default_mode_and_after_scope_remain_visible(self):
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always')
            with scope(False):
                emit()
            with scope(True):
                emit()
            emit()
            self.assertEqual(len(caught), 2)

    def test_exception_restores_callers_error_policy(self):
        with warnings.catch_warnings():
            warnings.simplefilter('error')
            before = list(warnings.filters)
            with self.assertRaisesRegex(ValueError, 'generation failed'):
                with scope(True):
                    emit()
                    raise ValueError('generation failed')
            self.assertEqual(warnings.filters, before)
            with self.assertRaises(UserWarning):
                emit()
            with scope(True):
                with self.assertRaisesRegex(UserWarning, 'unrelated'):
                    emit('unrelated')


class GenerationScopeTests(unittest.TestCase):
    """Execute the real caption_pil method with processor/torch/model test doubles."""
    def run_generation(self, mode, fail=False):
        tree = ast.parse((ROOT / 'engines/jlc_joy_caption_engine.py').read_text(encoding='utf-8'))
        cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'JoyCaptionEngine')
        method = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == 'caption_pil')
        method.decorator_list = []
        namespace = {
            'joy_8bit_inference_warnings': scope,
            'torch': types.SimpleNamespace(
                bfloat16='bf16', dtype=type(None), autocast=Mock(return_value=nullcontext())
            ),
            'resize_for_model': lambda image, size: image,
            'cleanup_caption': lambda text, config: text,
        }
        # Postponed annotations avoid importing PIL/torch just to test control flow.
        module = ast.Module(body=[ast.ImportFrom(module='__future__', names=[ast.alias(name='annotations')], level=0), method], type_ignores=[])
        exec(compile(ast.fix_missing_locations(module), '<caption_pil>', 'exec'), namespace)

        class Inputs(dict):
            def to(self, device):
                return self

        processor = Mock(return_value=Inputs(input_ids=types.SimpleNamespace(shape=(1, 2))))
        processor.tokenizer.eos_token_id = 2
        processor.tokenizer.pad_token_id = 2
        processor.tokenizer.decode.return_value = 'caption'

        def generate(**kwargs):
            emit()
            emit('Useful generation diagnostic')
            if fail:
                raise ValueError('generation failed')
            return [[1, 2, 3]]

        engine = types.SimpleNamespace(
            generation=types.SimpleNamespace(seed=None, max_new_tokens=10, temperature=0, repetition_penalty=1),
            config=types.SimpleNamespace(system_prompt='system', prompt='prompt', max_size=1024, memory_mode=mode),
            cleanup=None, inference_device='cpu', processor=processor,
            model=types.SimpleNamespace(dtype=None, generate=generate),
            prepare_for_inference=Mock(), cleanup_after_inference=Mock(side_effect=lambda: emit('Cleanup diagnostic')),
            _autocast_device_type=lambda: 'cpu', _autocast_enabled=lambda device: False,
        )
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always')
            before = list(warnings.filters)
            if fail:
                with self.assertRaisesRegex(ValueError, 'generation failed'):
                    namespace['caption_pil'](engine, Mock())
            else:
                self.assertEqual(namespace['caption_pil'](engine, Mock()), ('caption', 'caption'))
            self.assertEqual(warnings.filters, before)
        engine.cleanup_after_inference.assert_called_once()
        self.assertEqual(namespace['torch'].autocast.call_args.kwargs['dtype'], 'bf16')
        return [str(w.message) for w in caught]

    def test_balanced_generation(self):
        self.assertEqual(self.run_generation('Balanced (8-bit)'), ['Useful generation diagnostic', 'Cleanup diagnostic'])

    def test_default_generation(self):
        self.assertEqual(self.run_generation('Default'), [MESSAGE, 'Useful generation diagnostic', 'Cleanup diagnostic'])

    def test_failed_generation(self):
        self.assertEqual(self.run_generation('Balanced (8-bit)', fail=True), ['Useful generation diagnostic', 'Cleanup diagnostic'])


class DependencyContractTests(unittest.TestCase):
    def test_manager_and_package_dependencies_match(self):
        project = tomllib.loads((ROOT / 'pyproject.toml').read_text(encoding='utf-8'))['project']
        requirements = {
            line.strip() for line in (ROOT / 'requirements.txt').read_text().splitlines()
            if line.strip() and not line.lstrip().startswith('#')
        }
        self.assertEqual(requirements, set(project['dependencies']))
        self.assertIn('accelerate', requirements)
        self.assertIn('bitsandbytes>=0.46.1', requirements)
        self.assertEqual(project['optional-dependencies']['quantization'], ['bitsandbytes>=0.46.1'])


if __name__ == '__main__':
    unittest.main()
