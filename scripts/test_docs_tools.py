#!/usr/bin/env python3
"""Standard-library-only checks; no hardware, network, model load or training."""
import ast
import importlib.util
import json
from pathlib import Path
import re
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('metrics', ROOT / 'scripts/metrics_from_log.py')
metrics = importlib.util.module_from_spec(spec)
spec.loader.exec_module(metrics)


class ToolsTest(unittest.TestCase):
    def test_homepage_can_open_without_jekyll(self):
        homepage = (ROOT / 'docs/index.html').read_text()
        self.assertNotIn('{{', homepage)
        self.assertNotIn('{%', homepage)
        references = re.findall(r'(?:href|src|poster)="([^"]+)"', homepage)
        local_assets = [
            reference for reference in references
            if not reference.startswith(('http://', 'https://', '#'))
            and not reference.endswith('.html')
        ]
        for reference in local_assets:
            with self.subTest(reference=reference):
                self.assertTrue((ROOT / 'docs' / reference).is_file())

    def test_syntax(self):
        for p in (ROOT / 'scripts').glob('*.py'):
            ast.parse(p.read_text(), filename=str(p))

    def test_exact_and_rounded_steps(self):
        rows = metrics.parse('step:10 loss:0.3 grdn:2 lr:1e-5\rstep:8K loss:nan grdn:inf lr:1e-5')
        self.assertEqual(rows[0]['step'], 10)
        self.assertFalse(rows[0]['step_is_approximate'])
        self.assertEqual(rows[1]['step'], 8000)
        self.assertTrue(rows[1]['step_is_approximate'])
        self.assertEqual(rows[1]['loss'], 'nan')

    def test_resume_does_not_invent_monotonic_steps(self):
        rows = metrics.parse('step:90 loss:.1\nstep:50 loss:.2')
        self.assertEqual([r['step'] for r in rows], [90, 50])

    def test_reject_empty_postprocessor_before_model_import(self):
        with tempfile.TemporaryDirectory(prefix='so101-doc-test-') as temp:
            root = Path(temp)
            for name in ('config.json', 'model.safetensors', 'train_config.json', 'policy_preprocessor.json'):
                (root / name).touch()
            (root / 'policy_postprocessor.json').write_text(json.dumps({'steps': [
                {'registry_name': 'unnormalizer_processor', 'config': {'features': {}}}]}))
            result = subprocess.run([sys.executable, str(ROOT / 'scripts/verify_release.py'), str(root)],
                                    capture_output=True, text=True)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn('Missing/wrong action features', result.stderr)


if __name__ == '__main__':
    unittest.main()
