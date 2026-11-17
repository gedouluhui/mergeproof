import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest


class CLITests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)

    def call(self, *args):
        return subprocess.run([sys.executable, '-m', 'mergeproof', *map(str, args)],
                              text=True, capture_output=True, encoding='utf-8')

    def test_real_demo_generates_opposite_results(self):
        out = self.root / 'demo'
        p = self.call('demo', '--output', out)
        self.assertEqual(p.returncode, 0, p.stderr)
        self.assertEqual(json.loads((out / 'good.json').read_text())['status'], 'PASS')
        bad = json.loads((out / 'bad.json').read_text())
        self.assertEqual(bad['status'], 'FAIL')
        self.assertIn('merge_mismatch', [f['rule'] for f in bad['findings']])
        self.assertTrue((out / 'bad.html').is_file())
        self.assertNotEqual(self.call('demo', '--output', out).returncode, 0)

    def test_verify_exit_code_and_input_preservation(self):
        out = self.root / 'demo'
        p = self.call('demo', '--output', out)
        self.assertEqual(p.returncode, 0, p.stderr)
        args = ['verify', '--base', out / 'base.safetensors', '--adapter', out / 'adapter.safetensors',
                '--merged', out / 'bad.safetensors', '--manifest', out / 'manifest.json']
        p = self.call(*args)
        self.assertEqual(p.returncode, 1, p.stderr)
        self.assertEqual(json.loads(p.stdout)['status'], 'FAIL')
        target = out / 'base.safetensors'
        before = hashlib.sha256(target.read_bytes()).hexdigest()
        p = self.call(*args, '--json', target)
        self.assertEqual(p.returncode, 2)
        self.assertEqual(hashlib.sha256(target.read_bytes()).hexdigest(), before)

    def test_missing_input_structured_error(self):
        p = self.call('verify', '--base', 'absent', '--adapter', 'absent', '--merged', 'absent', '--manifest', 'absent')
        self.assertEqual(p.returncode, 2)
        self.assertEqual(json.loads(p.stdout)['status'], 'ERROR')

    def test_html_escapes_evidence(self):
        self.assertIsNotNone(importlib.util.find_spec('mergeproof.report'), 'report renderer must exist')
        from mergeproof.report import to_html
        text = '</script><img src=x onerror=alert(1)>'
        html = to_html({'status': 'FAIL', 'scope': text, 'inputs': {}, 'config': {}, 'targets': [],
                        'summary': {}, 'findings': [{'rule': text, 'tensor': text, 'detail': text}]})
        self.assertNotIn('<img', html)
        self.assertIn('&lt;img', html)
        self.assertNotIn('<script', html)

    def test_bad_argument_still_returns_json(self):
        p = self.call('verify', '--base', 'absent', '--adapter', 'absent', '--merged', 'absent',
                      '--manifest', 'absent', '--block-size', 'nope')
        self.assertEqual(p.returncode, 2)
        self.assertTrue(p.stdout.strip(), 'parse errors must have a structured response')
        self.assertEqual(json.loads(p.stdout)['status'], 'ERROR')

    def test_duplicate_json_keys_rejected(self):
        from mergeproof.manifest import read_json
        p = self.root / 'duplicate.json'
        p.write_text('{"a":1,"a":2}', encoding='utf-8')
        with self.assertRaises(ValueError):
            read_json(p)


if __name__ == '__main__':
    unittest.main()
