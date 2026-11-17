import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
from safetensors.numpy import save_file


class VerifyTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.base = {'layer.weight': np.array([[10, 20], [30, 40]], dtype='float32'),
                     'bias': np.array([1, 2], dtype='float32')}
        self.adapter = {'A': np.array([[1, 2]], dtype='float32'),
                        'B': np.array([[3], [4]], dtype='float32')}
        self.merged = {'layer.weight': np.array([[13, 26], [34, 48]], dtype='float32'),
                       'bias': self.base['bias'].copy()}
        self.manifest = {'schema_version': 1, 'variant': 'standard_linear',
                         'targets': [{'base': 'layer.weight', 'a': 'A', 'b': 'B', 'alpha': 1.0}]}

    def run_check(self, **kwargs):
        self.assertIsNotNone(importlib.util.find_spec('mergeproof'), 'mergeproof package must exist')
        import mergeproof
        self.assertTrue(callable(getattr(mergeproof, 'verify', None)), 'verify API must exist')
        for name, values in [('base', self.base), ('adapter', self.adapter), ('merged', self.merged)]:
            save_file(values, str(self.root / (name + '.safetensors')))
        m = self.root / 'manifest.json'
        m.write_text(json.dumps(self.manifest), encoding='utf-8')
        return mergeproof.verify(self.root / 'base.safetensors', self.root / 'adapter.safetensors',
                                 self.root / 'merged.safetensors', m, **kwargs)

    def rules(self, report):
        return {f['rule'] for f in report['findings']}

    def test_hand_computed_merge_and_non_target(self):
        r = self.run_check(block_size=1)
        self.assertEqual(r['status'], 'PASS')
        self.assertEqual(r['summary']['targets_checked'], 1)
        self.assertEqual(r['summary']['non_targets_checked'], 1)
        self.assertEqual(r['targets'][0]['max_abs_error'], 0)

    def test_wrong_alpha(self):
        self.manifest['targets'][0]['alpha'] = 2
        r = self.run_check()
        self.assertEqual(r['status'], 'FAIL')
        self.assertIn('merge_mismatch', self.rules(r))

    def test_missing_adapter_pair(self):
        del self.adapter['A']
        self.assertEqual(self.run_check()['status'], 'ERROR')

    def test_unused_adapter_rejected(self):
        self.adapter['unused'] = np.ones(1, dtype='float32')
        self.assertEqual(self.run_check()['status'], 'ERROR')

    def test_non_target_change(self):
        self.merged['bias'][0] = 9
        r = self.run_check()
        self.assertEqual(r['status'], 'FAIL')
        self.assertIn('non_target_changed', self.rules(r))

    def test_zero_update_is_not_proof_of_merge(self):
        self.adapter['B'][:] = 0
        self.merged['layer.weight'] = self.base['layer.weight'].copy()
        self.assertEqual(self.run_check()['status'], 'INDETERMINATE')

    def test_base_magnitude_does_not_hide_missing_update(self):
        self.base['layer.weight'][:] = 1e8
        self.merged['layer.weight'] = self.base['layer.weight'].copy()
        self.assertEqual(self.run_check()['status'], 'FAIL')

    def test_fp16_unsupported(self):
        self.adapter['A'] = self.adapter['A'].astype('float16')
        self.assertEqual(self.run_check()['status'], 'UNSUPPORTED')

    def test_nonfinite_each_input(self):
        for where, key in [('base', 'layer.weight'), ('adapter', 'A'), ('merged', 'layer.weight')]:
            with self.subTest(where=where):
                a = getattr(self, where)[key]
                old = a.flat[0]
                a.flat[0] = np.nan
                self.assertEqual(self.run_check()['status'], 'FAIL')
                a.flat[0] = old

    def test_bad_limits_and_tolerances(self):
        for kwargs in [{'block_size': True}, {'block_size': 0}, {'max_working_bytes': 1},
                       {'atol': float('nan')}, {'rtol': -1}, {'rtol': True}]:
            with self.subTest(kwargs=kwargs):
                self.assertEqual(self.run_check(**kwargs)['status'], 'ERROR')

    def test_tile_invariance(self):
        reports = [self.run_check(block_size=s) for s in (1, 2, 7)]
        self.assertEqual([r['status'] for r in reports], ['PASS'] * 3)
        self.assertEqual(reports[0]['targets'], reports[1]['targets'])
        self.assertEqual(reports[1]['targets'], reports[2]['targets'])

    def test_cancellation_rank_reduction_independent_of_tiles(self):
        # math.fsum is an independent scalar reference, not NumPy BLAS.
        import math
        values = [2.0 ** 54] + [1.0] * 7 + [-2.0 ** 54] + [1.0] * 7
        self.base['layer.weight'] = np.zeros((16, 16), dtype='float32')
        self.adapter['A'] = np.ones((16, 16), dtype='float32')
        self.adapter['B'] = np.tile(np.array(values, dtype='float32'), (16, 1))
        self.manifest['targets'][0]['alpha'] = 16
        reference = math.fsum(values)
        self.assertEqual(reference, 14.0)
        for value, expected in ((7.0, 'FAIL'), (reference, 'PASS')):
            self.merged['layer.weight'] = np.full((16, 16), value, dtype='float32')
            actual = [self.run_check(block_size=s)['status'] for s in (1, 2, 7, 128)]
            self.assertEqual(actual, [expected] * 4)

    def test_multitile_compensation_temporaries_are_budgeted(self):
        self.base['layer.weight'] = np.zeros((1024, 512), dtype='float32')
        self.adapter['A'] = np.ones((1, 512), dtype='float32')
        self.adapter['B'] = np.ones((1024, 1), dtype='float32')
        self.merged['layer.weight'] = np.ones((1024, 512), dtype='float32')
        self.assertEqual(self.run_check(block_size=512, max_working_bytes=25190400)['status'], 'ERROR')
        self.assertEqual(self.run_check(block_size=128, max_working_bytes=25190400)['status'], 'PASS')

    def test_budget_refuses_less_than_minimum_streaming_buffers(self):
        self.base['bias'] = np.ones(300000, dtype='float32')
        self.merged['bias'] = self.base['bias'].copy()
        self.assertEqual(self.run_check(block_size=1, max_working_bytes=144)['status'], 'ERROR')

    def test_empty_or_duplicate_target(self):
        target = self.manifest['targets'][0]
        for targets in ([], [target, target]):
            self.manifest['targets'] = targets
            self.assertEqual(self.run_check()['status'], 'ERROR')

    def test_manifest_unknown_fields_and_variant(self):
        self.manifest['surprise'] = 1
        self.assertEqual(self.run_check()['status'], 'ERROR')
        del self.manifest['surprise']
        self.manifest['variant'] = 'dora'
        self.assertEqual(self.run_check()['status'], 'UNSUPPORTED')

    def test_swapped_same_shape_targets(self):
        self.base['other.weight'] = self.base['layer.weight'].copy()
        self.adapter['A2'] = self.adapter['A'].copy()
        self.adapter['B2'] = 2 * self.adapter['B']
        self.manifest['targets'].append({'base': 'other.weight', 'a': 'A2', 'b': 'B2', 'alpha': 1})
        self.merged['other.weight'] = self.merged['layer.weight'].copy()
        self.merged['layer.weight'] = self.base['layer.weight'] + 2 * (self.adapter['B'] @ self.adapter['A'])
        r = self.run_check()
        self.assertEqual(r['status'], 'FAIL')
        self.assertEqual(len([f for f in r['findings'] if f['rule'] == 'merge_mismatch']), 2)

    def test_missing_and_extra_base_keys(self):
        del self.merged['bias']
        self.assertEqual(self.run_check()['status'], 'ERROR')

    def test_oversized_rank_shape_and_boolean_alpha(self):
        self.manifest['targets'][0]['alpha'] = True
        self.assertEqual(self.run_check()['status'], 'ERROR')
        self.manifest['targets'][0]['alpha'] = 1
        self.adapter['B'] = np.ones((3, 1), dtype='float32')
        self.assertEqual(self.run_check()['status'], 'ERROR')


if __name__ == '__main__':
    unittest.main()
