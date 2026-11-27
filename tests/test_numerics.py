"""Independent rational oracles for interval enclosure and certified decisions."""
from fractions import Fraction
import importlib.util
import inspect
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
from safetensors.numpy import save_file
from mergeproof import verify


def f(value):
    return Fraction.from_float(float(value))


class NumericsTests(unittest.TestCase):
    def module(self):
        self.assertIsNotNone(importlib.util.find_spec('mergeproof.numerics'), 'certification module must exist')
        from mergeproof import numerics
        return numerics

    def test_dot_intervals_enclose_independent_fraction(self):
        n = self.module()
        rng = np.random.default_rng(729)
        for _ in range(12):
            left = (rng.choice([-1., 1.], size=(2, 7)) * np.ldexp(rng.random((2, 7)), rng.integers(-140, 120, (2, 7)))).astype('float32')
            right = (rng.choice([-1., 1.], size=(7, 3)) * np.ldexp(rng.random((7, 3)), rng.integers(-140, 120, (7, 3)))).astype('float32')
            low, high = n.dot_interval(left, right)
            for i in range(2):
                for j in range(3):
                    exact = sum((f(left[i, k]) * f(right[k, j]) for k in range(7)), Fraction())
                    self.assertLessEqual(f(low[i, j]), exact)
                    self.assertGreaterEqual(f(high[i, j]), exact)

    def test_exact_boundary_and_alpha_integer(self):
        n = self.module()
        r = n.exact_compare(0., 2., np.array([1.], dtype='float32'), np.array([1.], dtype='float32'), 1, 1, 1., 0.)
        self.assertTrue(r['match'])
        self.assertTrue(r['baseline_match'])
        self.assertEqual(r['residual'], Fraction(1))
        r = n.exact_compare(1., float(2**60), np.array([1.], dtype='float32'), np.array([2**60], dtype='float32'), 1, 1, .5, 0.)
        self.assertFalse(r['match'])
        self.assertEqual(r['residual'], Fraction(1))

    def test_bounds_never_certify_wrong_decision(self):
        n = self.module()
        rng = np.random.default_rng(131)
        for _ in range(40):
            left = rng.normal(size=(1, 3)).astype('float32')
            right = rng.normal(size=(3, 2)).astype('float32')
            base = rng.normal(size=(1, 2)).astype('float32')
            merged = (base + left @ right).astype('float32')
            bounds = n.comparison_bounds(base, merged, n.dot_interval(left, right), 3, 3, 1e-6, 1e-5)
            for j in range(2):
                exact = n.exact_compare(base[0, j], merged[0, j], right[:, j], left[0], 3, 3, 1e-6, 1e-5)
                for key, truth in [('match', exact['match']), ('baseline', exact['baseline_match'])]:
                    if bounds[key][0, j] == 1:
                        self.assertTrue(truth)
                    elif bounds[key][0, j] == -1:
                        self.assertFalse(truth)


class CertifiedFileTests(unittest.TestCase):
    def check_case(self, values, base, merged, *, alpha=None, **kwargs):
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp)
            b = np.array([values], dtype='float32')
            a = np.ones((len(values), 1), dtype='float32')
            save_file({'w': np.array([[base]], dtype='float32')}, str(p / 'base'))
            save_file({'A': a, 'B': b}, str(p / 'adapter'))
            save_file({'w': np.array([[merged]], dtype='float32')}, str(p / 'merged'))
            spec = {'schema_version': 1, 'variant': 'standard_linear',
                    'targets': [{'base': 'w', 'a': 'A', 'b': 'B', 'alpha': alpha if alpha is not None else len(values)}]}
            (p / 'manifest').write_text(json.dumps(spec))
            return verify(p / 'base', p / 'adapter', p / 'merged', p / 'manifest', **kwargs)

    def test_double_cancellation_wrong_fails_right_passes(self):
        values = [2.**120, 2.**60, 1., -2.**60, -2.**120, 2.]
        self.assertEqual(sum(f(x) for x in values), 3)
        for tile in (1, 2, 128):
            self.assertEqual(self.check_case(values, 0, 2, block_size=tile)['status'], 'FAIL')
            self.assertEqual(self.check_case(values, 0, 3, block_size=tile)['status'], 'PASS')

    def test_observed_subtraction_is_also_certified(self):
        result = self.check_case([2.**60], 1, 2.**60, atol=.5, rtol=0)
        self.assertEqual(result['status'], 'FAIL')
        self.assertEqual(result['targets'][0]['max_abs_error'], 1)

    def test_budget_zero_is_unknown_not_false_pass(self):
        self.assertIn('max_exact_terms', inspect.signature(verify).parameters, 'public verifier needs exact fallback budget')
        result = self.check_case([2.**120, 2.**60, 1., -2.**60, -2.**120, 2.], 0, 2, max_exact_terms=0)
        self.assertEqual(result['status'], 'INDETERMINATE')
        self.assertEqual(result['summary']['exact_terms_used'], 0)
        self.assertEqual(result['summary']['unresolved_elements'], 1)

    def test_bad_exact_budget_rejected(self):
        self.assertIn('max_exact_terms', inspect.signature(verify).parameters)
        for value in (-1, True, 1.5):
            self.assertEqual(self.check_case([1.], 0, 1, max_exact_terms=value)['status'], 'ERROR')

    def test_exact_threshold_equality_matches(self):
        # The base is outside tolerance but the merged artifact is exactly on it.
        self.assertEqual(self.check_case([2.], 0, 3, atol=1, rtol=0)['status'], 'PASS')


if __name__ == '__main__':
    unittest.main()
