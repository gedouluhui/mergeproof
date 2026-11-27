"""Optional end-to-end check against a real PEFT merge; explicit runner covers CI."""
import tempfile
import unittest

import mergeproof

from peft_case import build_case, missing_dependencies

MISSING = missing_dependencies()


@unittest.skipIf(MISSING, 'optional integration dependencies missing: ' + ', '.join(MISSING))
class PeftIntegrationTests(unittest.TestCase):
    def check_rank(self, rank, block_size):
        with tempfile.TemporaryDirectory() as tmp:
            case = build_case(tmp, rank)
            self.assertGreater(case['max_update'], 0.0, 'the fixture must contain a nonzero LoRA update')
            arguments = dict(block_size=block_size)
            good = mergeproof.verify(case['base'], case['adapter'], case['merged'], case['manifest'], **arguments)
            wrong = mergeproof.verify(case['base'], case['adapter'], case['merged'], case['wrong_manifest'], **arguments)
            unchanged = mergeproof.verify(case['base'], case['adapter'], case['unchanged'], case['manifest'], **arguments)
            return good, wrong, unchanged

    def test_ranks_one_four_eight(self):
        for rank in (1, 4, 8):
            for block_size in (2, 128):
                with self.subTest(rank=rank, block_size=block_size):
                    good, wrong, unchanged = self.check_rank(rank, block_size)
                    self.assertEqual(good['status'], 'PASS', good['findings'])
                    self.assertEqual(wrong['status'], 'FAIL', wrong['findings'])
                    self.assertEqual(unchanged['status'], 'FAIL', unchanged['findings'])
                    self.assertEqual(good['summary']['targets_checked'], 1)
                    self.assertEqual(good['summary']['unresolved_elements'], 0)
                    self.assertEqual(good['summary']['non_targets_checked'], 3)
                    self.assertIn('merge_mismatch', {f['rule'] for f in wrong['findings']})
                    self.assertIn('merge_mismatch', {f['rule'] for f in unchanged['findings']})


if __name__ == '__main__':
    unittest.main()
