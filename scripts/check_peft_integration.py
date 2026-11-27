"""Required PEFT integration gate: never silently skips a missing dependency."""
import json
from pathlib import Path
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
sys.path.insert(0, str(ROOT / 'tests'))

import mergeproof  # noqa: E402
from peft_case import build_case, missing_dependencies  # noqa: E402


def main():
    missing = missing_dependencies()
    if missing:
        print(json.dumps({'status': 'UNSUPPORTED', 'missing': missing,
                          'detail': 'install the optional integration group: pip install .[peft]'}))
        return 3
    import torch
    import peft
    import safetensors
    results = []
    failures = []
    for rank in (1, 4, 8):
        with tempfile.TemporaryDirectory() as tmp:
            case = build_case(tmp, rank)
            if case['max_update'] <= 0.0:
                failures.append({'rank': rank, 'detail': 'fixture has no nonzero LoRA update'})
                continue
            statuses = {}
            for label, merged, manifest in (('good', case['merged'], case['manifest']),
                                            ('wrong_alpha', case['merged'], case['wrong_manifest']),
                                            ('unchanged_base', case['unchanged'], case['manifest'])):
                report = mergeproof.verify(case['base'], case['adapter'], merged, manifest, block_size=2)
                statuses[label] = report['status']
                expected = 'PASS' if label == 'good' else 'FAIL'
                if report['status'] != expected:
                    failures.append({'rank': rank, 'case': label, 'expected': expected,
                                     'observed': report['status'], 'findings': report['findings']})
                if label == 'good':
                    detail = report['targets'][0]
                    summary = {'unresolved_elements': report['summary']['unresolved_elements'],
                               'exact_elements': detail['elements_exact'],
                               'exact_terms_used': report['summary']['exact_terms_used'],
                               'max_abs_error': detail['max_abs_error'],
                               'witness': detail['witness_coordinate']}
            results.append({'rank': rank, 'statuses': statuses, 'max_update': case['max_update'],
                            'good_summary': summary})
    payload = {'status': 'PASS' if not failures else 'FAIL',
               'versions': {'mergeproof': mergeproof.__version__, 'torch': torch.__version__,
                            'peft': peft.__version__, 'safetensors': safetensors.__version__},
               'cases': results, 'failures': failures}
    print(json.dumps(payload, indent=2))
    return 0 if not failures else 1


if __name__ == '__main__':
    sys.exit(main())
