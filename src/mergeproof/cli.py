"""Small public CLI, with no remote services or credential requirements."""
import argparse
import json
from pathlib import Path
import sys

from .io import atomic_write, validate_outputs
from .report import to_html
from .verify import verify

EXIT_CODES = {'PASS': 0, 'FAIL': 1, 'ERROR': 2, 'INDETERMINATE': 3, 'UNSUPPORTED': 4}


class StructuredParser(argparse.ArgumentParser):
    def error(self, message):
        raise ValueError(message)


def demo(output):
    import numpy as np
    from safetensors.numpy import save_file
    out = Path(output)
    if out.exists():
        raise ValueError('demo output must be a new directory')
    out.mkdir(parents=False)
    base = np.array([[10, 20], [30, 40]], dtype='float32')
    a, b = np.array([[1, 2]], dtype='float32'), np.array([[3], [4]], dtype='float32')
    save_file({'layer.weight': base}, str(out / 'base.safetensors'))
    save_file({'A': a, 'B': b}, str(out / 'adapter.safetensors'))
    save_file({'layer.weight': base + b @ a}, str(out / 'good.safetensors'))
    save_file({'layer.weight': base + 2 * (b @ a)}, str(out / 'bad.safetensors'))
    spec = {'schema_version': 1, 'variant': 'standard_linear',
            'targets': [{'base': 'layer.weight', 'a': 'A', 'b': 'B', 'alpha': 1.0}]}
    atomic_write(out / 'manifest.json', json.dumps(spec, indent=2))
    statuses = {}
    for case in ('good', 'bad'):
        r = verify(out / 'base.safetensors', out / 'adapter.safetensors',
                   out / (case + '.safetensors'), out / 'manifest.json')
        atomic_write(out / (case + '.json'), json.dumps(r, indent=2, allow_nan=False))
        atomic_write(out / (case + '.html'), to_html(r))
        statuses[case] = r['status']
    return statuses


def main(argv=None):
    parser = StructuredParser(description='Independently verify declared FP32 Linear LoRA merge artifacts.')
    sub = parser.add_subparsers(dest='command', required=True)
    check = sub.add_parser('verify', help='verify three local safetensors files against an explicit manifest')
    for name in ('base', 'adapter', 'merged', 'manifest'):
        check.add_argument('--' + name, required=True)
    check.add_argument('--json')
    check.add_argument('--html')
    check.add_argument('--block-size', type=int, default=128)
    check.add_argument('--atol', type=float, default=1e-6)
    check.add_argument('--rtol', type=float, default=1e-5)
    check.add_argument('--max-working-bytes', type=int, default=67108864)
    sub.add_parser('demo', help='generate and verify a tiny good/broken example').add_argument('--output', required=True)
    try:
        args = parser.parse_args(argv)
        if args.command == 'demo':
            statuses = demo(args.output)
            print(json.dumps(statuses))
            return 0 if statuses == {'good': 'PASS', 'bad': 'FAIL'} else 2
        inputs = [args.base, args.adapter, args.merged, args.manifest]
        validate_outputs([args.json, args.html], inputs)
        report = verify(*inputs, block_size=args.block_size, atol=args.atol, rtol=args.rtol,
                        max_working_bytes=args.max_working_bytes)
        text = json.dumps(report, indent=2, ensure_ascii=True, allow_nan=False)
        if args.json:
            atomic_write(args.json, text + '\n')
        if args.html:
            atomic_write(args.html, to_html(report))
        print(text)
        return EXIT_CODES[report['status']]
    except (ValueError, OSError) as exc:
        detail = str(exc) if isinstance(exc, ValueError) else type(exc).__name__
        print(json.dumps({'status': 'ERROR', 'error': detail}))
        return 2


if __name__ == '__main__':
    sys.exit(main())
