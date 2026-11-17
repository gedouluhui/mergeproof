"""Tiled standard Linear LoRA artifact verification, not model evaluation."""
from contextlib import ExitStack
from pathlib import Path

import numpy as np
from safetensors import safe_open

from .io import compare_raw, digest, header
from .manifest import Unsupported, finite_number, load_manifest


def rank_ordered_product(left, right):
    """Neumaier-compensated rank-order reduction, independent of output tiles."""
    total = np.zeros((left.shape[0], right.shape[1]), dtype=np.float64)
    correction = np.zeros_like(total)
    for k in range(left.shape[1]):
        product = left[:, k:k + 1] * right[k:k + 1, :]
        updated = total + product
        correction += np.where(np.abs(total) >= np.abs(product),
                               (total - updated) + product, (product - updated) + total)
        total = updated
    return total + correction


def verify(base, adapter, merged, manifest, *, block_size=128, atol=1e-6,
           rtol=1e-5, max_working_bytes=67108864):
    report = {'schema_version': 1, 'tool': 'mergeproof', 'version': '0.1.0',
              'status': 'ERROR', 'inputs': {}, 'config': {}, 'targets': [], 'findings': [],
              'summary': {'targets_checked': 0, 'non_targets_checked': 0},
              'scope': 'Explicit single-file FP32 standard Linear LoRA. Not model quality, provenance or proof of execution.'}

    def finding(rule, tensor, detail):
        report['findings'].append({'rule': rule, 'tensor': tensor, 'detail': detail})

    try:
        for name, value in [('block_size', block_size), ('max_working_bytes', max_working_bytes)]:
            if type(value) is not int or value <= 0:
                raise ValueError(name + ' must be a positive integer')
        if max_working_bytes < 4096:
            raise ValueError('max_working_bytes must be at least 4096 for streaming buffers')
        io_chunk = min(1024 * 1024, (max_working_bytes // 8 // 4) * 4)
        atol = finite_number(atol, 'atol')
        rtol = finite_number(rtol, 'rtol')
        report['config'] = dict(block_size=block_size, atol=atol, rtol=rtol,
                                max_working_bytes=max_working_bytes,
                                comparison='abs((merged-base)-delta) <= atol+rtol*abs(delta)',
                                io_chunk_bytes=io_chunk,
                                reduction='rank-ordered Neumaier compensated float64',
                                memory_scope='estimated tile arrays and bounded streaming buffers; excludes metadata, interpreter/library overhead and OS page cache')
        spec = load_manifest(manifest)
        paths = dict(base=base, adapter=adapter, merged=merged, manifest=manifest)
        for label, path in paths.items():
            report['inputs'][label] = {'name': Path(path).name, 'sha256': digest(path, io_chunk)}
        bh, bs = header(base)
        ah, _ = header(adapter)
        mh, ms = header(merged)
        with ExitStack() as stack:
            b = stack.enter_context(safe_open(str(base), framework='np'))
            a = stack.enter_context(safe_open(str(adapter), framework='np'))
            m = stack.enter_context(safe_open(str(merged), framework='np'))
            if set(b.keys()) != set(m.keys()):
                raise ValueError('base and merged tensor inventories differ')
            if set(bh) != set(b.keys()) or set(ah) != set(a.keys()) or set(mh) != set(m.keys()):
                raise ValueError('metadata inventory disagreement')
            for inventory in (bh, ah, mh):
                if any(info['dtype'] != 'F32' for info in inventory.values()):
                    raise Unsupported('v0.1 accepts only FP32 tensor files')
            target_keys = {t['base'] for t in spec['targets']}
            adapter_keys = {t[k] for t in spec['targets'] for k in ('a', 'b')}
            if not target_keys <= set(bh) or adapter_keys != set(ah):
                raise ValueError('missing targets or missing/unused adapter tensors')
            for key in bh:
                if bh[key]['shape'] != mh[key]['shape'] or bh[key]['dtype'] != mh[key]['dtype']:
                    raise ValueError('base/merged shape or dtype mismatch: ' + key)
            for target in spec['targets']:
                key = target['base']
                shape = bh[key]['shape']
                ashape, bshape = ah[target['a']]['shape'], ah[target['b']]['shape']
                if (len(shape) != 2 or len(ashape) != 2 or len(bshape) != 2
                        or any(d <= 0 for d in shape + ashape + bshape)
                        or ashape[1] != shape[1] or bshape[0] != shape[0] or bshape[1] != ashape[0]):
                    raise ValueError('invalid Linear LoRA dimensions: ' + key)
                rows, cols = shape
                rank = ashape[0]
                nr, nc = min(rows, block_size), min(cols, block_size)
                # Reserve four float64 panel copies and 32 cell-sized arrays,
                # covering compensated reduction, comparisons and transient operands.
                estimate = 32 * rank * (nr + nc) + 256 * nr * nc
                if estimate > max_working_bytes:
                    raise ValueError('tile allocation estimate exceeds memory budget; decrease block_size')
                scale = float(target['alpha']) / rank
                if not np.isfinite(scale):
                    raise ValueError('nonfinite LoRA scale')
                worst, excess, witness = 0.0, 0.0, None
                baseline_passes, finite, matches = True, True, True
                for i in range(0, rows, block_size):
                    for j in range(0, cols, block_size):
                        ii, jj = slice(i, min(i + block_size, rows)), slice(j, min(j + block_size, cols))
                        bt = b.get_slice(key)[ii, jj].astype(np.float64)
                        mt = m.get_slice(key)[ii, jj].astype(np.float64)
                        at = a.get_slice(target['a'])[:, jj].astype(np.float64)
                        lt = a.get_slice(target['b'])[ii, :].astype(np.float64)
                        if not all(np.isfinite(v).all() for v in (bt, mt, at, lt)):
                            finite = False
                            del bt, mt, at, lt
                            continue
                        with np.errstate(over='raise', invalid='raise'):
                            delta = rank_ordered_product(lt, at) * scale
                            tolerance = atol + rtol * np.abs(delta)
                            error = np.abs((mt - bt) - delta)
                        baseline_passes = baseline_passes and bool((np.abs(delta) <= tolerance).all())
                        matches = matches and bool((error <= tolerance).all())
                        local = float(error.max())
                        pos = np.unravel_index(int(error.argmax()), error.shape)
                        coordinate = [i + int(pos[0]), j + int(pos[1])]
                        if local > worst or (local == worst and local > 0 and (witness is None or coordinate < witness)):
                            worst, witness = local, coordinate
                        excess = max(excess, float(np.max(error - tolerance)))
                        # No previous tile's arrays survive into the next reduction.
                        del bt, mt, at, lt, delta, tolerance, error
                status = 'PASS'
                if not finite:
                    status = 'FAIL'
                    finding('nonfinite_tensor', key, 'A numerical input contains NaN or infinity.')
                elif not matches:
                    status = 'FAIL'
                    finding('merge_mismatch', key, 'Observed update exceeds the declared delta-relative tolerance.')
                elif baseline_passes:
                    status = 'INDETERMINATE'
                    finding('unobservable_update', key, 'The unchanged base also satisfies this tolerance; merge execution is not distinguishable.')
                report['targets'].append({'tensor': key, 'status': status, 'shape': shape, 'rank': rank,
                                          'alpha': float(target['alpha']), 'elements_checked': rows * cols,
                                          'max_abs_error': worst if finite else None,
                                          'max_tolerance_excess': excess if finite else None,
                                          'witness_coordinate': witness if finite else None})
                report['summary']['targets_checked'] += 1
            for key in sorted(set(bh) - target_keys):
                same, finite = compare_raw(base, merged, bh[key], mh[key], bs, ms, io_chunk)
                report['summary']['non_targets_checked'] += 1
                if not same:
                    finding('non_target_changed', key, 'Non-target tensor bytes are not identical.')
                if not finite:
                    finding('nonfinite_tensor', key, 'A non-target tensor contains NaN or infinity.')
        for label, path in paths.items():
            if digest(path, io_chunk) != report['inputs'][label]['sha256']:
                raise ValueError('input changed during verification')
        rules = {f['rule'] for f in report['findings']}
        report['status'] = ('FAIL' if rules - {'unobservable_update'} else
                            'INDETERMINATE' if 'unobservable_update' in rules else 'PASS')
    except Unsupported as exc:
        report['status'] = 'UNSUPPORTED'
        finding('unsupported', '', str(exc))
    except Exception as exc:
        report['status'] = 'ERROR'
        # Do not leak arbitrary OS messages containing user paths.
        detail = str(exc) if type(exc) in (ValueError, FloatingPointError) else type(exc).__name__
        finding('input_or_execution_error', '', detail)
    return report
