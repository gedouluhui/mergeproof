"""Tiled standard Linear LoRA artifact verification, not model evaluation.

Outward-rounded interval enclosures certify the requested inequality. Elements
whose intervals straddle a decision boundary are re-evaluated by a bounded
exact rational oracle, so neither cancellation nor a rounded subtraction can
produce a false PASS. The compensated float reference is reported only as a
diagnostic and never decides a status.
"""
from contextlib import ExitStack
from pathlib import Path

import numpy as np
from safetensors import safe_open

from .io import compare_raw, digest, header
from .manifest import Unsupported, finite_number, load_manifest
from .numerics import (MATCH, MISMATCH, UNKNOWN, comparison_bounds, dot_interval,
                       exact_compare)

INDETERMINATE_RULES = frozenset({'unobservable_update', 'unresolved_elements'})


def rank_ordered_product(left, right):
    """Neumaier-compensated rank-order diagnostic, independent of output tiles."""
    total = np.zeros((left.shape[0], right.shape[1]), dtype=np.float64)
    correction = np.zeros_like(total)
    for k in range(left.shape[1]):
        product = left[:, k:k + 1] * right[k:k + 1, :]
        updated = total + product
        correction += np.where(np.abs(total) >= np.abs(product),
                               (total - updated) + product, (product - updated) + total)
        total = updated
    return total + correction


def _finite_float(value):
    """Exact evidence as a JSON-safe float; None when it is not representable."""
    try:
        result = float(value)
    except OverflowError:
        return None
    return result if np.isfinite(result) else None


def _consider(worst, witness, value, coordinate):
    """Deterministic maximum with lowest-coordinate tie-breaking."""
    if value is None:
        return worst, witness
    if (worst is None or value > worst
            or (value == worst and value > 0 and (witness is None or list(coordinate) < witness))):
        return value, list(coordinate)
    return worst, witness


def verify(base, adapter, merged, manifest, *, block_size=128, atol=1e-6, rtol=1e-5,
           max_exact_terms=100000, max_working_bytes=67108864):
    report = {'schema_version': 1, 'tool': 'mergeproof', 'version': '0.2.0',
              'status': 'ERROR', 'inputs': {}, 'config': {}, 'targets': [], 'findings': [],
              'summary': {'targets_checked': 0, 'non_targets_checked': 0, 'exact_terms_used': 0,
                          'certified_elements': 0, 'exact_elements': 0, 'unresolved_elements': 0,
                          'mismatched_elements': 0, 'nonfinite_elements': 0},
              'scope': 'Explicit single-file FP32 standard Linear LoRA. Not model quality, provenance or proof of execution.'}

    def finding(rule, tensor, detail):
        report['findings'].append({'rule': rule, 'tensor': tensor, 'detail': detail})

    try:
        for name, value in [('block_size', block_size), ('max_working_bytes', max_working_bytes)]:
            if type(value) is not int or value <= 0:
                raise ValueError(name + ' must be a positive integer')
        if max_working_bytes < 4096:
            raise ValueError('max_working_bytes must be at least 4096 for streaming buffers')
        if type(max_exact_terms) is not int or max_exact_terms < 0:
            raise ValueError('max_exact_terms must be a non-negative integer')
        io_chunk = min(1024 * 1024, (max_working_bytes // 8 // 4) * 4)
        atol = finite_number(atol, 'atol')
        rtol = finite_number(rtol, 'rtol')
        report['config'] = dict(block_size=block_size, atol=atol, rtol=rtol,
                                max_working_bytes=max_working_bytes,
                                max_exact_terms=max_exact_terms,
                                comparison='certified enclosure of abs((merged-base)-delta) <= atol+rtol*abs(delta)',
                                certification='outward-rounded binary64 intervals; exact rational fallback in row-major order under one shared term budget',
                                diagnostic='rank-ordered Neumaier float64 residual for interval-certified elements; never decides a status',
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
        exact_terms_used = 0
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
                    raise Unsupported('v0.2 accepts only FP32 tensor files')
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
                # Tile panels plus the bounded exact-fallback buffers.
                estimate = 32 * rank * (nr + nc) + 256 * nr * nc + 32 * max_exact_terms
                if estimate > max_working_bytes:
                    raise ValueError('tile allocation estimate exceeds memory budget; decrease block_size or max_exact_terms')
                scale = float(target['alpha']) / rank
                if not np.isfinite(scale):
                    raise ValueError('nonfinite LoRA scale')
                affordable = max(0, (max_exact_terms - exact_terms_used) // rank)
                worst, witness, excesses = None, None, None
                finite, exclusion = True, False
                certified, exact_count, mismatches, unresolved, nonfinite = 0, 0, 0, 0, 0
                pending = []
                for i in range(0, rows, block_size):
                    for j in range(0, cols, block_size):
                        ii, jj = slice(i, min(i + block_size, rows)), slice(j, min(j + block_size, cols))
                        bt = b.get_slice(key)[ii, jj].astype(np.float64)
                        mt = m.get_slice(key)[ii, jj].astype(np.float64)
                        at = a.get_slice(target['a'])[:, jj].astype(np.float64)
                        lt = a.get_slice(target['b'])[ii, :].astype(np.float64)
                        if not all(np.isfinite(v).all() for v in (bt, mt, at, lt)):
                            finite = False
                            nonfinite += bt.size
                            del bt, mt, at, lt
                            continue
                        low, high = dot_interval(lt, at)
                        bounds = comparison_bounds(bt, mt, (low, high), target['alpha'],
                                                   rank, atol, rtol)
                        matched = bounds['match'] == MATCH
                        mismatched = bounds['match'] == MISMATCH
                        undecided = bounds['match'] == UNKNOWN
                        certified += int(matched.sum()) + int(mismatched.sum())
                        mismatches += int(mismatched.sum())
                        if np.any(bounds['baseline'] == MISMATCH):
                            exclusion = True
                        positions = np.argwhere(undecided)
                        room = affordable - len(pending)
                        taken = positions[:room] if room > 0 else positions[:0]
                        for row, col in taken:
                            pending.append((i + int(row), j + int(col),
                                            at[:, int(col)].copy(), lt[int(row), :].copy(),
                                            float(bt[row, col]), float(mt[row, col])))
                        unresolved += len(positions) - len(taken)
                        if np.any(matched | mismatched):
                            with np.errstate(over='ignore', invalid='ignore', under='ignore'):
                                compensated = rank_ordered_product(lt, at) * scale
                                diagnostic = np.abs((mt - bt) - compensated)
                                evidence = np.where(matched, np.where(np.isfinite(diagnostic), diagnostic,
                                                                     bounds['error_upper']),
                                                    np.where(mismatched, bounds['error_lower'], np.nan))
                                excess = evidence - bounds['tolerance_lower']
                            usable = np.isfinite(evidence)
                            if usable.any():
                                filled = np.where(usable, evidence, -np.inf)
                                position = np.unravel_index(int(np.argmax(filled)), filled.shape)
                                worst, witness = _consider(worst, witness, float(filled.max()),
                                                           (i + int(position[0]), j + int(position[1])))
                            usable = np.isfinite(excess)
                            if usable.any():
                                highest = float(np.where(usable, excess, -np.inf).max())
                                if excesses is None or highest > excesses:
                                    excesses = highest
                            del compensated, diagnostic, evidence, excess
                        del bt, mt, at, lt, low, high, bounds
                exact_target = 0
                for i, j, a_column, b_row, base_value, merged_value in pending:
                    outcome = exact_compare(base_value, merged_value, a_column, b_row,
                                            target['alpha'], rank, atol, rtol)
                    exact_terms_used += rank
                    exact_target += rank
                    exact_count += 1
                    certified += 1
                    if not outcome['match']:
                        mismatches += 1
                    if not outcome['baseline_match']:
                        exclusion = True
                    residual = _finite_float(outcome['residual'])
                    tolerance = _finite_float(outcome['tolerance'])
                    worst, witness = _consider(worst, witness, residual, (i, j))
                    if residual is not None and tolerance is not None:
                        if excesses is None or residual - tolerance > excesses:
                            excesses = residual - tolerance
                status = 'PASS'
                if not finite:
                    status = 'FAIL'
                    finding('nonfinite_tensor', key, 'A numerical input contains NaN or infinity.')
                elif mismatches:
                    status = 'FAIL'
                    finding('merge_mismatch', key, 'Observed update exceeds the declared delta-relative tolerance.')
                elif unresolved:
                    status = 'INDETERMINATE'
                    finding('unresolved_elements', key,
                            'Interval bounds could not decide the comparison and the exact term budget was exhausted.')
                elif not exclusion:
                    status = 'INDETERMINATE'
                    finding('unobservable_update', key,
                            'No element certifies that the unchanged base lies outside the tolerance; merge execution is not distinguishable.')
                report['targets'].append({'tensor': key, 'status': status, 'shape': shape, 'rank': rank,
                                          'alpha': _finite_float(target['alpha']),
                                          'elements_checked': rows * cols,
                                          'elements_exact': exact_count,
                                          'elements_certified': certified,
                                          'elements_mismatch': mismatches,
                                          'elements_unresolved': unresolved,
                                          'elements_nonfinite': nonfinite,
                                          'exact_terms_used': exact_target,
                                          'baseline_exclusion': exclusion,
                                          'max_abs_error': worst if finite else None,
                                          'max_tolerance_excess': excesses if finite else None,
                                          'witness_coordinate': witness if finite else None})
                report['summary']['targets_checked'] += 1
                report['summary']['exact_terms_used'] += exact_target
                report['summary']['exact_elements'] += exact_count
                report['summary']['certified_elements'] += certified
                report['summary']['mismatched_elements'] += mismatches
                report['summary']['unresolved_elements'] += unresolved
                report['summary']['nonfinite_elements'] += nonfinite
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
        report['status'] = ('FAIL' if rules - INDETERMINATE_RULES else
                            'INDETERMINATE' if rules else 'PASS')
    except Unsupported as exc:
        report['status'] = 'UNSUPPORTED'
        finding('unsupported', '', str(exc))
    except Exception as exc:
        report['status'] = 'ERROR'
        # Do not leak arbitrary OS messages containing user paths.
        detail = str(exc) if type(exc) in (ValueError, FloatingPointError) else type(exc).__name__
        finding('input_or_execution_error', '', detail)
    return report
