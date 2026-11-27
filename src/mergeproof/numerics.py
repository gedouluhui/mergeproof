"""Conservative enclosures and a bounded-use exact scalar oracle.

The operands originate from finite FP32 checkpoints. Their products are exact
in binary64 (at most 48 significand bits; no binary64 under/overflow). All later
operations used for certification are outward-rounded. Diagnostic approximations
must never substitute for the masks returned here.
"""
from fractions import Fraction
import math

import numpy as np

MATCH, UNKNOWN, MISMATCH = 1, 0, -1


def _down(value):
    return np.nextafter(value, -np.inf)


def _up(value):
    return np.nextafter(value, np.inf)


def fraction_interval(value):
    """Two binary64 endpoints enclosing the exact rational scalar."""
    try:
        rounded = float(value)
    except OverflowError:
        rounded = math.inf if value > 0 else -math.inf
    if math.isinf(rounded):
        return (np.finfo(np.float64).max, np.inf) if value > 0 else (-np.inf, -np.finfo(np.float64).max)
    exact_rounded = Fraction.from_float(rounded)
    low = float(_down(rounded)) if exact_rounded > value else rounded
    high = float(_up(rounded)) if exact_rounded < value else rounded
    return low, high


def dot_interval(left, right):
    left = np.asarray(left, dtype=np.float64)
    right = np.asarray(right, dtype=np.float64)
    if left.ndim != 2 or right.ndim != 2 or left.shape[1] != right.shape[0]:
        raise ValueError('interval dot requires compatible matrices')
    low = np.zeros((left.shape[0], right.shape[1]), dtype=np.float64)
    high = np.zeros_like(low)
    with np.errstate(over='ignore', invalid='ignore', under='ignore'):
        for k in range(left.shape[1]):
            product = left[:, k:k + 1] * right[k:k + 1, :]
            low = _down(low + product)
            high = _up(high + product)
    return low, high


def _absolute(low, high):
    lower = np.where((low <= 0) & (high >= 0), 0., np.minimum(np.abs(low), np.abs(high)))
    upper = np.maximum(np.abs(low), np.abs(high))
    return lower, upper


def _classification(value_low, value_high, threshold_low, threshold_high, valid):
    result = np.zeros(value_low.shape, dtype=np.int8)
    result[valid & (value_high <= threshold_low)] = MATCH
    result[valid & (value_low > threshold_high)] = MISMATCH
    return result


def comparison_bounds(base, merged, dot_bounds, alpha, rank, atol, rtol):
    """Classify only when intervals certify the exact requested inequality."""
    base = np.asarray(base, dtype=np.float64)
    merged = np.asarray(merged, dtype=np.float64)
    dot_low, dot_high = dot_bounds
    scale_low, scale_high = fraction_interval(Fraction(alpha) / rank)
    with np.errstate(over='ignore', invalid='ignore', under='ignore'):
        delta_low = _down(np.minimum(dot_low * scale_low, dot_low * scale_high))
        delta_high = _up(np.maximum(dot_high * scale_low, dot_high * scale_high))
        abs_low, abs_high = _absolute(delta_low, delta_high)
        observed_low = _down(merged - base)
        observed_high = _up(merged - base)
        error_low, error_high = _absolute(_down(observed_low - delta_high), _up(observed_high - delta_low))
        if rtol == 0:
            tolerance_low = np.full(base.shape, atol, dtype=np.float64)
            tolerance_high = tolerance_low.copy()
        else:
            tolerance_low = np.maximum(0., _down(atol + _down(rtol * abs_low)))
            tolerance_high = _up(atol + _up(rtol * abs_high))
        valid = np.ones(base.shape, dtype=bool)
        for value in (dot_low, dot_high, delta_low, delta_high, error_low, error_high, tolerance_low, tolerance_high):
            valid &= np.isfinite(value)
        valid &= (dot_low <= dot_high) & (delta_low <= delta_high)
        valid &= (error_low <= error_high) & (tolerance_low <= tolerance_high)
        if not (math.isfinite(scale_low) and math.isfinite(scale_high)):
            valid[:] = False
        return {
            'match': _classification(error_low, error_high, tolerance_low, tolerance_high, valid),
            'baseline': _classification(abs_low, abs_high, tolerance_low, tolerance_high, valid),
            'valid': valid,
            'error_lower': error_low,
            'error_upper': error_high,
            'tolerance_lower': tolerance_low,
            'tolerance_upper': tolerance_high,
        }


def exact_compare(base, merged, a_column, b_row, alpha, rank, atol, rtol):
    """Exact real equation on represented operands; callers enforce term budget."""
    dot = sum((Fraction.from_float(float(a)) * Fraction.from_float(float(b))
               for a, b in zip(a_column, b_row, strict=True)), Fraction())
    delta = dot * Fraction(alpha) / rank
    observed = Fraction.from_float(float(merged)) - Fraction.from_float(float(base))
    residual = abs(observed - delta)
    tolerance = Fraction.from_float(float(atol)) + Fraction.from_float(float(rtol)) * abs(delta)
    return {'match': residual <= tolerance, 'baseline_match': abs(delta) <= tolerance,
            'residual': residual, 'tolerance': tolerance, 'delta': delta}
