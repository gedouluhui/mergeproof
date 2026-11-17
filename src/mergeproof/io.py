"""Bounded file inspection; never deserialize Python objects."""
import hashlib
import json
import os
from pathlib import Path
import struct
import tempfile

import numpy as np

from .manifest import unique_object

CHUNK = 1024 * 1024


def digest(path, chunk_bytes=CHUNK):
    value = hashlib.sha256()
    with Path(path).open('rb', buffering=0) as stream:
        for block in iter(lambda: stream.read(chunk_bytes), b''):
            value.update(block)
    return value.hexdigest()


def header(path):
    with Path(path).open('rb') as stream:
        prefix = stream.read(8)
        if len(prefix) != 8:
            raise ValueError('truncated safetensors prefix')
        length = struct.unpack('<Q', prefix)[0]
        if length > 16 * CHUNK or length < 2:
            raise ValueError('safetensors header outside limit')
        raw = stream.read(length)
        if len(raw) != length:
            raise ValueError('truncated safetensors header')
        result = json.loads(raw, object_pairs_hook=unique_object)
    if not isinstance(result, dict) or len(result) > 100001:
        raise ValueError('invalid tensor inventory')
    result.pop('__metadata__', None)
    return result, length + 8


def compare_raw(left, right, left_info, right_info, left_start, right_start, chunk_bytes=CHUNK):
    """Compare bounded float32 tensor chunks; safe_open validates offsets first."""
    count = left_info['data_offsets'][1] - left_info['data_offsets'][0]
    equal, finite = True, True
    with Path(left).open('rb', buffering=0) as a, Path(right).open('rb', buffering=0) as b:
        a.seek(left_start + left_info['data_offsets'][0])
        b.seek(right_start + right_info['data_offsets'][0])
        while count:
            n = min(chunk_bytes, count)
            x, y = a.read(n), b.read(n)
            if len(x) != n or len(y) != n:
                raise ValueError('tensor data changed or truncated during inspection')
            equal = equal and x == y
            finite = finite and bool(np.isfinite(np.frombuffer(x, dtype='<f4')).all())
            finite = finite and bool(np.isfinite(np.frombuffer(y, dtype='<f4')).all())
            count -= n
    return equal, finite


def validate_outputs(outputs, inputs=()):
    seen = []
    protected = [Path(p).resolve() for p in inputs]
    for item in outputs:
        if item is None:
            continue
        path = Path(item)
        resolved = path.resolve()
        if resolved in protected or resolved in seen:
            raise ValueError('report destination aliases an input or another report')
        if path.is_symlink():
            raise ValueError('symlink report destinations are not supported')
        for other in list(inputs) + seen:
            if path.exists() and Path(other).exists() and os.path.samefile(path, other):
                raise ValueError('report destination aliases a protected file')
        if not path.parent.is_dir() or (path.exists() and not path.is_file()):
            raise ValueError('report destination must have an existing parent directory')
        seen.append(resolved)


def atomic_write(path, text):
    path = Path(path)
    fd, temp = tempfile.mkstemp(prefix='.mergeproof-', dir=path.parent)
    try:
        with os.fdopen(fd, 'w', encoding='utf-8', newline='\n') as stream:
            stream.write(text)
        os.replace(temp, path)
    finally:
        if os.path.exists(temp):
            os.unlink(temp)
