"""The intentionally narrow, explicit merge contract."""
import json
import math
from pathlib import Path


class Unsupported(ValueError):
    pass


def unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError('duplicate JSON key: ' + key)
        result[key] = value
    return result


def read_json(path, limit=2 * 1024 * 1024):
    with Path(path).open('rb') as stream:
        raw = stream.read(limit + 1)
    if len(raw) > limit:
        raise ValueError('JSON size limit exceeded')
    return json.loads(raw, object_pairs_hook=unique_object)


def finite_number(value, name, *, positive=False):
    if type(value) not in (int, float) or not math.isfinite(value):
        raise ValueError(name + ' must be finite and numeric')
    if value < 0 or (positive and value == 0):
        raise ValueError(name + ' is outside its permitted range')
    return float(value)


def load_manifest(path):
    data = read_json(path)
    if not isinstance(data, dict) or set(data) != {'schema_version', 'variant', 'targets'}:
        raise ValueError('manifest fields must be schema_version, variant, targets')
    if type(data['schema_version']) is not int or data['schema_version'] != 1:
        raise ValueError('unsupported manifest schema')
    if data['variant'] != 'standard_linear':
        raise Unsupported('only standard_linear LoRA is supported')
    targets = data['targets']
    if not isinstance(targets, list) or not 1 <= len(targets) <= 10000:
        raise ValueError('manifest needs 1..10000 targets')
    bases, used = set(), set()
    for target in targets:
        if not isinstance(target, dict) or set(target) != {'base', 'a', 'b', 'alpha'}:
            raise ValueError('each target needs exactly base, a, b, alpha')
        for key in ('base', 'a', 'b'):
            if not isinstance(target[key], str) or not target[key] or len(target[key]) > 2048:
                raise ValueError('invalid tensor name')
        if target['base'] in bases or target['a'] in used or target['b'] in used or target['a'] == target['b']:
            raise ValueError('tensor mappings must be unique')
        bases.add(target['base'])
        used.update((target['a'], target['b']))
        finite_number(target['alpha'], 'alpha', positive=True)
    return data
