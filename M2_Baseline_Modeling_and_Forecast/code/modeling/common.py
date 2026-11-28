import hashlib
import json
from pathlib import Path

import numpy as np


def require(condition, message):
    if not condition:
        raise ValueError(message)


def digest(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, sort_keys=True,
                               indent=2, allow_nan=False) + '\n', encoding='utf-8')


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def qcol(tau):
    return f'q_{round(tau * 100):03d}'


def repair_quantiles(raw):
    raw = np.asarray(raw, dtype='float64')
    require(raw.ndim == 2 and np.isfinite(raw).all(), 'Nonfinite quantile prediction')
    fixed = np.sort(np.maximum(raw, 0), axis=1)
    stats = {'negative_values': int((raw < 0).sum()),
             'crossing_rows': int((np.diff(raw, axis=1) < 0).any(axis=1).sum()),
             'changed_values': int((raw != fixed).sum())}
    return fixed, stats
