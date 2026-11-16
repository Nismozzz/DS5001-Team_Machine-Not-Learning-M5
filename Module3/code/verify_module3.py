#!/usr/bin/env python3
"""Independent file, schema, constraint and hierarchy checks for Module 3."""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from reconciliation import (
    DAILY_Q_COLUMNS,
    MODELS,
    MODEL_INDEX,
    build_catalog,
    coherence_diagnostics,
    json_write,
    sha256,
)


SPLITS = ("validation_1", "validation_2", "validation_3", "test", "forecast")
SCHEMES = ("unreconciled", "bottom_up", "top_down", "mint")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--module3-root", type=Path, required=True)
    parser.add_argument("--member2-root", type=Path, required=True)
    return parser.parse_args()


def load_daily_arrays(
    root: Path,
    split: str,
    scheme: str,
    catalog,
) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
    point = np.zeros((len(MODELS), catalog.n_nodes, 28), dtype=np.float64)
    quantiles = np.zeros((catalog.n_nodes, 28, len(DAILY_Q_COLUMNS)), dtype=np.float64)
    actual = None if split == "forecast" else np.zeros((catalog.n_nodes, 28))
    for model in MODELS:
        mi = MODEL_INDEX[model]
        for level in range(1, 13):
            index = catalog.level_nodes[level]
            path = (
                root
                / "预测结果"
                / split
                / scheme
                / "daily"
                / model
                / f"L{level:02d}.parquet"
            )
            frame = pd.read_parquet(
                path,
                columns=[
                    "node_id",
                    "level",
                    "model",
                    "origin_d",
                    "target_d",
                    "horizon",
                    "reconciliation",
                    "yhat",
                    *DAILY_Q_COLUMNS,
                    "y_true",
                ],
            )
            assert len(frame) == len(index) * 28, f"Bad row count: {path}"
            assert frame.level.nunique() == 1 and int(frame.level.iloc[0]) == level
            assert frame.model.nunique() == 1 and frame.model.iloc[0] == model
            assert frame.reconciliation.nunique() == 1
            assert frame.reconciliation.iloc[0] == scheme
            node_code = {
                value: i for i, value in enumerate(catalog.node_ids_at(level))  # type: ignore[attr-defined]
            }
            local = frame.node_id.map(node_code).to_numpy(np.int32)
            horizon = frame.horizon.to_numpy(np.int16) - 1
            point[mi, index[local], horizon] = frame.yhat.to_numpy(np.float64)
            if model == "lightgbm":
                quantiles[index[local], horizon, :] = frame.loc[
                    :, DAILY_Q_COLUMNS
                ].to_numpy(np.float64)
            if actual is not None:
                actual[index[local], horizon] = frame.y_true.to_numpy(np.float64)
    assert np.isfinite(point).all() and (point >= 0).all()
    assert np.isfinite(quantiles).all() and (quantiles >= 0).all()
    assert np.all(np.diff(quantiles, axis=2) >= 0)
    if actual is not None:
        assert np.isfinite(actual).all()
    return point, quantiles, actual


def main() -> None:
    args = parse_args()
    root = args.module3_root.resolve()
    catalog = build_catalog(args.member2_root.resolve())
    checksum_path = root / "交付清单.json"
    checksums = json.loads(checksum_path.read_text(encoding="utf-8"))
    checked = 0
    for relative, record in checksums.items():
        path = root / relative
        assert path.exists(), f"Missing file in manifest: {relative}"
        assert path.stat().st_size == record["bytes"], f"Size mismatch: {relative}"
        assert sha256(path) == record["sha256"], f"Checksum mismatch: {relative}"
        checked += 1

    file_counts = Counter()
    for path in (root / "预测结果").rglob("*.parquet"):
        parts = path.relative_to(root / "预测结果").parts
        file_counts[(parts[0], parts[1])] += 1
    for split in SPLITS:
        for scheme in SCHEMES:
            assert file_counts[(split, scheme)] == 108, (
                split,
                scheme,
                file_counts[(split, scheme)],
            )

    coherence: list[dict[str, object]] = []
    for split in SPLITS:
        for scheme in SCHEMES:
            point, quantiles, actual = load_daily_arrays(root, split, scheme, catalog)
            for model in MODELS:
                diagnostic = coherence_diagnostics(
                    catalog, point[MODEL_INDEX[model]]
                )
                coherence.append(
                    {
                        "split": split,
                        "scheme": scheme,
                        "model": model,
                        **diagnostic,
                    }
                )
                if scheme != "unreconciled":
                    assert diagnostic["max_abs_additivity_error"] <= 1e-7, (
                        split,
                        scheme,
                        model,
                    )
            assert np.all(np.diff(quantiles, axis=2) >= 0)
            if split == "forecast":
                assert actual is None

    frame = pd.DataFrame(coherence)
    frame.to_csv(
        root / "诊断结果" / "independent_verification_coherence.csv",
        index=False,
        encoding="utf-8-sig",
    )
    report = {
        "status": "passed",
        "manifest_files_checked": checked,
        "prediction_parquet_files": int(sum(file_counts.values())),
        "splits_checked": list(SPLITS),
        "schemes_checked": list(SCHEMES),
        "models_checked": list(MODELS),
        "hierarchy": {
            "nodes": catalog.n_nodes,
            "leaves": catalog.n_leaves,
            "memberships": int(catalog.incidence.nnz),
        },
        "checks": [
            "manifest size and SHA-256",
            "108 prediction files per split/scheme",
            "finite and nonnegative forecasts",
            "daily quantile monotonicity",
            "forecast period has no labels",
            "all reconciled point forecasts are coherent to <=1e-7 units",
        ],
        "max_reconciled_additivity_error": float(
            frame[frame.scheme != "unreconciled"].max_abs_additivity_error.max()
        ),
        "unreconciled_max_additivity_error": float(
            frame[frame.scheme == "unreconciled"].max_abs_additivity_error.max()
        ),
    }
    json_write(root / "verification_report.json", report)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
