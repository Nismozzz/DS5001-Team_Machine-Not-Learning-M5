#!/usr/bin/env python3
"""Hierarchy construction and forecast reconciliation for the M5 project.

This module intentionally uses only the Member 1 hierarchy contract and the
Member 2 forecast/residual handoff.  It does not inspect future sales when
constructing top-down shares or MinT weights.
"""
from __future__ import annotations

import hashlib
import json
import re
import time
from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path
from typing import Iterable, Iterator, Sequence

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from scipy.sparse import csc_matrix


MODELS = ("seasonal_naive", "ets", "tsb", "lightgbm")
MODEL_INDEX = {name: i for i, name in enumerate(MODELS)}
DAILY_QUANTILES = (0.50, 0.90, 0.95, 0.99)
DAILY_Q_COLUMNS = ("q_050", "q_090", "q_095", "q_099")
LEVEL_KEYS: dict[int, tuple[str, ...]] = {
    1: (),
    2: ("state_id",),
    3: ("store_id",),
    4: ("cat_id",),
    5: ("dept_id",),
    6: ("state_id", "cat_id"),
    7: ("state_id", "dept_id"),
    8: ("store_id", "cat_id"),
    9: ("store_id", "dept_id"),
    10: ("item_id",),
    11: ("state_id", "item_id"),
    12: ("item_id", "store_id"),
}
EXPECTED_LEVEL_COUNTS = {
    1: 1,
    2: 3,
    3: 10,
    4: 3,
    5: 7,
    6: 9,
    7: 21,
    8: 30,
    9: 70,
    10: 3049,
    11: 9147,
    12: 30490,
}
SPLIT_SPECS = {
    "validation_1": ("final_backtest", "predictions/validation_1", 1829),
    "validation_2": ("final_backtest", "predictions/validation_2", 1857),
    "validation_3": ("final_backtest", "predictions/validation_3", 1885),
    "test": ("final_test", "predictions/test", 1913),
    "forecast": ("final_forecast", "predictions/forecast", 1941),
}
CALIBRATION_ORIGINS = (1745, 1773, 1801)
HISTORY_LOOKBACK_DAYS = 224
MINT_SHRINKAGE = 0.20
MINT_FLOOR_RATIO = 1e-8
ORIGIN_DATE = date(2011, 1, 29)


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def json_write(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False)
        + "\n",
        encoding="utf-8",
    )


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def day_to_date(day: int) -> date:
    return ORIGIN_DATE + timedelta(days=int(day) - 1)


def node_id(level: int, values: Sequence[str]) -> str:
    return f"L{level:02d}:" + json.dumps(list(values), separators=(",", ":"))


def leaf_metadata(series_id: str) -> dict[str, str]:
    match = re.fullmatch(r"(.+)_(CA|TX|WI)_([1-4])", series_id)
    require(match is not None, f"Cannot parse M5 series_id: {series_id}")
    item_id, state_id, store_number = match.groups()
    parts = item_id.split("_")
    require(len(parts) >= 2, f"Cannot parse item hierarchy: {item_id}")
    return {
        "item_id": item_id,
        "store_id": f"{state_id}_{store_number}",
        "state_id": state_id,
        "cat_id": parts[0],
        "dept_id": "_".join(parts[:2]),
    }


@dataclass
class Catalog:
    root: Path
    node_ids: list[str]
    node_level: np.ndarray
    node_index: dict[str, int]
    level_nodes: dict[int, np.ndarray]
    leaf_series: np.ndarray
    leaf_meta: pd.DataFrame
    leaf_nodes: np.ndarray
    incidence: csc_matrix

    @property
    def n_nodes(self) -> int:
        return len(self.node_ids)

    @property
    def n_leaves(self) -> int:
        return len(self.leaf_series)


def build_catalog(member2_root: Path) -> Catalog:
    """Build and validate the complete 12-level hierarchy from Member 2 coverage."""
    split_root = member2_root / "final_test" / "predictions" / "test"
    level_nodes: dict[int, np.ndarray] = {}
    for level in range(1, 13):
        coverage_path = split_root / f"L{level:02d}" / "coverage.json"
        require(coverage_path.exists(), f"Missing hierarchy coverage: {coverage_path}")
        values = json.loads(coverage_path.read_text(encoding="utf-8"))["node_ids"]
        require(
            len(values) == EXPECTED_LEVEL_COUNTS[level],
            f"Level {level} count mismatch: {len(values)}",
        )
        level_nodes[level] = np.asarray(sorted(values), dtype=object)

    leaf_node_ids = level_nodes[12].tolist()
    leaf_series = []
    for value in leaf_node_ids:
        payload = json.loads(value.split(":", 1)[1])
        leaf_series.append(f"{payload[0]}_{payload[1]}")
    leaf_meta = pd.DataFrame([leaf_metadata(s) for s in leaf_series])
    leaf_meta.insert(0, "series_id", leaf_series)
    require(len(leaf_meta) == 30490 and leaf_meta.series_id.is_unique, "Invalid leaf catalogue")

    derived: dict[int, list[str]] = {}
    for level, keys in LEVEL_KEYS.items():
        if not keys:
            values = [node_id(level, []) for _ in range(len(leaf_meta))]
        else:
            values = [
                node_id(level, row)
                for row in leaf_meta[list(keys)].itertuples(index=False, name=None)
            ]
        derived[level] = values
        observed = set(values)
        expected = set(level_nodes[level].tolist())
        require(
            observed == expected,
            f"Derived level {level} does not match Member 1/2 node IDs "
            f"({len(observed)} vs {len(expected)})",
        )

    all_nodes: list[str] = []
    node_levels: list[int] = []
    ranges: dict[int, np.ndarray] = {}
    for level in range(1, 13):
        start = len(all_nodes)
        values = level_nodes[level].tolist()
        all_nodes.extend(values)
        node_levels.extend([level] * len(values))
        ranges[level] = np.arange(start, start + len(values), dtype=np.int32)
    node_index = {value: i for i, value in enumerate(all_nodes)}
    require(len(node_index) == len(all_nodes) == 42840, "Hierarchy node count mismatch")

    membership_rows: list[np.ndarray] = []
    for level in range(1, 13):
        membership_rows.append(
            np.fromiter(
                (node_index[value] for value in derived[level]),
                dtype=np.int32,
                count=len(leaf_meta),
            )
        )
    rows = np.concatenate(membership_rows)
    columns = np.tile(np.arange(len(leaf_meta), dtype=np.int32), len(membership_rows))
    incidence = csc_matrix(
        (np.ones(len(rows), dtype=np.float64), (rows, columns)),
        shape=(len(all_nodes), len(leaf_meta)),
    )
    leaf_nodes = np.fromiter(
        (node_index[value] for value in derived[12]), dtype=np.int32, count=len(leaf_meta)
    )
    require(np.all(node_levels_arr := np.asarray(node_levels, dtype=np.int8) >= 1), "Bad level")
    return Catalog(
        root=member2_root,
        node_ids=all_nodes,
        node_level=node_levels_arr,
        node_index=node_index,
        level_nodes=ranges,
        leaf_series=np.asarray(leaf_series, dtype=object),
        leaf_meta=leaf_meta,
        leaf_nodes=leaf_nodes,
        incidence=incidence,
    )


def split_folder(member2_root: Path, split: str) -> Path:
    run_name, relative, origin = SPLIT_SPECS[split]
    require(SPLIT_SPECS[split][2] == origin, "Split origin mismatch")
    return member2_root / run_name / relative


def _read_model_file(
    path: Path,
    columns: Sequence[str],
    model: str,
) -> pd.DataFrame:
    return pq.read_table(
        path,
        columns=list(columns),
        filters=[("model", "=", model)],
    ).to_pandas()


def load_daily_base(
    member2_root: Path,
    split: str,
    catalog: Catalog,
) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
    """Load compact arrays: model x node x horizon, plus LightGBM daily quantiles."""
    folder = split_folder(member2_root, split)
    origin = SPLIT_SPECS[split][2]
    y = np.zeros((len(MODELS), catalog.n_nodes, 28), dtype=np.float64)
    q = np.zeros((catalog.n_nodes, 28, len(DAILY_QUANTILES)), dtype=np.float64)
    actual = (
        None
        if split == "forecast"
        else np.zeros((catalog.n_nodes, 28), dtype=np.float64)
    )
    actual_seen = np.zeros((catalog.n_nodes, 28), dtype=bool)

    for level in range(1, 13):
        index = catalog.level_nodes[level]
        node_ids = catalog.node_ids_at(level)  # type: ignore[attr-defined]
        code = {value: i for i, value in enumerate(node_ids)}
        for model in MODELS:
            base = folder / f"L{level:02d}" / "daily.parquet"
            wanted = ["node_id", "horizon", "yhat"]
            if model == "lightgbm":
                wanted.extend(DAILY_Q_COLUMNS)
            frame = _read_model_file(base, wanted, model)
            require(
                len(frame) == len(index) * 28,
                f"Incomplete {split} L{level:02d} {model}: {len(frame)} rows",
            )
            local = np.fromiter((code[v] for v in frame.node_id), dtype=np.int32, count=len(frame))
            horizon = frame.horizon.to_numpy(dtype=np.int16) - 1
            require(np.all((horizon >= 0) & (horizon < 28)), "Invalid horizon")
            y[MODEL_INDEX[model], index[local], horizon] = frame.yhat.to_numpy(np.float64)
            if model == "lightgbm":
                q[index[local], horizon, :] = frame.loc[:, DAILY_Q_COLUMNS].to_numpy(
                    np.float64
                )

        if actual is not None:
            residual_path = folder / f"L{level:02d}" / "residuals.parquet"
            frame = _read_model_file(
                residual_path, ["node_id", "horizon", "y_true"], "lightgbm"
            )
            local = np.fromiter((code[v] for v in frame.node_id), dtype=np.int32, count=len(frame))
            horizon = frame.horizon.to_numpy(dtype=np.int16) - 1
            actual[index[local], horizon] = frame.y_true.to_numpy(np.float64)
            actual_seen[index[local], horizon] = True

    require(np.isfinite(y).all() and (y >= 0).all(), "Invalid point forecasts")
    require(
        np.isfinite(q).all()
        and (q >= 0).all()
        and np.all(np.diff(q, axis=2) >= 0),
        "Invalid LightGBM daily quantiles",
    )
    if actual is not None:
        require(actual_seen.all() and np.isfinite(actual).all(), "Incomplete actual labels")
    return y, q, actual


def _catalog_nodes_at(catalog: Catalog, level: int) -> list[str]:
    return [catalog.node_ids[i] for i in catalog.level_nodes[level]]


# Small convenience method registered outside the dataclass so pickling/simple
# construction remains unchanged.
Catalog.node_ids_at = _catalog_nodes_at  # type: ignore[attr-defined]


def load_period_quantiles(
    member2_root: Path,
    split: str,
    catalog: Catalog,
) -> tuple[np.ndarray, np.ndarray]:
    folder = split_folder(member2_root, split)
    first = pd.read_parquet(folder / "L01" / "period.parquet", columns=["tau"])
    taus = np.sort(first.tau.unique())
    require(len(taus) == 23, "Unexpected period quantile grid")
    values = np.zeros((catalog.n_nodes, len(taus)), dtype=np.float64)
    seen = np.zeros((catalog.n_nodes, len(taus)), dtype=bool)
    tau_index = {float(value): i for i, value in enumerate(taus)}
    for level in range(1, 13):
        index = catalog.level_nodes[level]
        node_ids = catalog.node_ids_at(level)  # type: ignore[attr-defined]
        code = {value: i for i, value in enumerate(node_ids)}
        frame = pq.read_table(
            folder / f"L{level:02d}" / "period.parquet",
            columns=["node_id", "tau", "forecast"],
            filters=[("model", "=", "lightgbm")],
        ).to_pandas()
        require(len(frame) == len(index) * len(taus), "Incomplete period quantiles")
        local = np.fromiter((code[v] for v in frame.node_id), dtype=np.int32, count=len(frame))
        qidx = np.fromiter(
            (tau_index[float(v)] for v in frame.tau), dtype=np.int16, count=len(frame)
        )
        values[index[local], qidx] = frame.forecast.to_numpy(np.float64)
        seen[index[local], qidx] = True
    require(
        seen.all()
        and np.isfinite(values).all()
        and (values >= 0).all()
        and np.all(np.diff(values, axis=1) >= 0),
        "Invalid period quantiles",
    )
    return values, taus


def historical_leaf_observations(
    member2_root: Path,
    split: str,
    catalog: Catalog,
) -> tuple[np.ndarray, np.ndarray]:
    """Return weekday sums and all-day sums from labels known at the origin."""
    target_origin = SPLIT_SPECS[split][2]
    sources: list[tuple[int, Path]] = []
    calibration_root = member2_root / "final_calibration" / "predictions"
    for origin in CALIBRATION_ORIGINS:
        sources.append(
            (origin, calibration_root / f"calibration_{origin}" / "L12" / "residuals.parquet")
        )
    for name in ("validation_1", "validation_2", "validation_3", "test"):
        origin = SPLIT_SPECS[name][2]
        if origin < target_origin:
            sources.append(
                (origin, split_folder(member2_root, name) / "L12" / "residuals.parquet")
            )
    sources.sort()

    leaf_pos = np.zeros(catalog.n_nodes, dtype=np.int32)
    leaf_pos[catalog.leaf_nodes] = np.arange(catalog.n_leaves, dtype=np.int32)
    weekday_sums = np.zeros((7, catalog.n_leaves), dtype=np.float64)
    all_sum = np.zeros(catalog.n_leaves, dtype=np.float64)
    used_origins: list[int] = []
    for origin, path in sources:
        if origin >= target_origin or not path.exists():
            continue
        frame = _read_model_file(
            path, ["node_id", "target_d", "y_true"], "lightgbm"
        )
        target_d = frame.target_d.to_numpy(np.int16)
        mask = (target_d <= target_origin) & (
            target_d >= target_origin - HISTORY_LOOKBACK_DAYS + 1
        )
        if not mask.any():
            continue
        frame = frame.loc[mask]
        target_d = target_d[mask]
        local = np.fromiter(
            (leaf_pos[catalog.node_index[v]] for v in frame.node_id),
            dtype=np.int32,
            count=len(frame),
        )
        values = frame.y_true.to_numpy(np.float64)
        weekdays = np.fromiter(
            (day_to_date(day).weekday() for day in target_d),
            dtype=np.int64,
            count=len(target_d),
        )
        weekday_sums += np.bincount(
            (weekdays * catalog.n_leaves + local).ravel(),
            weights=values,
            minlength=7 * catalog.n_leaves,
        ).reshape(7, catalog.n_leaves)
        all_sum += np.bincount(local, weights=values, minlength=catalog.n_leaves)
        used_origins.append(origin)
    require(len(used_origins) > 0, f"No leak-free history available for {split}")
    return weekday_sums, all_sum


def top_down_shares(
    weekday_sums: np.ndarray,
    all_sum: np.ndarray,
    origin: int,
) -> tuple[np.ndarray, pd.DataFrame]:
    nleaves = len(all_sum)
    shares = np.zeros((28, nleaves), dtype=np.float64)
    rows: list[dict[str, object]] = []
    all_positive = np.maximum(all_sum, 0.0)
    all_total = float(all_positive.sum())
    equal = np.full(nleaves, 1.0 / nleaves)
    for i in range(28):
        target_d = origin + i + 1
        weekday = day_to_date(target_d).weekday()
        values = np.maximum(weekday_sums[weekday], 0.0)
        total = float(values.sum())
        fallback = "none"
        if total <= 0:
            fallback = "all_day_history"
            values = all_positive
            total = all_total
        if total <= 0:
            fallback = "equal_leaf"
            shares[i] = equal
        else:
            shares[i] = values / total
        zero_share_count = int((shares[i] == 0).sum())
        rows.append(
            {
                "horizon": i + 1,
                "target_d": target_d,
                "target_weekday": weekday,
                "fallback": fallback,
                "zero_share_leaves": zero_share_count,
                "positive_share_leaves": nleaves - zero_share_count,
            }
        )
    require(np.allclose(shares.sum(axis=1), 1.0, atol=1e-12), "Top-down shares do not sum")
    return shares, pd.DataFrame(rows)


def _shrink_weights(
    mean_square: np.ndarray,
    counts: np.ndarray,
    shrinkage: float = MINT_SHRINKAGE,
) -> tuple[np.ndarray, np.ndarray, float]:
    valid = counts > 0
    require(valid.any(), "No residuals for MinT weights")
    pooled = float(np.average(mean_square[valid], weights=counts[valid]))
    pooled = max(pooled, 1e-12)
    variance = (1.0 - shrinkage) * mean_square + shrinkage * pooled
    variance = np.maximum(variance, pooled * MINT_FLOOR_RATIO)
    weight = 1.0 / variance
    weight /= weight.mean()
    return variance, weight, pooled


def estimate_mint_weights(
    member2_root: Path,
    catalog: Catalog,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Estimate daily and 28-day diagonal residual variances from calibration."""
    s1 = np.zeros((len(MODELS), catalog.n_nodes), dtype=np.float64)
    s2 = np.zeros_like(s1)
    counts = np.zeros_like(s1)
    p1 = np.zeros_like(s1)
    p2 = np.zeros_like(s1)
    pcounts = np.zeros_like(s1)

    calibration_root = member2_root / "final_calibration" / "predictions"
    for origin in CALIBRATION_ORIGINS:
        folder = calibration_root / f"calibration_{origin}"
        for level in range(1, 13):
            index = catalog.level_nodes[level]
            node_ids = catalog.node_ids_at(level)  # type: ignore[attr-defined]
            code = {value: i for i, value in enumerate(node_ids)}
            for model in MODELS:
                frame = _read_model_file(
                    folder / f"L{level:02d}" / "residuals.parquet",
                    ["node_id", "residual"],
                    model,
                )
                local = np.fromiter(
                    (code[v] for v in frame.node_id), dtype=np.int32, count=len(frame)
                )
                residual = frame.residual.to_numpy(np.float64)
                require(np.isfinite(residual).all(), "Non-finite calibration residual")
                global_nodes = index[local]
                mi = MODEL_INDEX[model]
                s1[mi, global_nodes] += residual
                s2[mi, global_nodes] += residual**2
                counts[mi, global_nodes] += 1

                grouped = pd.DataFrame(
                    {"node": global_nodes, "residual": residual}
                ).groupby("node", sort=False).residual.sum()
                groups = grouped.index.to_numpy(dtype=np.int32)
                p1[mi, groups] += grouped.to_numpy(np.float64)
                p2[mi, groups] += grouped.to_numpy(np.float64) ** 2
                pcounts[mi, groups] += 1

    daily_meansq = np.divide(
        s2, counts, out=np.zeros_like(s2), where=counts > 0
    )
    period_meansq = np.divide(
        p2, pcounts, out=np.zeros_like(p2), where=pcounts > 0
    )
    daily_var = np.zeros_like(s1)
    daily_weight = np.zeros_like(s1)
    period_var = np.zeros_like(s1)
    period_weight = np.zeros_like(s1)
    for mi in range(len(MODELS)):
        daily_var[mi], daily_weight[mi], _ = _shrink_weights(
            daily_meansq[mi], counts[mi]
        )
        period_var[mi], period_weight[mi], _ = _shrink_weights(
            period_meansq[mi], pcounts[mi]
        )
    return daily_var, daily_weight, period_var, period_weight


def reconstitute(incidence: csc_matrix, leaf_values: np.ndarray) -> np.ndarray:
    return np.asarray(incidence @ leaf_values)


def pcg_reconcile(
    incidence: csc_matrix,
    weights: np.ndarray,
    base_nodes: np.ndarray,
    leaf_start: np.ndarray,
    tolerance: float = 1e-8,
    max_iter: int = 500,
) -> tuple[np.ndarray, dict[str, float]]:
    """Solve independent right-hand sides with column-wise PCG.

    The same sparse system is shared by all forecast horizons/quantiles.  Each
    column nevertheless keeps its own CG scalar, which is essential for
    preserving monotone quantiles without relying on a post-hoc repair.
    """
    require(base_nodes.ndim == 2 and leaf_start.ndim == 2, "Need two-dimensional blocks")
    w = np.asarray(weights, dtype=np.float64)
    w = w / w.mean()
    diag = np.asarray(incidence.T @ w).ravel()
    require((diag > 0).all(), "MinT preconditioner has non-positive entries")
    b = np.asarray(incidence.T @ (w[:, None] * base_nodes))

    def amult(vector: np.ndarray) -> np.ndarray:
        return np.asarray(incidence.T @ (w[:, None] * (incidence @ vector)))

    x = np.maximum(np.asarray(leaf_start, dtype=np.float64).copy(), 0.0)
    r = b - amult(x)
    z = r / diag[:, None]
    p = z.copy()
    rz = np.sum(r * z, axis=0)
    bnorm = np.maximum(np.sqrt(np.sum(b * b, axis=0)), 1e-12)
    relative_columns = np.sqrt(np.sum(r * r, axis=0)) / bnorm
    initial = float(relative_columns.max())
    converged = initial <= tolerance
    iterations = 0
    for iterations in range(1, max_iter + 1):
        ap = amult(p)
        denominator = np.sum(p * ap, axis=0)
        require(np.all(denominator > 0), "PCG breakdown")
        alpha = rz / denominator
        x += p * alpha
        r -= ap * alpha
        relative_columns = np.sqrt(np.sum(r * r, axis=0)) / bnorm
        if float(relative_columns.max()) <= tolerance:
            converged = True
            break
        z = r / diag[:, None]
        rz_new = np.sum(r * z, axis=0)
        beta = rz_new / rz
        p = z + p * beta
        rz = rz_new
    return x, {
        "cg_iterations": iterations,
        "cg_relative_residual": float(relative_columns.max()),
        "cg_converged": float(converged),
        "cg_initial_residual": initial,
    }


def mint_reconcile(
    catalog: Catalog,
    weights: np.ndarray,
    base_nodes: np.ndarray,
    quantile_shape: tuple[int, int] | None = None,
    tolerance: float = 1e-8,
    max_iter: int = 500,
) -> tuple[np.ndarray, dict[str, float]]:
    """Reconcile a node x columns block under nonnegative leaf forecasts.

    The first WLS solution is inspected, not silently clipped.  Negatively
    signed leaves form an active set fixed at zero, and the remaining leaves
    are re-estimated.  This preserves the hierarchy sum while avoiding
    negative sales.  Quantile columns are monotone-repaired at leaves and then
    re-aggregated.
    """
    negative_leaf_before = 0
    negative_reconciled_before = 0
    active_rounds = 0
    total_cg_iterations = 0
    initial_residual = 0.0
    final_residual = 0.0
    all_converged = True
    leaves = np.zeros((catalog.n_leaves, base_nodes.shape[1]), dtype=np.float64)
    for column in range(base_nodes.shape[1]):
        base = base_nodes[:, column]
        coherent_leaf = base[catalog.leaf_nodes]
        coherent_check = reconstitute(catalog.incidence, coherent_leaf[:, None])
        scale = max(float(np.max(np.abs(base))), 1.0)
        if (
            np.all(coherent_leaf >= 0)
            and np.max(np.abs(coherent_check[:, 0] - base)) <= scale * 1e-10
        ):
            leaves[:, column] = coherent_leaf
            continue
        free = np.ones(catalog.n_leaves, dtype=bool)
        if quantile_shape is not None:
            quantile_count = quantile_shape[1]
            lower = (
                leaves[:, column - 1].copy()
                if column % quantile_count
                else np.zeros(catalog.n_leaves, dtype=np.float64)
            )
        else:
            lower = np.zeros(catalog.n_leaves, dtype=np.float64)
        value = lower.copy()
        first = True
        while True:
            active_rounds += 1
            free_incidence = catalog.incidence[:, free]
            free_start = value[free, None]
            fixed_adjustment = np.asarray(
                catalog.incidence[:, ~free] @ value[~free, None]
            ) if (~free).any() else np.zeros((catalog.n_nodes, 1))
            solution, diagnostics = pcg_reconcile(
                free_incidence,
                weights,
                base[:, None] - fixed_adjustment,
                free_start,
                tolerance=tolerance,
                max_iter=max_iter,
            )
            total_cg_iterations += int(diagnostics["cg_iterations"])
            final_residual = max(
                final_residual, float(diagnostics["cg_relative_residual"])
            )
            all_converged &= bool(diagnostics["cg_converged"])
            value[free] = solution[:, 0]
            if first:
                initial_residual = max(
                    initial_residual, float(diagnostics["cg_initial_residual"])
                )
                negative_leaf_before += int((value < 0).sum())
                negative_reconciled_before += int(
                    (reconstitute(catalog.incidence, value[:, None]) < 0).sum()
                )
                first = False
            violating = free & (value < lower - 1e-10)
            if not violating.any():
                break
            free[violating] = False
            value[~free] = lower[~free]
            require(free.any(), "All leaves entered the nonnegative active set")
        value = np.maximum(value, lower)
        leaves[:, column] = value

    crossings_before = 0
    if quantile_shape is not None:
        moved = np.diff(leaves.reshape(catalog.n_leaves, *quantile_shape), axis=2) < 0
        crossings_before = int(np.any(moved, axis=2).sum())
        repaired = np.maximum.accumulate(
            leaves.reshape(catalog.n_leaves, *quantile_shape), axis=2
        )
        leaves = repaired.reshape(catalog.n_leaves, -1)
    reconciled = reconstitute(catalog.incidence, leaves)
    return reconciled, {
        "cg_iterations": total_cg_iterations,
        "cg_relative_residual": final_residual,
        "cg_converged": float(all_converged),
        "cg_initial_residual": initial_residual,
        "active_set_rounds": active_rounds,
        "negative_leaf_before_clip": negative_leaf_before,
        "negative_reconciled_before_clip": negative_reconciled_before,
        "quantile_crossings_before_repair": crossings_before,
        "negative_after_clip": int((reconciled < 0).sum()),
    }


def daily_bottom_up(catalog: Catalog, base: np.ndarray) -> np.ndarray:
    """Bottom-up reconciliation for one model, shape node x horizon."""
    return reconstitute(catalog.incidence, base[catalog.leaf_nodes])


def daily_top_down(
    catalog: Catalog,
    root_values: np.ndarray,
    shares: np.ndarray,
) -> np.ndarray:
    """root_values is horizon x quantiles; shares is horizon x leaves."""
    require(shares.shape == (28, catalog.n_leaves), "Top-down share shape mismatch")
    if root_values.ndim == 1:
        leaves = (root_values[:, None] * shares).T
        return reconstitute(catalog.incidence, leaves)
    else:
        leaves = shares[:, :, None] * root_values[:, None, :]
        leaves = np.transpose(leaves, (1, 0, 2)).reshape(catalog.n_leaves, -1)
        return reconstitute(catalog.incidence, leaves)


def reconcile_quantile_block(
    catalog: Catalog,
    scheme: str,
    base_quantiles: np.ndarray,
    *,
    shares: np.ndarray | None = None,
    weights: np.ndarray | None = None,
    layout: tuple[int, int] | None = None,
    tolerance: float = 1e-8,
    max_iter: int = 500,
) -> tuple[np.ndarray, dict[str, float] | None]:
    """Reconcile an external node-by-quantile block.

    This is the post-calibration hook for point-only models such as Seasonal
    Naive, ETS and TSB.  Module 4 can first construct daily or 28-day quantiles
    from historical errors, then pass the resulting node x columns matrix here
    so that the final inventory input remains hierarchy-coherent.

    ``layout=(horizons, quantiles)`` defines the column order.  Top-down
    requires the exact 28-day share matrix.  MinT requires a node-level weight
    vector estimated without future labels.
    """
    base = np.asarray(base_quantiles, dtype=np.float64)
    require(base.shape[0] == catalog.n_nodes, "External quantile node count mismatch")
    if layout is not None:
        require(
            base.shape[1] == layout[0] * layout[1],
            "External quantile column count differs from layout",
        )
    if scheme == "unreconciled":
        return base.copy(), None
    if scheme == "bottom_up":
        leaves = base[catalog.leaf_nodes]
        return reconstitute(catalog.incidence, leaves), None
    if scheme == "top_down":
        require(layout is not None and shares is not None, "Top-down inputs missing")
        require(layout[0] == 28, "Top-down layout must use 28 horizons")
        root = base[0].reshape(layout)
        return daily_top_down(catalog, root, shares), None
    if scheme == "mint":
        require(weights is not None, "MinT weights missing")
        return mint_reconcile(
            catalog,
            weights,
            base,
            quantile_shape=layout,
            tolerance=tolerance,
            max_iter=max_iter,
        )
    raise ValueError(f"Unknown reconciliation scheme: {scheme}")


def coherence_diagnostics(
    catalog: Catalog,
    values: np.ndarray,
) -> dict[str, float]:
    """Compare every node with the leaf-based reconstitution."""
    reconstructed = reconstitute(catalog.incidence, values[catalog.leaf_nodes])
    difference = np.abs(values - reconstructed)
    denominator = np.maximum(np.abs(values), 1.0)
    relative = difference / denominator
    return {
        "max_abs_additivity_error": float(difference.max()),
        "max_relative_additivity_error": float(relative.max()),
        "additivity_violations_gt_1e8": int((relative > 1e-8).sum()),
        "sum_abs_additivity_error": float(difference.sum()),
    }


def point_metrics(
    values: np.ndarray,
    actual: np.ndarray,
) -> dict[str, float | int]:
    error = values - actual
    absolute = np.abs(error)
    actual_sum = float(actual.sum())
    return {
        "rows": int(actual.size),
        "actual_sum": actual_sum,
        "forecast_sum": float(values.sum()),
        "absolute_error_sum": float(absolute.sum()),
        "squared_error_sum": float((error**2).sum()),
        "wape": float(absolute.sum() / actual_sum) if actual_sum > 0 else np.nan,
        "mae": float(absolute.mean()),
        "rmse": float(np.sqrt((error**2).mean())),
        "bias": float(error.mean()),
        "relative_bias": float(error.sum() / actual_sum) if actual_sum > 0 else np.nan,
    }


def pinball(values: np.ndarray, actual: np.ndarray, tau: float) -> float:
    residual = actual - values
    return float(np.mean(np.maximum(tau * residual, (tau - 1.0) * residual)))


def write_daily_predictions(
    output_root: Path,
    split: str,
    scheme: str,
    origin: int,
    catalog: Catalog,
    point: np.ndarray,
    quantiles: np.ndarray,
    actual: np.ndarray | None,
) -> int:
    rows_written = 0
    for model in MODELS:
        mi = MODEL_INDEX[model]
        for level in range(1, 13):
            index = catalog.level_nodes[level]
            node_ids = catalog.node_ids_at(level)  # type: ignore[attr-defined]
            horizons = np.tile(np.arange(1, 29, dtype=np.int16), len(index))
            frame = pd.DataFrame(
                {
                    "node_id": pd.Categorical(np.repeat(node_ids, 28)),
                    "level": np.int8(level),
                    "model": pd.Categorical([model] * (len(index) * 28)),
                    "origin_d": np.int16(origin),
                    "target_d": (origin + horizons).astype(np.int16),
                    "horizon": horizons,
                    "reconciliation": pd.Categorical([scheme] * (len(index) * 28)),
                    "yhat": point[mi, index, :].reshape(-1),
                    "y_true": (
                        None
                        if actual is None
                        else actual[index, :].reshape(-1)
                    ),
                }
            )
            if model == "lightgbm":
                qblock = quantiles[index, :, :]
                for qi, column in enumerate(DAILY_Q_COLUMNS):
                    frame[column] = qblock[:, :, qi].reshape(-1)
            else:
                for column in DAILY_Q_COLUMNS:
                    frame[column] = np.nan
            path = (
                output_root
                / "预测结果"
                / split
                / scheme
                / "daily"
                / model
                / f"L{level:02d}.parquet"
            )
            path.parent.mkdir(parents=True, exist_ok=True)
            frame.to_parquet(path, index=False, compression="zstd")
            rows_written += len(frame)
    return rows_written


def write_period_predictions(
    output_root: Path,
    split: str,
    scheme: str,
    origin: int,
    catalog: Catalog,
    period_point: np.ndarray,
    period_q: np.ndarray,
    taus: np.ndarray,
    actual_sum: np.ndarray | None,
) -> int:
    rows_written = 0
    for model in MODELS:
        mi = MODEL_INDEX[model]
        for level in range(1, 13):
            index = catalog.level_nodes[level]
            node_ids = catalog.node_ids_at(level)  # type: ignore[attr-defined]
            frame = pd.DataFrame(
                {
                    "node_id": pd.Categorical(node_ids),
                    "level": np.int8(level),
                    "model": pd.Categorical([model] * len(index)),
                    "origin_d": np.int16(origin),
                    "period_start_d": np.int16(origin + 1),
                    "period_end_d": np.int16(origin + 28),
                    "reconciliation": pd.Categorical([scheme] * len(index)),
                    "forecast": period_point[mi, index],
                    "y_true": None if actual_sum is None else actual_sum[index],
                }
            )
            path = (
                output_root
                / "预测结果"
                / split
                / scheme
                / "period_point"
                / model
                / f"L{level:02d}.parquet"
            )
            path.parent.mkdir(parents=True, exist_ok=True)
            frame.to_parquet(path, index=False, compression="zstd")
            rows_written += len(frame)

    for level in range(1, 13):
        index = catalog.level_nodes[level]
        node_ids = catalog.node_ids_at(level)  # type: ignore[attr-defined]
        frame = pd.DataFrame(
            {
                "node_id": pd.Categorical(np.repeat(node_ids, len(taus))),
                "level": np.int8(level),
                "model": pd.Categorical(["lightgbm"] * (len(index) * len(taus))),
                "origin_d": np.int16(origin),
                "period_start_d": np.int16(origin + 1),
                "period_end_d": np.int16(origin + 28),
                "reconciliation": pd.Categorical([scheme] * (len(index) * len(taus))),
                "tau": np.tile(taus, len(index)),
                "forecast": period_q[index, :].reshape(-1),
            }
        )
        path = (
            output_root
            / "预测结果"
            / split
            / scheme
            / "period_quantile"
            / f"L{level:02d}.parquet"
        )
        path.parent.mkdir(parents=True, exist_ok=True)
        frame.to_parquet(path, index=False, compression="zstd")
        rows_written += len(frame)
    return rows_written


def scheme_rows(
    scheme: str,
    split: str,
    point: np.ndarray,
    quantiles: np.ndarray,
    actual: np.ndarray | None,
    catalog: Catalog,
    daily_taus: Sequence[float],
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    metric_rows: list[dict[str, object]] = []
    coherence_rows: list[dict[str, object]] = []
    for model in MODELS:
        mi = MODEL_INDEX[model]
        coherence = coherence_diagnostics(catalog, point[mi])
        coherence_rows.append(
            {
                "split": split,
                "scheme": scheme,
                "model": model,
                "max_abs_additivity_error": coherence["max_abs_additivity_error"],
                "max_relative_additivity_error": coherence[
                    "max_relative_additivity_error"
                ],
                "additivity_violations_gt_1e8": coherence[
                    "additivity_violations_gt_1e8"
                ],
                "sum_abs_additivity_error": coherence["sum_abs_additivity_error"],
            }
        )
        for level in range(1, 13):
            index = catalog.level_nodes[level]
            row: dict[str, object] = {
                "split": split,
                "scheme": scheme,
                "model": model,
                "level": level,
                "nodes": len(index),
            }
            if actual is not None:
                row.update(point_metrics(point[mi, index], actual[index]))
                if model == "lightgbm":
                    for qi, tau in enumerate(daily_taus):
                        row[f"pinball_{tau:g}"] = pinball(
                            quantiles[index, :, qi], actual[index], tau
                        )
            metric_rows.append(row)
    return metric_rows, coherence_rows


def period_scheme_rows(
    scheme: str,
    split: str,
    period_point: np.ndarray,
    period_q: np.ndarray,
    actual_sum: np.ndarray | None,
    catalog: Catalog,
    taus: Sequence[float],
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for model in MODELS:
        mi = MODEL_INDEX[model]
        for level in range(1, 13):
            index = catalog.level_nodes[level]
            row: dict[str, object] = {
                "split": split,
                "scheme": scheme,
                "model": model,
                "level": level,
                "nodes": len(index),
            }
            if actual_sum is not None:
                row.update(point_metrics(period_point[mi, index], actual_sum[index]))
                if model == "lightgbm":
                    for qi, tau in enumerate(taus):
                        row[f"pinball_{tau:g}"] = pinball(
                            period_q[index, qi], actual_sum[index], tau
                        )
            rows.append(row)
    return rows

