#!/usr/bin/env python3
"""Run Module 3: bottom-up, top-down and diagonal-MinT reconciliation."""
from __future__ import annotations

import argparse
import json
import platform
import shutil
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow
import scipy

from reconciliation import (
    DAILY_QUANTILES,
    MODELS,
    MODEL_INDEX,
    SPLIT_SPECS,
    Catalog,
    build_catalog,
    coherence_diagnostics,
    daily_bottom_up,
    daily_top_down,
    estimate_mint_weights,
    historical_leaf_observations,
    json_write,
    load_daily_base,
    load_period_quantiles,
    mint_reconcile,
    period_scheme_rows,
    scheme_rows,
    sha256,
    top_down_shares,
    write_daily_predictions,
    write_period_predictions,
)


SCHEMES = ("unreconciled", "bottom_up", "top_down", "mint")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--member2-root",
        type=Path,
        required=True,
        help="Folder containing final_backtest/, final_test/, final_forecast/ and final_calibration/.",
    )
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument(
        "--splits",
        nargs="+",
        default=["validation_1", "validation_2", "validation_3", "test", "forecast"],
        choices=sorted(SPLIT_SPECS),
    )
    parser.add_argument(
        "--schemes",
        nargs="+",
        default=list(SCHEMES),
        choices=SCHEMES,
    )
    parser.add_argument("--cg-tolerance", type=float, default=1e-8)
    parser.add_argument("--cg-max-iter", type=int, default=500)
    parser.add_argument(
        "--keep-existing",
        action="store_true",
        help="Do not remove an existing prediction tree before writing.",
    )
    return parser.parse_args()


def write_catalogue(output_root: Path, catalog: Catalog) -> None:
    nodes = pd.DataFrame(
        {
            "node_id": catalog.node_ids,
            "level": catalog.node_level,
        }
    )
    nodes.to_parquet(
        output_root / "hierarchy_nodes_final.parquet", index=False, compression="zstd"
    )

    mappings: list[pd.DataFrame] = []
    for level in range(1, 13):
        index = catalog.level_nodes[level]
        memberships = catalog.incidence[index, :].tocoo()
        mappings.append(
            pd.DataFrame(
                {
                    "level": np.full(len(memberships.row), level, dtype=np.int8),
                    "node_id": pd.Categorical(
                        [
                            catalog.node_ids[int(index[row])]
                            for row in memberships.row
                        ]
                    ),
                    "series_id": pd.Categorical(catalog.leaf_series[memberships.col]),
                }
            )
        )
    mapping = pd.concat(mappings, ignore_index=True)
    mapping.to_parquet(
        output_root / "hierarchy_map_final.parquet",
        index=False,
        compression="zstd",
    )
    catalog.leaf_meta.to_parquet(
        output_root / "leaf_metadata_final.parquet",
        index=False,
        compression="zstd",
    )
    summary = (
        mapping.groupby("level", observed=True)
        .agg(nodes=("node_id", "nunique"), memberships=("node_id", "size"))
        .reset_index()
    )
    summary.to_csv(
        output_root / "诊断结果" / "hierarchy_audit.csv", index=False, encoding="utf-8-sig"
    )


def write_mint_weights(
    output_root: Path,
    catalog: Catalog,
    daily_var: np.ndarray,
    daily_weight: np.ndarray,
    period_var: np.ndarray,
    period_weight: np.ndarray,
) -> None:
    frames: list[pd.DataFrame] = []
    for model in MODELS:
        mi = MODEL_INDEX[model]
        frames.append(
            pd.DataFrame(
                {
                    "model": pd.Categorical([model] * catalog.n_nodes),
                    "node_id": pd.Categorical(catalog.node_ids),
                    "level": catalog.node_level,
                    "daily_error_variance": daily_var[mi],
                    "daily_mint_weight": daily_weight[mi],
                    "period_error_variance": period_var[mi],
                    "period_mint_weight": period_weight[mi],
                }
            )
        )
    weights = pd.concat(frames, ignore_index=True)
    weights.to_parquet(
        output_root / "mint_weights.parquet", index=False, compression="zstd"
    )
    summary = (
        weights.groupby(["model", "level"], observed=True)
        .agg(
            nodes=("node_id", "size"),
            daily_variance_mean=("daily_error_variance", "mean"),
            daily_variance_median=("daily_error_variance", "median"),
            daily_weight_min=("daily_mint_weight", "min"),
            daily_weight_max=("daily_mint_weight", "max"),
            period_variance_mean=("period_error_variance", "mean"),
            period_variance_median=("period_error_variance", "median"),
            period_weight_min=("period_mint_weight", "min"),
            period_weight_max=("period_mint_weight", "max"),
        )
        .reset_index()
    )
    summary.to_csv(
        output_root / "诊断结果" / "mint_weight_summary.csv",
        index=False,
        encoding="utf-8-sig",
    )


def top_down_quantiles(
    catalog: Catalog,
    root_quantiles: np.ndarray,
    shares: np.ndarray,
) -> np.ndarray:
    """Reconcile horizon x quantile root forecasts through historical shares."""
    leaves = shares[:, :, None] * root_quantiles[:, None, :]
    flat = np.transpose(leaves, (1, 0, 2)).reshape(catalog.n_leaves, -1)
    return np.asarray(catalog.incidence @ flat).reshape(
        catalog.n_nodes, root_quantiles.shape[0], root_quantiles.shape[1]
    )


def arrays_for_scheme(
    scheme: str,
    catalog: Catalog,
    base: np.ndarray,
    base_q: np.ndarray,
    shares: np.ndarray,
    weights: np.ndarray,
    tolerance: float,
    max_iter: int,
) -> tuple[np.ndarray, np.ndarray, list[dict[str, object]]]:
    point = np.zeros_like(base)
    quantiles = np.zeros_like(base_q)
    cg_rows: list[dict[str, object]] = []
    if scheme == "unreconciled":
        return base.copy(), base_q.copy(), cg_rows

    if scheme == "bottom_up":
        for model in MODELS:
            mi = MODEL_INDEX[model]
            point[mi] = daily_bottom_up(catalog, base[mi])
        flat_q = base_q[catalog.leaf_nodes].reshape(catalog.n_leaves, -1)
        quantiles = np.asarray(catalog.incidence @ flat_q).reshape(
            catalog.n_nodes, 28, len(DAILY_QUANTILES)
        )
        return point, quantiles, cg_rows

    if scheme == "top_down":
        for model in MODELS:
            mi = MODEL_INDEX[model]
            point[mi] = daily_top_down(catalog, base[mi, 0], shares)
        quantiles = top_down_quantiles(catalog, base_q[0], shares)
        return point, quantiles, cg_rows

    if scheme != "mint":
        raise ValueError(f"Unknown scheme: {scheme}")
    for model in MODELS:
        mi = MODEL_INDEX[model]
        reconciled, diagnostics = mint_reconcile(
            catalog,
            weights[mi],
            base[mi],
            tolerance=tolerance,
            max_iter=max_iter,
        )
        point[mi] = reconciled
        cg_rows.append(
            {
                "block": "daily_point",
                "model": model,
                **diagnostics,
            }
        )

    lgbm = MODEL_INDEX["lightgbm"]
    flat_q = base_q.reshape(catalog.n_nodes, -1)
    reconciled_q, diagnostics = mint_reconcile(
        catalog,
        weights[lgbm],
        flat_q,
        quantile_shape=(28, len(DAILY_QUANTILES)),
        tolerance=tolerance,
        max_iter=max_iter,
    )
    quantiles = reconciled_q.reshape(
        catalog.n_nodes, 28, len(DAILY_QUANTILES)
    )
    cg_rows.append(
        {
            "block": "daily_quantile",
            "model": "lightgbm",
            **diagnostics,
        }
    )
    return point, quantiles, cg_rows


def representative_rows(
    split: str,
    scheme: str,
    target_d: int,
    catalog: Catalog,
    point: np.ndarray,
    quantiles: np.ndarray,
    actual: np.ndarray | None,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    horizon_index = target_d - SPLIT_SPECS[split][2] - 1
    for level in range(1, 13):
        index = catalog.level_nodes[level]
        # The first node at each level is deterministic and easy to cross-check.
        node = int(index[0])
        for model in MODELS:
            mi = MODEL_INDEX[model]
            rows.append(
                {
                    "split": split,
                    "scheme": scheme,
                    "model": model,
                    "level": level,
                    "node_id": catalog.node_ids[node],
                    "target_d": target_d,
                    "horizon": horizon_index + 1,
                    "point_forecast": point[mi, node, horizon_index],
                    "q_0.95": (
                        quantiles[node, horizon_index, 2]
                        if model == "lightgbm"
                        else np.nan
                    ),
                    "actual": (
                        None if actual is None else actual[node, horizon_index]
                    ),
                }
            )
    return rows


def write_top_down_history(
    output_root: Path,
    split: str,
    catalog: Catalog,
    shares: np.ndarray,
    fallback: pd.DataFrame,
) -> None:
    folder = output_root / "诊断结果" / "top_down_shares"
    folder.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame(
        {
            "split": pd.Categorical([split] * (catalog.n_leaves * 28)),
            "horizon": np.repeat(np.arange(1, 29, dtype=np.int8), catalog.n_leaves),
            "series_id": pd.Categorical(np.tile(catalog.leaf_series, 28)),
            "allocation_share": shares.reshape(-1),
        }
    )
    frame.to_parquet(folder / f"{split}.parquet", index=False, compression="zstd")
    fallback = fallback.copy()
    fallback.insert(0, "split", split)
    fallback.to_csv(
        output_root / "诊断结果" / "top_down_share_fallbacks.csv",
        mode="a",
        index=False,
        header=not (
            output_root / "诊断结果" / "top_down_share_fallbacks.csv"
        ).exists(),
        encoding="utf-8-sig",
    )


def write_checksums(output_root: Path) -> None:
    records: dict[str, dict[str, object]] = {}
    excluded = {"交付清单.json", "verification_report.json"}
    for path in sorted(output_root.rglob("*")):
        if not path.is_file():
            continue
        relative = path.relative_to(output_root).as_posix()
        if (
            relative in excluded
            or "__pycache__" in path.parts
            or path.name in {".DS_Store"}
            or ".git" in path.parts
        ):
            continue
        records[relative] = {"bytes": path.stat().st_size, "sha256": sha256(path)}
    json_write(output_root / "交付清单.json", records)


def main() -> None:
    args = parse_args()
    started = time.time()
    member2_root = args.member2_root.resolve()
    output_root = args.output_root.resolve()
    require_root = member2_root / "final_test"
    if not require_root.exists():
        raise FileNotFoundError(f"Member 2 handoff not found: {member2_root}")
    if output_root.exists() and not args.keep_existing:
        prediction_tree = output_root / "预测结果"
        if prediction_tree.exists():
            shutil.rmtree(prediction_tree)
    for name in ("诊断结果", "预测结果"):
        (output_root / name).mkdir(parents=True, exist_ok=True)

    print("Building and validating the 12-level hierarchy...", flush=True)
    catalog = build_catalog(member2_root)
    write_catalogue(output_root, catalog)
    print(
        f"Hierarchy: {catalog.n_nodes:,} nodes, {catalog.n_leaves:,} leaves, "
        f"{catalog.incidence.nnz:,} memberships",
        flush=True,
    )

    print("Estimating fixed MinT residual weights...", flush=True)
    daily_var, daily_weight, period_var, period_weight = estimate_mint_weights(
        member2_root, catalog
    )
    write_mint_weights(
        output_root, catalog, daily_var, daily_weight, period_var, period_weight
    )

    all_daily_metrics: list[dict[str, object]] = []
    all_period_metrics: list[dict[str, object]] = []
    all_coherence: list[dict[str, object]] = []
    all_cg: list[dict[str, object]] = []
    all_constraint: list[dict[str, object]] = []
    all_samples: list[dict[str, object]] = []
    runtime_rows: list[dict[str, object]] = []
    fallback_path = output_root / "诊断结果" / "top_down_share_fallbacks.csv"
    if fallback_path.exists():
        fallback_path.unlink()

    for split in args.splits:
        origin = SPLIT_SPECS[split][2]
        print(f"[{split}] loading Member 2 forecasts at d_{origin}...", flush=True)
        base, base_q, actual = load_daily_base(member2_root, split, catalog)
        period_q, taus = load_period_quantiles(member2_root, split, catalog)
        weekday_sums, all_sum = historical_leaf_observations(
            member2_root, split, catalog
        )
        shares, fallback = top_down_shares(weekday_sums, all_sum, origin)
        write_top_down_history(output_root, split, catalog, shares, fallback)
        actual_sum = None if actual is None else actual.sum(axis=1)

        for scheme in args.schemes:
            print(f"[{split}] reconciling {scheme}...", flush=True)
            scheme_started = time.time()
            point, quantiles, mint_cg = arrays_for_scheme(
                scheme,
                catalog,
                base,
                base_q,
                shares,
                daily_weight,
                args.cg_tolerance,
                args.cg_max_iter,
            )
            generate_seconds = time.time() - scheme_started
            print(f"[{split}] writing {scheme}...", flush=True)
            write_started = time.time()
            daily_rows = write_daily_predictions(
                output_root,
                split,
                scheme,
                origin,
                catalog,
                point,
                quantiles,
                actual,
            )

            period_point = point.sum(axis=2)
            if scheme in {"unreconciled", "bottom_up"}:
                period_reconciled_q = period_q.copy()
            elif scheme == "top_down":
                cumulative_share = shares.mean(axis=0)
                leaves = period_q[0][None, :] * cumulative_share[:, None]
                period_reconciled_q = np.asarray(catalog.incidence @ leaves)
            else:
                lgbm = MODEL_INDEX["lightgbm"]
                period_reconciled_q, diagnostics = mint_reconcile(
                    catalog,
                    period_weight[lgbm],
                    period_q,
                    quantile_shape=(1, len(taus)),
                    tolerance=args.cg_tolerance,
                    max_iter=args.cg_max_iter,
                )
                mint_cg.append(
                    {
                        "block": "period_quantile",
                        "model": "lightgbm",
                        **diagnostics,
                    }
                )
            period_rows = write_period_predictions(
                output_root,
                split,
                scheme,
                origin,
                catalog,
                period_point,
                period_reconciled_q,
                taus,
                actual_sum,
            )
            write_seconds = time.time() - write_started
            runtime_rows.append(
                {
                    "split": split,
                    "scheme": scheme,
                    "node_count": catalog.n_nodes,
                    "leaf_count": catalog.n_leaves,
                    "horizons": 28,
                    "daily_rows_written": daily_rows,
                    "period_rows_written": period_rows,
                    "generate_seconds": generate_seconds,
                    "write_seconds": write_seconds,
                }
            )

            metric_rows, coherence_rows = scheme_rows(
                scheme,
                split,
                point,
                quantiles,
                actual,
                catalog,
                DAILY_QUANTILES,
            )
            all_daily_metrics.extend(metric_rows)
            all_coherence.extend(coherence_rows)
            all_period_metrics.extend(
                period_scheme_rows(
                    scheme,
                    split,
                    period_point,
                    period_reconciled_q,
                    actual_sum,
                    catalog,
                    taus,
                )
            )
            for row in mint_cg:
                all_cg.append({"split": split, "scheme": scheme, **row})
            for model in MODELS:
                mi = MODEL_INDEX[model]
                daily_q_ok = bool(
                    np.all(np.diff(quantiles[:, :, :], axis=2) >= 0)
                )
                period_q_ok = bool(np.all(np.diff(period_reconciled_q, axis=1) >= 0))
                all_constraint.append(
                    {
                        "split": split,
                        "scheme": scheme,
                        "model": model,
                        "daily_negative_point": int((point[mi] < -1e-12).sum()),
                        "period_negative_point": int((period_point[mi] < -1e-12).sum()),
                        "daily_quantiles_monotone": (
                            daily_q_ok if model == "lightgbm" else None
                        ),
                        "period_quantiles_monotone": (
                            period_q_ok if model == "lightgbm" else None
                        ),
                        "daily_max_additivity_error": coherence_diagnostics(
                            catalog, point[mi]
                        )["max_abs_additivity_error"],
                    }
                )
            all_samples.extend(
                representative_rows(
                    split,
                    scheme,
                    origin + 1,
                    catalog,
                    point,
                    quantiles,
                    actual,
                )
            )

    pd.DataFrame(all_daily_metrics).to_csv(
        output_root / "诊断结果" / "daily_forecast_metrics.csv",
        index=False,
        encoding="utf-8-sig",
    )
    pd.DataFrame(all_period_metrics).to_csv(
        output_root / "诊断结果" / "period_forecast_metrics.csv",
        index=False,
        encoding="utf-8-sig",
    )
    pd.DataFrame(all_coherence).to_csv(
        output_root / "诊断结果" / "coherence_diagnostics.csv",
        index=False,
        encoding="utf-8-sig",
    )
    pd.DataFrame(all_constraint).to_csv(
        output_root / "诊断结果" / "constraint_checks.csv",
        index=False,
        encoding="utf-8-sig",
    )
    pd.DataFrame(all_cg).to_csv(
        output_root / "诊断结果" / "mint_cg_diagnostics.csv",
        index=False,
        encoding="utf-8-sig",
    )
    pd.DataFrame(runtime_rows).to_csv(
        output_root / "诊断结果" / "runtime_complexity.csv",
        index=False,
        encoding="utf-8-sig",
    )
    pd.DataFrame(all_samples).to_csv(
        output_root / "诊断结果" / "representative_samples.csv",
        index=False,
        encoding="utf-8-sig",
    )

    context = {
        "schema_version": "m5-module03-v1",
        "member2_handoff": "external path supplied by CLI; omitted from public metadata",
        "input_scope": "M5 full Member 2 handoff",
        "splits": args.splits,
        "schemes": args.schemes,
        "models": list(MODELS),
        "daily_quantiles": list(DAILY_QUANTILES),
        "period_taus": taus.tolist(),
        "hierarchy": {
            "nodes": catalog.n_nodes,
            "leaves": catalog.n_leaves,
            "levels": 12,
            "memberships": int(catalog.incidence.nnz),
        },
        "mint": {
            "projection": "S (S' W^-1 S)^-1 S' W^-1",
            "weight": "diagonal inverse residual variance",
            "residual_source": [
                f"final_calibration/calibration_{origin}" for origin in (1745, 1773, 1801)
            ],
            "shrinkage_to_pooled_variance": 0.20,
            "variance_floor_ratio_to_pooled": 1e-8,
            "solver": "column-wise preconditioned conjugate gradient",
            "cg_tolerance": args.cg_tolerance,
            "cg_max_iter": args.cg_max_iter,
            "nonnegative_repair": (
                "active-set WLS: fix violating leaves at their lower bound, re-solve free leaves, "
                "then re-aggregate"
            ),
            "quantile_repair": "monotone cumulative maximum across tau at leaves, then re-aggregate",
        },
        "top_down": {
            "lookback_days": 224,
            "allocation": "same-weekday historical leaf share; fallback to all-day share; then equal leaf share",
        },
        "randomness": "none; deterministic hierarchy construction and numerical algorithms",
        "environment": {
            "python": platform.python_version(),
            "system": platform.system().lower(),
            "machine": platform.machine().lower(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "pyarrow": pyarrow.__version__,
            "scipy": scipy.__version__,
        },
        "elapsed_seconds_before_checksums": time.time() - started,
    }
    json_write(output_root / "run_context.json", context)
    print("Writing checksums...", flush=True)
    write_checksums(output_root)
    print(f"Completed in {time.time() - started:.1f}s: {output_root}", flush=True)


if __name__ == "__main__":
    main()
