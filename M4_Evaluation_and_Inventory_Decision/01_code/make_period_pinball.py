r"""Offline recomputation: 23-point pinball at the 28-day cumulative level (one grid for the four models)

Background
----
The pinball block in `04_evaluate.py` uses the **daily** convention:
  - LightGBM uses M2's native daily quantiles, only four points: 0.5/0.9/0.95/0.99;
  - ETS / SeasonalNaive / TSB use daily converted quantiles from "point forecast × period-level scale ratio", 23 points.
The τ grids differ between models, so the pinball values in `forecast_metrics.csv` cannot be compared across models;
and paragraphs 26 / 86 of the report describe the "23-point quantile loss" as the 28-day cumulative level, which does not match the implementation.

This script recomputes exactly one thing: at the **28-day cumulative level**, using exactly the same
order target S (CYCLE_TARGET) as `04_evaluate.py`, the pinball of the four models on one τ grid.
It does not rerun the simulation or recompute cost and coverage; only this metric group is recomputed.

Source of S (consistent with the pipeline)
------------------------
  lightgbm          : the 23 native cumulative quantiles of M2 `period.parquet` (max(forecast,0))
  ets / snaive / tsb: C-4 error addition max(S_point + Q_{τ,m}, 0), where Q takes the
                      τ quantile of the "28-day cumulative error (truth − cumulative point forecast)" pooled per model
                      from earlier windows, then cummax over τ to enforce monotonicity. Same formula as
                      point_to_cycle / prefix_error_quantiles in 04_evaluate.py.

Self-check
----
At the end the script cross-validates against `coverage_report.csv`: the recomputed SKU-level realised coverage and
mean_S must match the pipeline artifacts digit for digit, otherwise S was not reproduced and the results are unusable.

Usage
----
    $env:PYTHONPATH=$null
    .\M5\.venv\Scripts\python.exe .\tools\make_period_pinball.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.stdout.reconfigure(encoding="utf-8")

ROOT = Path(__file__).resolve().parents[1]
M2 = ROOT / "data" / "processed_final" / "member2"
TARGETS = ROOT / "data" / "processed" / "final" / "targets"
HIER_PATH = ROOT / "data" / "processed" / "hierarchy_map_final.parquet"
COVERAGE = ROOT / "outputs" / "04_evaluate_final" / "coverage_report.csv"
OUT_DIR = ROOT / "outputs" / "_supplement_period_pinball"

ORIGINS = [1829, 1857, 1885, 1913]
POINT_MODELS = ["ets", "seasonal_naive", "tsb"]
TAU_GRID = [0.50, 2.0 / 3.0, 5.0 / 6.0] + [round(0.80 + 0.01 * i, 2) for i in range(20)]
EVAL_DIRS = ["validation_1", "validation_2", "validation_3", "test"]
CAL_DIRS = ["calibration_1745", "calibration_1773", "calibration_1801"]


def node_ids_to_series(s: pd.Series) -> pd.Series:
    s = s.astype(str).str.replace(r"^L\d+:\[", "", regex=True)
    s = s.str.replace(r"\]$", "", regex=True)
    s = s.str.replace('"', "", regex=False)
    return s.str.replace(",", "_", regex=False)


def pinball(y, yhat, tau: float) -> float:
    y = np.asarray(y, float)
    yhat = np.asarray(yhat, float)
    e = y - yhat
    return float(np.mean(np.maximum(tau * e, (tau - 1.0) * e)))


# ---------------------------------------------------------------- 1) point forecasts
print("[1/6] read M2 daily point forecasts (three baselines)")
frames = []
for d in EVAL_DIRS:
    p = M2 / d / "daily.parquet"
    df = pd.read_parquet(p, columns=["series_id", "origin_d", "target_d", "model", "yhat"])
    frames.append(df[df["origin_d"].isin(ORIGINS) & df["model"].isin(POINT_MODELS)])
PT = pd.concat(frames, ignore_index=True)
PT.rename(columns={"yhat": "y_hat"}, inplace=True)
print(f"  rows {len(PT):,} | windows {sorted(PT.origin_d.unique())} | models {sorted(PT.model.unique())}")

# 28-day cumulative point forecasts
S_POINT = (PT.groupby(["series_id", "origin_d", "model"], as_index=False)
             .agg(S_point=("y_hat", "sum")))
print(f"  cumulative point forecasts {len(S_POINT):,} rows")

# ---------------------------------------------------------------- 2) residuals
print("[2/6] read M2 residuals (4 evaluation windows + 3 calibration windows)")
r_frames = []
for d in EVAL_DIRS + CAL_DIRS:
    p = M2 / d / "residuals.parquet"
    if not p.exists():
        raise FileNotFoundError(p)
    df = pd.read_parquet(p, columns=["node_id", "origin_d", "target_d",
                                     "model", "yhat", "y_true", "residual"])
    df["series_id"] = node_ids_to_series(df["node_id"])
    r_frames.append(df[["series_id", "origin_d", "target_d", "model", "residual"]])
RESID = pd.concat(r_frames, ignore_index=True)
print(f"  rows {len(RESID):,} | windows {sorted(RESID.origin_d.unique())}")

CUM = (RESID.groupby(["series_id", "origin_d", "model"], as_index=False)
             .agg(e=("residual", "sum")))

# ---------------------------------------------------------------- 3) baseline cumulative quantiles
print("[3/6] C-4 error addition: baseline 28-day cumulative quantiles")
resid_origins = sorted(int(x) for x in RESID["origin_d"].unique())
rows = []
for o in ORIGINS:
    prev = [w for w in resid_origins if w < o]
    if not prev:
        raise RuntimeError(f"window {o} has no earlier window available")
    q_rows = []
    for model, g in CUM[CUM["origin_d"].isin(prev)].groupby("model", sort=False):
        e = g["e"].to_numpy(float)
        qs = np.quantile(e, TAU_GRID)
        for t, v in zip(TAU_GRID, qs):
            q_rows.append((model, float(t), float(v)))
    q = pd.DataFrame(q_rows, columns=["model", "tau", "Q"])
    sub = S_POINT[S_POINT["origin_d"] == o].merge(q, on="model", how="left")
    sub["S"] = np.maximum(sub["S_point"].to_numpy(float) + sub["Q"].fillna(0.0).to_numpy(float), 0.0)
    sub["origin_d"] = o
    rows.append(sub[["series_id", "origin_d", "model", "tau", "S"]])
BASE = pd.concat(rows, ignore_index=True)
BASE = BASE.sort_values(["series_id", "origin_d", "model", "tau"], kind="mergesort")
BASE["S"] = BASE.groupby(["series_id", "origin_d", "model"], sort=False)["S"].cummax()
print(f"  rows {len(BASE):,} | models {sorted(BASE.model.unique())}")

# ---------------------------------------------------------------- 4) LightGBM native
print("[4/6] read M2 period.parquet directly (lightgbm 23 native cumulative quantiles)")
p_frames = []
for d in EVAL_DIRS:
    df = pd.read_parquet(M2 / d / "period.parquet",
                         columns=["node_id", "tau", "forecast", "origin_d"])
    df["series_id"] = node_ids_to_series(df["node_id"])
    p_frames.append(df[df["origin_d"].isin(ORIGINS)][["series_id", "origin_d", "tau", "forecast"]])
LGBM = pd.concat(p_frames, ignore_index=True)
LGBM["model"] = "lightgbm"
LGBM["S"] = np.maximum(LGBM["forecast"].to_numpy(float), 0.0)
LGBM = LGBM[["series_id", "origin_d", "model", "tau", "S"]]
print(f"  rows {len(LGBM):,} | τ {LGBM.tau.nunique()} points")

CYCLE_S = pd.concat([BASE, LGBM], ignore_index=True)

# ---------------------------------------------------------------- 5) truth
print("[5/6] read the 28-day cumulative demand truth")
t_frames = []
for o in ORIGINS:
    t = pd.read_parquet(TARGETS / f"origin_{o}.parquet", columns=["series_id", "origin_d",
                                                                 "target_d", "units"])
    t_frames.append(t[t["origin_d"] == o])
TRUTH = pd.concat(t_frames, ignore_index=True)
Y28 = (TRUTH.groupby(["series_id", "origin_d"], as_index=False)
            .agg(units=("units", "sum")))
print(f"  cumulative demand {len(Y28):,} rows | mean {Y28['units'].mean():.4f}")

# ---------------------------------------------------------------- 6) pinball
print("[6/6] compute period-level pinball (sku / store / total)")
HIER = pd.read_parquet(HIER_PATH, columns=["series_id", "level", "node_id"])
LEVEL_NAME = {1: "total", 3: "store"}

out_rows = []

# 6a. sku level
m = CYCLE_S.merge(Y28, on=["series_id", "origin_d"], how="inner", validate="many_to_one")
for (model, origin, tau), g in m.groupby(["model", "origin_d", "tau"], sort=True):
    out_rows.append(dict(model=model, level="sku", origin_d=int(origin), metric="pinball",
                         tau=float(tau), value=pinball(g["units"], g["S"], float(tau)),
                         n_series=int(g["series_id"].nunique()), n_obs=len(g)))
print(f"  sku level {len(out_rows):,} rows")

# 6b. aggregate levels
for lv, lname in LEVEL_NAME.items():
    mp = (HIER[HIER["level"] == lv][["series_id", "node_id"]].drop_duplicates("series_id")
          .set_index("series_id")["node_id"])
    d_agg = (Y28.assign(node_id=Y28["series_id"].map(mp)).dropna(subset=["node_id"])
                .groupby(["node_id", "origin_d"], as_index=False).agg(units=("units", "sum")))
    s_agg = (m.assign(node_id=m["series_id"].map(mp)).dropna(subset=["node_id"])
              .groupby(["node_id", "origin_d", "model", "tau"], as_index=False).agg(S=("S", "sum")))
    s_agg = s_agg.merge(d_agg, on=["node_id", "origin_d"], how="inner")
    n0 = len(out_rows)
    for (model, origin, tau), g in s_agg.groupby(["model", "origin_d", "tau"], sort=True):
        out_rows.append(dict(model=model, level=lname, origin_d=int(origin), metric="pinball",
                             tau=float(tau), value=pinball(g["units"], g["S"], float(tau)),
                             n_series=int(g["node_id"].nunique()), n_obs=len(g)))
    print(f"  {lname} level {len(out_rows) - n0:,} rows")
    del s_agg, d_agg

METRICS = pd.DataFrame(out_rows)

# ---------------------------------------------------------------- self-check
print("\nself-check: cross-validated against the pipeline's coverage_report.csv (realised coverage of the same S)")
cov = (m.assign(hit=(m["S"] >= m["units"]).astype(float))
         .groupby(["model", "origin_d", "tau"], as_index=False)
         .agg(empirical=("hit", "mean"), mean_S=("S", "mean")))
ref = pd.read_csv(COVERAGE)
chk = cov.merge(ref, on=["model", "origin_d", "tau"], suffixes=("_calc", "_pipe"))
chk["d_emp"] = (chk["empirical_calc"] - chk["empirical_pipe"]).abs()
chk["d_S"] = (chk["mean_S_calc"] - chk["mean_S_pipe"]).abs()
chk["d_Y"] = (chk["mean_S_calc"] * 0 + 0)  # placeholder; the mean of the truth is checked separately
print(f"  compared rows {len(chk):,} (should be 4 models × 4 windows × 23 τ = 368)")
print(f"  max |Δempirical| = {chk['d_emp'].max():.3e}")
print(f"  max |Δmean_S|     = {chk['d_S'].max():.3e}")
ymean = (m.groupby(["origin_d"], as_index=False).agg(mean_Y=("units", "mean")))
ymean_ref = ref.groupby("origin_d", as_index=False).agg(mean_Y=("mean_Y", "mean"))
dd = (ymean.merge(ymean_ref, on="origin_d").assign(d=lambda x: (x["mean_Y_x"] - x["mean_Y_y"]).abs()))
print(f"  max |Δmean_Y|     = {dd['d'].max():.3e}")
assert len(chk) == 368, "row count mismatch, S was not reproduced"
assert chk["d_emp"].max() < 1e-6, "realised coverage differs from the pipeline; S was not reproduced, results unusable"
assert chk["d_S"].max() < 1e-6, "mean_S differs from the pipeline; S was not reproduced, results unusable"
assert dd["d"].max() < 1e-4, "cumulative demand truth differs from the pipeline"  # the CSV stores 6 decimals, so the tolerance is 1e-4
print("  ✓ S and the truth are both reproduced digit for digit")

OUT_DIR.mkdir(parents=True, exist_ok=True)
out_csv = OUT_DIR / "forecast_metrics_period_pinball.csv"
METRICS.to_csv(out_csv, index=False, encoding="utf-8-sig")
print(f"\nwrote {out_csv} | {len(METRICS):,} rows | {out_csv.stat().st_size:,} B")

print("\nτ=0.95 period-level pinball (per unit of demand; sku level, four windows):")
h = METRICS[(METRICS["level"] == "sku") & np.isclose(METRICS["tau"], 0.95)]
print(h.pivot_table(index="model", columns="origin_d", values="value").round(4).to_string())
print("\nfour-model mean over the whole τ grid (sku level, four-window average):")
allm = (METRICS[METRICS["level"] == "sku"]
        .groupby("model", as_index=False).agg(pinball_mean=("value", "mean"),
                                              tau_pts=("tau", "nunique")))
print(allm.round(6).to_string(index=False))
print("\nτ=0.95 by level (four-window average):")
lay = (METRICS[np.isclose(METRICS["tau"], 0.95)]
       .groupby(["level", "model"], as_index=False).agg(v=("value", "mean")))
print(lay.pivot_table(index="level", columns="model", values="v").round(2).to_string())

with open(OUT_DIR / "summary.json", "w", encoding="utf-8") as f:
    json.dump({
        "script": "tools/make_period_pinball.py",
        "scope": "final 30,490 SKU / 4 windows / 28-day cumulative level",
        "tau_grid": TAU_GRID,
        "models": sorted(METRICS["model"].unique()),
        "levels": sorted(METRICS["level"].unique()),
        "rows": int(len(METRICS)),
        "selfcheck": {
            "rows_compared": int(len(chk)),
            "max_abs_delta_empirical": float(chk["d_emp"].max()),
            "max_abs_delta_mean_S": float(chk["d_S"].max()),
            "max_abs_delta_mean_Y": float(dd["d"].max()),
        },
        "note": "S is rebuilt with the same formula as CYCLE_TARGET in 04_evaluate.py and aligned with coverage_report.csv digit for digit; "
                "only the period-level pinball is recomputed, with cost, coverage, decision map and simulation parameters untouched.",
    }, f, ensure_ascii=False, indent=2)
print(f"wrote {OUT_DIR / 'summary.json'}")
