r"""离线补算：28 天累计层的 23 点 pinball（四模型同网格）

背景
----
`04_evaluate.py` 的 pinball 块走的是**日级**口径：
  - LightGBM 用 M2 原生日分位，只有 0.5/0.9/0.95/0.99 四点；
  - ETS / SeasonalNaive / TSB 用「点预测 × 累计层尺度比」的日级转换分位，23 点。
两个模型的 τ 网格不同，`forecast_metrics.csv` 里的 pinball 不可横比；
而报告第 26 / 86 段把「23 点分位损失」写成了 28 天累计层，与实现不符。

本脚本只补算一件事：在**28 天累计层**上，用与 `04_evaluate.py` 完全相同的
订单目标 S（CYCLE_TARGET），对四个模型算同一 τ 网格的 pinball。
不重跑仿真、不重算成本与覆盖率，只重算这一组指标。

S 的来源（与流水线一致）
------------------------
  lightgbm        ：M2 `period.parquet` 的 23 点原生累计分位（max(forecast,0)）
  ets / snaive / tsb：C-4 误差加法 max(S_point + Q_{τ,m}, 0)，Q 取更早窗口
                     「28 天累计误差（真值 − 点预测累计）」按模型池化的 τ 分位，
                     再对 τ 做 cummax 保证单调。与 04_evaluate.py 的
                     point_to_cycle / prefix_error_quantiles 同式。

自检
----
脚本末尾用 `coverage_report.csv` 交叉验证：重算出的 sku 层实测覆盖率与
mean_S 必须与流水线产物逐位一致，否则说明 S 没复现出来，结果不可用。

用法
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


# ---------------------------------------------------------------- 1) 点预测
print("[1/6] 读 M2 日级点预测（三个基线）")
frames = []
for d in EVAL_DIRS:
    p = M2 / d / "daily.parquet"
    df = pd.read_parquet(p, columns=["series_id", "origin_d", "target_d", "model", "yhat"])
    frames.append(df[df["origin_d"].isin(ORIGINS) & df["model"].isin(POINT_MODELS)])
PT = pd.concat(frames, ignore_index=True)
PT.rename(columns={"yhat": "y_hat"}, inplace=True)
print(f"  行数 {len(PT):,} | 窗口 {sorted(PT.origin_d.unique())} | 模型 {sorted(PT.model.unique())}")

# 28 天累计点预测
S_POINT = (PT.groupby(["series_id", "origin_d", "model"], as_index=False)
             .agg(S_point=("y_hat", "sum")))
print(f"  累计点预测 {len(S_POINT):,} 行")

# ---------------------------------------------------------------- 2) 残差
print("[2/6] 读 M2 残差（4 评估窗口 + 3 标定窗口）")
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
print(f"  行数 {len(RESID):,} | 窗口 {sorted(RESID.origin_d.unique())}")

CUM = (RESID.groupby(["series_id", "origin_d", "model"], as_index=False)
             .agg(e=("residual", "sum")))

# ---------------------------------------------------------------- 3) 基线累计分位
print("[3/6] C-4 误差加法：基线 28 天累计分位")
resid_origins = sorted(int(x) for x in RESID["origin_d"].unique())
rows = []
for o in ORIGINS:
    prev = [w for w in resid_origins if w < o]
    if not prev:
        raise RuntimeError(f"窗口 {o} 没有更早窗口可用")
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
print(f"  行数 {len(BASE):,} | 模型 {sorted(BASE.model.unique())}")

# ---------------------------------------------------------------- 4) LightGBM 原生
print("[4/6] 直读 M2 period.parquet（lightgbm 23 点原生累计分位）")
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
print(f"  行数 {len(LGBM):,} | τ {LGBM.tau.nunique()} 点")

CYCLE_S = pd.concat([BASE, LGBM], ignore_index=True)

# ---------------------------------------------------------------- 5) 真值
print("[5/6] 读 28 天累计需求真值")
t_frames = []
for o in ORIGINS:
    t = pd.read_parquet(TARGETS / f"origin_{o}.parquet", columns=["series_id", "origin_d",
                                                                 "target_d", "units"])
    t_frames.append(t[t["origin_d"] == o])
TRUTH = pd.concat(t_frames, ignore_index=True)
Y28 = (TRUTH.groupby(["series_id", "origin_d"], as_index=False)
            .agg(units=("units", "sum")))
print(f"  累计需求 {len(Y28):,} 行 | 均值 {Y28['units'].mean():.4f}")

# ---------------------------------------------------------------- 6) pinball
print("[6/6] 计算累计层 pinball（sku / store / total）")
HIER = pd.read_parquet(HIER_PATH, columns=["series_id", "level", "node_id"])
LEVEL_NAME = {1: "total", 3: "store"}

out_rows = []

# 6a. sku 层
m = CYCLE_S.merge(Y28, on=["series_id", "origin_d"], how="inner", validate="many_to_one")
for (model, origin, tau), g in m.groupby(["model", "origin_d", "tau"], sort=True):
    out_rows.append(dict(model=model, level="sku", origin_d=int(origin), metric="pinball",
                         tau=float(tau), value=pinball(g["units"], g["S"], float(tau)),
                         n_series=int(g["series_id"].nunique()), n_obs=len(g)))
print(f"  sku 层 {len(out_rows):,} 行")

# 6b. 聚合层
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
    print(f"  {lname} 层 {len(out_rows) - n0:,} 行")
    del s_agg, d_agg

METRICS = pd.DataFrame(out_rows)

# ---------------------------------------------------------------- 自检
print("\n自检：与流水线 coverage_report.csv 交叉验证（同一份 S 的实测覆盖率）")
cov = (m.assign(hit=(m["S"] >= m["units"]).astype(float))
         .groupby(["model", "origin_d", "tau"], as_index=False)
         .agg(empirical=("hit", "mean"), mean_S=("S", "mean")))
ref = pd.read_csv(COVERAGE)
chk = cov.merge(ref, on=["model", "origin_d", "tau"], suffixes=("_calc", "_pipe"))
chk["d_emp"] = (chk["empirical_calc"] - chk["empirical_pipe"]).abs()
chk["d_S"] = (chk["mean_S_calc"] - chk["mean_S_pipe"]).abs()
chk["d_Y"] = (chk["mean_S_calc"] * 0 + 0)  # 占位，真值均值另行核对
print(f"  比对行数 {len(chk):,}（应为 4 模型 × 4 窗口 × 23 τ = 368）")
print(f"  |Δempirical| 最大 = {chk['d_emp'].max():.3e}")
print(f"  |Δmean_S|    最大 = {chk['d_S'].max():.3e}")
ymean = (m.groupby(["origin_d"], as_index=False).agg(mean_Y=("units", "mean")))
ymean_ref = ref.groupby("origin_d", as_index=False).agg(mean_Y=("mean_Y", "mean"))
dd = (ymean.merge(ymean_ref, on="origin_d").assign(d=lambda x: (x["mean_Y_x"] - x["mean_Y_y"]).abs()))
print(f"  |Δmean_Y|    最大 = {dd['d'].max():.3e}")
assert len(chk) == 368, "比对行数不符，S 未复现"
assert chk["d_emp"].max() < 1e-6, "实测覆盖率与流水线不一致，S 未复现，结果不可用"
assert chk["d_S"].max() < 1e-6, "mean_S 与流水线不一致，S 未复现，结果不可用"
assert dd["d"].max() < 1e-4, "累计需求真值与流水线不一致"  # CSV 只存 6 位小数，容差取 1e-4
print("  ✓ S 与真值均已逐位复现")

OUT_DIR.mkdir(parents=True, exist_ok=True)
out_csv = OUT_DIR / "forecast_metrics_period_pinball.csv"
METRICS.to_csv(out_csv, index=False, encoding="utf-8-sig")
print(f"\n写出 {out_csv} | {len(METRICS):,} 行 | {out_csv.stat().st_size:,} B")

print("\nτ=0.95 累计层 pinball（每件需求；sku 层，四窗口）：")
h = METRICS[(METRICS["level"] == "sku") & np.isclose(METRICS["tau"], 0.95)]
print(h.pivot_table(index="model", columns="origin_d", values="value").round(4).to_string())
print("\n全 τ 网格上四模型均值（sku 层，四窗口平均）：")
allm = (METRICS[METRICS["level"] == "sku"]
        .groupby("model", as_index=False).agg(pinball_mean=("value", "mean"),
                                              tau_pts=("tau", "nunique")))
print(allm.round(6).to_string(index=False))
print("\nτ=0.95 各层（四窗口平均）：")
lay = (METRICS[np.isclose(METRICS["tau"], 0.95)]
       .groupby(["level", "model"], as_index=False).agg(v=("value", "mean")))
print(lay.pivot_table(index="level", columns="model", values="v").round(2).to_string())

with open(OUT_DIR / "summary.json", "w", encoding="utf-8") as f:
    json.dump({
        "script": "tools/make_period_pinball.py",
        "scope": "final 30,490 SKU / 4 窗口 / 28 天累计层",
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
        "note": "S 与 04_evaluate.py 的 CYCLE_TARGET 同式重建，并与 coverage_report.csv 逐位对齐；"
                "仅补算累计层 pinball，未改动成本、覆盖率、决策图与仿真参数。",
    }, f, ensure_ascii=False, indent=2)
print(f"写出 {OUT_DIR / 'summary.json'}")
