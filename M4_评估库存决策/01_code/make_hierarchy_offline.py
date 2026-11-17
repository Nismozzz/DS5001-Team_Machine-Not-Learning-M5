r"""M4 离线补充：层级调和可用性 + 下单粒度（单品 vs 门店总量）
不重跑主流水线，只产出 1 份可引用的对照。
输出目录：outputs\_supplement_hierarchy_recon\
"""
import os, sys, json, time
sys.stdout.reconfigure(encoding="utf-8")
from pathlib import Path
import numpy as np, pandas as pd


def _find_root(start: Path) -> Path:
    for p in [start, *start.parents]:
        if (p / "data").is_dir() or (p / "M5").is_dir():
            return p
    return start


BASE = Path(os.environ.get("M5_ROOT") or _find_root(Path(__file__).resolve().parent))
M2 = BASE / "data" / "processed_final" / "member2"
TG = BASE / "data" / "processed" / "final" / "targets"
M3 = Path(os.environ.get("M3_HANDOFF") or BASE / "Module3_Full_Handoff_Module4")
if not M3.is_dir():
    sys.exit(f"[error] 找不到模块 03 交接包：{M3}\n"
             f"请设置环境变量 M3_HANDOFF 指向该目录后重试")
OUT = BASE / "outputs" / "_supplement_hierarchy_recon"
OUT.mkdir(parents=True, exist_ok=True)

SPLITS = {"validation_1": 1829, "validation_2": 1857, "validation_3": 1885, "test": 1913}
MODELS = ["lightgbm", "tsb", "ets", "seasonal_naive"]
RATIOS = {"1:1": (1.0, 1.0), "2:1": (2.0, 1.0), "5:1": (5.0, 1.0), "9:1": (9.0, 1.0)}
LEVELS = {1: "total", 2: "state", 3: "store"}

t0 = time.time()

# ---------- 1. 真值：每个窗口每个 SKU 的 28 天累计需求 ----------
truth_win = {}
for split, origin in SPLITS.items():
    fo = pd.read_parquet(M2 / split / "period.parquet",
                         columns=["node_id", "tau", "forecast", "model", "period_start_d", "period_end_d"])
    fo = fo[fo["model"] == "lightgbm"]
    ps, pe = int(fo["period_start_d"].iloc[0]), int(fo["period_end_d"].iloc[0])
    tr = pd.read_parquet(TG / f"origin_{origin}.parquet")
    tr = tr[(tr["target_d"] >= ps) & (tr["target_d"] <= pe)]
    y = tr.groupby("series_id")["units"].sum()
    fo = fo.copy()
    fo["series_id"] = fo["node_id"].astype(str).map(lambda s: "_".join(json.loads(s.split(":", 1)[1])))
    truth_win[split] = y
    print(f"{split}: 窗口 {ps}-{pe}，真值 {len(y):,} 序列，Σ需求 {y.sum():,.0f}", flush=True)

TAUS = sorted(pd.read_parquet(M2 / "test" / "period.parquet", columns=["tau"])["tau"].unique().tolist())
print(f"τ 共 {len(TAUS)} 个", flush=True)

# ---------- 2. 序列索引 & 窗口矩阵 ----------
series_all = sorted(truth_win["test"].index)
idx = pd.Index(series_all, name="series_id")
DEM = np.zeros((len(idx), len(SPLITS)))
for k, s in enumerate(SPLITS):
    DEM[:, k] = truth_win[s].reindex(idx).fillna(0.0).to_numpy(float)
STORE_OF = pd.Series(idx.str.split("_").str[-2:].str.join("_").to_numpy(), index=idx)

# ---------- 3. 高层分位面板（M3 交接包）与覆盖率校准 ----------
rows_cal = []
for split in SPLITS:
    y = truth_win[split]
    for lv, lname in LEVELS.items():
        q = pd.read_parquet(M3 / "预测结果" / split / "unreconciled" / "period_quantile" / f"L{lv:02d}.parquet",
                            columns=["node_id", "tau", "forecast"])
        q["key"] = q["node_id"].astype(str).map(lambda s: "_".join(json.loads(s.split(":", 1)[1])) if s.split(":", 1)[1] != "[]" else "__TOTAL__")
        # 真值聚合
        if lv == 1:
            truth_agg = pd.Series({"__TOTAL__": y.sum()})
        elif lv == 2:
            truth_agg = y.groupby(y.index.str.split("_").str[-2]).sum()
        else:
            truth_agg = y.groupby(y.index.str.split("_").str[-2:].str.join("_")).sum()
        for tau in TAUS:
            d = q[q["tau"] == tau].set_index("key")["forecast"]
            j = pd.concat([d.rename("q"), truth_agg.rename("y")], axis=1).dropna()
            rows_cal.append(dict(split=split, level=lname, n_nodes=len(j), tau=round(float(tau), 6),
                                 covered=int((j["q"] >= j["y"]).sum()),
                                 coverage=float((j["q"] >= j["y"]).mean()),
                                 q_sum=float(j["q"].sum()), y_sum=float(j["y"].sum()),
                                 ratio=float(j["q"].sum() / j["y"].sum())))
CAL = pd.DataFrame(rows_cal)
CAL.to_csv(OUT / "high_level_quantile_calibration.csv", index=False, encoding="utf-8-sig")
print(f"[ok] 高层分位覆盖率表 {len(CAL)} 行", flush=True)

# ---------- 4. 两种下单口径的仿真 ----------
def simulate(target, demand, cu, co):
    """target/demand: (n, cycles)；与 04_evaluate.py 的 simulate_group 完全同口径"""
    n, c = demand.shape
    on_hand = np.zeros(n)
    hold = np.zeros(n); short = np.zeros(n)
    served = np.zeros(n); dm = np.zeros(n)
    for k in range(c):
        order = np.maximum(0.0, target[:, k] - on_hand)
        on_hand += order
        s = np.minimum(on_hand, demand[:, k])
        on_hand -= s
        served += s
        dm += demand[:, k]
        short += cu * (demand[:, k] - s)
        hold += co * on_hand
    return dict(hold=hold, short=short, served=served, demand=dm, end_inv=on_hand)


def sku_quantiles(split, model):
    """返回 (n_series × n_tau) 的 S 矩阵（未调和，即 M2 原始分位）"""
    fo = pd.read_parquet(M2 / split / "period.parquet", columns=["node_id", "tau", "forecast", "model"])
    fo = fo[fo["model"] == model]
    fo = fo.copy()
    fo["series_id"] = fo["node_id"].astype(str).map(lambda s: "_".join(json.loads(s.split(":", 1)[1])))
    m = fo.pivot_table(index="series_id", columns="tau", values="forecast")
    return m.reindex(index=idx).reindex(columns=TAUS)


def store_quantiles(split, tau_list):
    """门店层 28 天累计分位（M3 包 unreconciled 面板）"""
    q = pd.read_parquet(M3 / "预测结果" / split / "unreconciled" / "period_quantile" / "L03.parquet",
                        columns=["node_id", "tau", "forecast"])
    q = q.copy()
    q["store"] = q["node_id"].astype(str).map(lambda s: json.loads(s.split(":", 1)[1])[0])
    q = q[q["tau"].isin(tau_list)]
    return q.pivot_table(index="store", columns="tau", values="forecast")


res = []
for model in ["lightgbm"]:            # 只有 lightgbm 在两侧都有 28 天累计分位（M2 period.parquet 只含该模型）
    # 每个模型只读一次 4 个窗口的分位矩阵
    Q = {split: sku_quantiles(split, model) for split in SPLITS}
    SQ = {split: store_quantiles(split, TAUS) for split in SPLITS}
    BASE = {split: Q[split][0.5].fillna(0.0) for split in SPLITS}
    for ratio_name, (cu, co) in RATIOS.items():
        for tau in TAUS:
            # ---- 口径 A：按单品下单 ----
            S_A = np.column_stack([Q[split][tau].fillna(0.0).to_numpy(float) for split in SPLITS])
            rA = simulate(S_A, DEM, cu, co)
            res.append(dict(split="4窗口合计", model=model, ratio=ratio_name, tau=round(float(tau), 6),
                            mode="A_单品下单",
                            cost_per_unit_demand=(rA["hold"].sum() + rA["short"].sum()) / rA["demand"].sum(),
                            fill_rate=rA["served"].sum() / rA["demand"].sum(),
                            order_units=float(S_A.sum()), end_inventory_units=float(rA["end_inv"].sum())))

            # ---- 口径 B：门店总量下单（总量按 SKU 点预测占比分摊到单品） ----
            cols = []
            for split in SPLITS:
                base = BASE[split]
                store_q = SQ[split][tau]
                tot = base.groupby(STORE_OF).sum().reindex(store_q.index).fillna(0.0)
                share = base / STORE_OF.map(tot).replace(0.0, np.nan)
                alloc = (STORE_OF.map(store_q).fillna(0.0) * share.fillna(0.0)).reindex(idx).fillna(0.0)
                cols.append(alloc.to_numpy(float))
            S_B = np.column_stack(cols)
            rB = simulate(S_B, DEM, cu, co)
            res.append(dict(split="4窗口合计", model=model, ratio=ratio_name, tau=round(float(tau), 6),
                            mode="B_门店总量下单",
                            cost_per_unit_demand=(rB["hold"].sum() + rB["short"].sum()) / rB["demand"].sum(),
                            fill_rate=rB["served"].sum() / rB["demand"].sum(),
                            order_units=float(S_B.sum()), end_inventory_units=float(rB["end_inv"].sum())))
        print(f"[{model} {ratio_name}] 23 个 τ 完成 {time.time()-t0:.0f}s", flush=True)

R = pd.DataFrame(res)
R.to_csv(OUT / "order_granularity_compare.csv", index=False, encoding="utf-8-sig")
print(f"[ok] 下单粒度对照 {len(R)} 行", flush=True)

# ---------- 5. 与主结果交叉校验（A 口径 5:1 τ=0.95 应等于 headline） ----------
head = R[(R["mode"] == "A_单品下单") & (R["ratio"] == "5:1") & (np.isclose(R["tau"], 0.95))]
print("\nA 口径 5:1 τ=0.95（应与 headline 一致）：")
print(head[["model", "cost_per_unit_demand", "fill_rate"]].to_string(index=False))

hub = R[(R["ratio"] == "5:1") & (np.isclose(R["tau"], 0.95))].pivot_table(
    index="model", columns="mode", values=["cost_per_unit_demand", "fill_rate", "order_units", "end_inventory_units"])
print("\n5:1 τ=0.95 两口径对照：")
print(hub.round(4).to_string())

R.to_csv(OUT / "order_granularity_compare.csv", index=False, encoding="utf-8-sig")
print(f"\n总耗时 {time.time()-t0:.0f}s → {OUT}")

# ---------- 6. 分位不可加的直接证据 ----------
q1 = pd.read_parquet(M3 / "预测结果" / "test" / "bottom_up" / "period_quantile" / "L01.parquet", columns=["tau", "forecast"])
q12 = pd.read_parquet(M3 / "预测结果" / "test" / "bottom_up" / "period_quantile" / "L12.parquet", columns=["tau", "forecast"])
root = q1.groupby("tau")["forecast"].sum()
leaf = q12.groupby("tau")["forecast"].sum()
rel = (root - leaf).abs() / leaf
add_tbl = pd.DataFrame({"tau": rel.index.round(6), "level1_total": root.values, "leaf_sum": leaf.values,
                        "rel_diff": rel.values})
add_tbl.to_csv(OUT / "quantile_additivity_gap.csv", index=False, encoding="utf-8-sig")
print(f"\n分位不可加：τ=0.5 差 {rel.loc[0.5]*100:.2f}% / τ=0.95 差 {rel.loc[0.95]*100:.2f}% / 最大 {rel.max()*100:.2f}%")

# ---------- 7. 图：成本随 τ（5:1，两口径） ----------
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

fig_dir = OUT
sub = R[(R["ratio"] == "5:1") & (R["model"] == "lightgbm")]
fig, ax = plt.subplots(figsize=(7.2, 4.2), dpi=150)
for mode, label, style in [("A_单品下单", "SKU-level order", dict(color="#1f4e79", marker="o", lw=1.6)),
                           ("B_门店总量下单", "Store-total order", dict(color="#b5651d", marker="s", lw=1.6))]:
    d = sub[sub["mode"] == mode].sort_values("tau")
    ax.plot(d["tau"], d["cost_per_unit_demand"], label=label, ms=4, **style)
ax.set_xlabel("target service level τ")
ax.set_ylabel("cost per unit demand (5:1)")
ax.set_title("Order granularity: SKU-level vs store-total (lightgbm, 4 windows)")
ax.grid(alpha=.3)
ax.legend(frameon=False)
fig.tight_layout()
fig.savefig(fig_dir / "fig_order_granularity.png")
plt.close(fig)

# ---------- 8. summary.json ----------
head_row = R[(R["ratio"] == "5:1") & (np.isclose(R["tau"], 0.95))]
cal = CAL[(CAL["level"] == "store") & (np.isclose(CAL["tau"], 0.95))]
summary = {
    "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    "scope": "CA_1-CA_4 / TX_1-TX_3 / WI_1-WI_3，30,490 SKU，28 天累计、4 个评估窗口",
    "model_for_granularity": "lightgbm",
    "tau_grid": len(TAUS),
    "quantile_additivity": {
        "tau0.5_rel_diff_pct": round(float(rel.loc[0.5]) * 100, 2),
        "tau0.95_rel_diff_pct": round(float(rel.loc[0.95]) * 100, 2),
        "max_rel_diff_pct": round(float(rel.max()) * 100, 2),
        "note": "M3 bottom_up 分位面板中，顶层节点值对全叶分位求和的相对差；分位不可加意味着高层分位不能由叶层分位相加得到",
    },
    "store_level_quantile_calibration_tau0.95": {
        "mean_coverage": round(float(cal["coverage"].mean()), 4),
        "n_nodes": int(cal["n_nodes"].iloc[0]) if len(cal) else 0,
        "note": "门店层 28 天累计分位对门店真实需求的覆盖率（10 店 × 4 窗口）",
    },
    "granularity_5to1_tau0.95": {
        r["mode"]: {"cost_per_unit_demand": round(float(r["cost_per_unit_demand"]), 4),
                    "fill_rate": round(float(r["fill_rate"]), 4),
                    "order_units": round(float(r["order_units"]), 1),
                    "end_inventory_units": round(float(r["end_inventory_units"]), 1)}
        for _, r in head_row.iterrows()
    },
    "limitations": [
        "B 口径（门店总量下单）的总量按 SKU 点预测占比分摊到单品，未使用门店内更优的分摊规则",
        "只有 lightgbm 在 M2 与 M3 两侧都有 28 天累计分位；三基线的累计分位由 M4 用“点预测 + 误差经验分位”自建，未纳入本对照",
        "分位覆盖率为逐节点统计（门店 10 个 × 4 窗口），样本量小，尾部 τ 仅供参考",
    ],
}
(OUT / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
print("\n=== summary.json ===")
print(json.dumps(summary, ensure_ascii=False, indent=2)[:1600])
