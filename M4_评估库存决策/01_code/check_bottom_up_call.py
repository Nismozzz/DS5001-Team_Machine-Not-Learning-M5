"""Verify the real call path of M3's reconcile_quantile_block(scheme='bottom_up').

Builds a Catalog from the M3 handoff's hierarchy_map_final.parquet (no coverage.json
inside the package), loads Member 2 test-split leaf quantiles, calls M3's own
function, and compares the reconciled panel against a manual leaf-sum.
"""
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from scipy.sparse import csc_matrix

sys.stdout.reconfigure(encoding="utf-8")


def _find_root(start: Path) -> Path:
    for p in [start, *start.parents]:
        if (p / "data").is_dir() or (p / "M5").is_dir():
            return p
    return start


ROOT = Path(os.environ.get("M5_ROOT") or _find_root(Path(__file__).resolve().parent))
M3 = Path(os.environ.get("M3_HANDOFF") or ROOT / "Module3_Full_Handoff_Module4")
if not M3.is_dir():
    sys.exit(f"[error] 找不到模块 03 交接包：{M3}\n"
             f"请设置环境变量 M3_HANDOFF 指向该目录后重试")
sys.path.insert(0, str(M3 / "code"))
import reconciliation as R  # noqa: E402

print("scipy ok, M3 module loaded:", hasattr(R, "reconcile_quantile_block"))

# ---------- 1. build Catalog from hierarchy_map_final.parquet ----------
HMAP = M3 / "hierarchy_map_final.parquet"
if not HMAP.exists():
    HMAP = M3 / "artifacts" / "hierarchy_map_final.parquet"
print("hmap path:", HMAP, HMAP.exists())
hmap = pq.read_table(HMAP, columns=["level", "node_id", "series_id"]).to_pandas()
hmap["node_id"] = hmap["node_id"].astype(str)
hmap["series_id"] = hmap["series_id"].astype(str)
print("hmap rows:", len(hmap), "levels:", sorted(hmap["level"].unique().tolist()))

node_ids, node_levels, level_nodes = [], [], {}
for level in range(1, 13):
    vals = sorted(hmap.loc[hmap["level"] == level, "node_id"].unique().tolist())
    start = len(node_ids)
    node_ids.extend(vals)
    node_levels.extend([level] * len(vals))
    level_nodes[level] = np.arange(start, start + len(vals), dtype=np.int32)
node_index = {v: i for i, v in enumerate(node_ids)}
print("nodes:", len(node_ids), "leaves:", len(level_nodes[12]))

leaf_ids = [node_ids[i] for i in level_nodes[12]]
leaf_series = [f"{json.loads(v.split(':', 1)[1])[0]}_{json.loads(v.split(':', 1)[1])[1]}" for v in leaf_ids]
leaf_meta = pd.DataFrame([R.leaf_metadata(s) for s in leaf_series])
leaf_meta.insert(0, "series_id", leaf_series)

rows = hmap["node_id"].map(node_index).to_numpy(dtype=np.int32)
cols = hmap["series_id"].map({s: i for i, s in enumerate(leaf_series)})
if pd.isna(cols).any():
    raise SystemExit("leaf series mapping failed")
cols = cols.to_numpy(dtype=np.int32)
incidence = csc_matrix((np.ones(len(rows), dtype=np.float64), (rows, cols)),
                       shape=(len(node_ids), len(leaf_series)))
leaf_nodes = level_nodes[12].astype(np.int32)

catalog = R.Catalog(
    root=M3, node_ids=node_ids, node_level=np.asarray(node_levels, dtype=np.int8),
    node_index=node_index, level_nodes=level_nodes, leaf_series=np.asarray(leaf_series, dtype=object),
    leaf_meta=leaf_meta, leaf_nodes=leaf_nodes, incidence=incidence,
)
print("catalog built:", catalog.n_nodes, catalog.n_leaves, "nnz:", incidence.nnz)

# ---------- 2. load M2 test leaf quantiles ----------
per = pq.read_table(ROOT / "data" / "processed_final" / "member2" / "test" / "period.parquet",
                    columns=["node_id", "tau", "forecast", "model"],
                    filters=[("model", "=", "lightgbm")]).to_pandas()
per["node_id"] = per["node_id"].astype(str)
taus = np.array(sorted(per["tau"].unique()))
print("period rows:", len(per), "taus:", len(taus))

pivot = per.pivot_table(index="node_id", columns="tau", values="forecast", aggfunc="mean")
pivot = pivot.reindex(taus, axis=1)
base = np.zeros((catalog.n_nodes, len(taus)), dtype=np.float64)
leaf_pos = pd.Series(np.arange(len(leaf_series)), index=leaf_series)
sid = pivot.index.to_series().map(
    lambda v: f"{json.loads(v.split(':', 1)[1])[0]}_{json.loads(v.split(':', 1)[1])[1]}"
)
pos = sid.map(leaf_pos)
missing = pos.isna().sum()
print("leaf rows in M2 file:", len(pivot), "unmapped:", int(missing))
base[leaf_nodes[pos.to_numpy(dtype=np.int32)]] = pivot.to_numpy(dtype=np.float64)

# ---------- 3. call M3's function ----------
recon, diag = R.reconcile_quantile_block(catalog, "bottom_up", base)
print("call ok; diag:", diag, "shape:", recon.shape)

manual = R.reconstitute(catalog.incidence, base[catalog.leaf_nodes])
print("max |api - manual sum|:", float(np.abs(recon - manual).max()))
leaf_diff = float(np.abs(recon[catalog.leaf_nodes] - base[catalog.leaf_nodes]).max())
print("leaf rows unchanged (max diff):", leaf_diff)

# ---------- 4. compare with actual demand ----------
tg = pq.read_table(ROOT / "data" / "processed" / "final" / "targets" / "origin_1913.parquet",
                   columns=["series_id", "units"]).to_pandas()
tg["series_id"] = tg["series_id"].astype(str)
actual_leaf = tg.groupby("series_id")["units"].sum().reindex(pd.Index(leaf_series)).fillna(0.0).to_numpy()
lvl12_sums = np.asarray(catalog.incidence[level_nodes[12]] @ actual_leaf).ravel()

root_i = level_nodes[1][0]
tau_pos = {round(float(t), 6): j for j, t in enumerate(taus)}
out = []
for tau in (0.50, 0.90, 0.95):
    j = tau_pos[round(tau, 6)]
    total_q = float(recon[root_i, j])
    out.append({"scope": "L01 total", "tau": tau, "reconciled": total_q,
                "actual": float(actual_leaf.sum()),
                "ratio_to_actual": total_q / float(actual_leaf.sum()),
                "leaf_sum": float(base[leaf_nodes, j].sum())})

for si, s in enumerate(leaf_series):
    if s.startswith("HOBBIES_1_001_CA_1") or s.startswith("FOODS_3_090_CA_1"):
        j50, j95 = tau_pos[0.5], tau_pos[0.95]
        out.append({"scope": s, "tau": 0.50, "reconciled": float(base[leaf_nodes[si], j50]),
                    "actual": float(actual_leaf[si]),
                    "ratio_to_actual": float(base[leaf_nodes[si], j50]) / max(actual_leaf[si], 1e-9),
                    "leaf_sum": float(base[leaf_nodes[si], j50])})
        out.append({"scope": s, "tau": 0.95, "reconciled": float(base[leaf_nodes[si], j95]),
                    "actual": float(actual_leaf[si]),
                    "ratio_to_actual": float(base[leaf_nodes[si], j95]) / max(actual_leaf[si], 1e-9),
                    "leaf_sum": float(base[leaf_nodes[si], j95])})

res = pd.DataFrame(out)
print(res.to_string(index=False, float_format=lambda x: f"{x:,.4f}"))

dest = ROOT / "outputs" / "_supplement_hierarchy_recon" / "bottom_up_api_check.csv"
res.to_csv(dest, index=False, encoding="utf-8-sig")
print("written:", dest, dest.stat().st_size)
