#!/usr/bin/env python3
"""Independently recompute artifact counts, hierarchy mass, features and split boundaries."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq


def check(value, message):
    if not value:
        raise AssertionError(message)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("data/processed"))
    args = parser.parse_args()
    out = args.output_dir
    artifact_manifest = json.loads((out / "artifact_manifest.json").read_text())
    for relative, expected in artifact_manifest.items():
        path = out / relative
        digest = hashlib.sha256()
        with path.open("rb") as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(block)
        check(path.stat().st_size == expected["bytes"] and digest.hexdigest() == expected["sha256"], f"Artifact checksum: {relative}")
    manifest = json.loads((out / "split_manifest.json").read_text())
    meta = pd.read_parquet(out / "series.parquet")
    sales = pd.read_parquet(out / "sales_final.parquet")
    dev = pd.read_parquet(out / "sales_development.parquet")
    calendar = pd.read_parquet(out / "calendar.parquet").set_index("d_num")
    check(len(sales) == 30490 and len(dev) == manifest["development"]["size"], "Sales cardinality")
    check(sales.series_id.tolist() == meta.series_id.tolist(), "Sales order")
    check(dev.series_id.tolist() == manifest["development"]["series_ids"], "Sample list")
    check([c for c in sales if c.startswith("d_")] == [f"d_{d}" for d in range(1, 1942)], "1941 dates")
    check(calendar.loc[1, "date"] == "2011-01-29" and calendar.loc[1941, "date"] == "2016-05-22", "Actual sales date range")
    recomputed = hashlib.sha256(("\n".join(dev.series_id) + "\n").encode()).hexdigest()
    check(recomputed == manifest["development"]["series_ids_sha256"], "Sample hash")
    hierarchy_checks = []
    hierarchy_keys = [[], ["state_id"], ["store_id"], ["cat_id"], ["dept_id"],
                      ["state_id", "cat_id"], ["state_id", "dept_id"], ["store_id", "cat_id"],
                      ["store_id", "dept_id"], ["item_id"], ["state_id", "item_id"], ["item_id", "store_id"]]
    expected_node_counts = [1, 3, 10, 3, 7, 9, 21, 30, 70, 3049, 9147, 30490]
    for scope, leaves in [("final", sales), ("development", dev)]:
        mapping = pd.read_parquet(out / f"hierarchy_map_{scope}.parquet")
        check(len(mapping) == len(leaves) * 12 and not mapping.duplicated(["series_id", "level"]).any(), "12 nodes per leaf")
        merged = mapping.merge(leaves[["series_id", "state_id", "store_id", "cat_id", "dept_id", "item_id", "d_1", "d_1829", "d_1941"]], on="series_id", validate="many_to_one")
        for level, group in merged.groupby("level"):
            aggregates = group.groupby("node_id")[["d_1", "d_1829", "d_1941"]].sum()
            keys = hierarchy_keys[level - 1]
            expected_keys = [json.dumps(list(values), separators=(",", ":")) for values in group[keys].itertuples(index=False, name=None)] if keys else ["[]"] * len(group)
            expected_nodes = [f"L{level:02d}:{key}" for key in expected_keys]
            check(group.node_id.tolist() == expected_nodes, f"Hierarchy parent keys {scope}/{level}")
            if scope == "final":
                check(len(aggregates) == expected_node_counts[level - 1], f"Official node count {level}")
            check(aggregates.sum().equals(leaves[["d_1", "d_1829", "d_1941"]].sum()), f"Mass conservation {scope}/{level}")
            hierarchy_checks.append({"scope": scope, "level": int(level), "nodes": len(aggregates), "mass_conserved": True})
    sample_ids = sorted(set(meta.series_id.iloc[[0, 15245, -1]].tolist() + dev.series_id.iloc[:3].tolist()))
    sample_sales = sales.set_index("series_id").loc[sample_ids]
    prices = pd.read_parquet(out / "prices.parquet")
    price_sample = prices[prices.series_id.isin(sample_ids)].copy()
    week_end = calendar.reset_index().groupby("wm_yr_wk").d_num.min() + 6
    price_sample["week_end"] = price_sample.wm_yr_wk.map(week_end)
    counts = []
    feature_list = json.loads((out / "feature_list.json").read_text())["predictors"]
    for origin in manifest["protocol"]["training_origins"]:
        for scope, size in [("final", len(meta)), ("development", len(dev))]:
            feature_path = out / scope / "features" / f"origin_{origin}.parquet"
            label_path = out / scope / "targets" / f"origin_{origin}.parquet"
            check(pq.read_metadata(feature_path).num_rows == size * 28, "Feature row count")
            check(pq.read_metadata(label_path).num_rows == size * 28, "Target row count")
            schema = pq.read_schema(feature_path).names
            check("units" not in schema and set(feature_list) <= set(schema), "Predictor schema")
            # Validate all key rows, label availability and uniqueness, without loading all features.
            labels = pd.read_parquet(label_path)
            check(labels.origin_d.eq(origin).all() and labels.target_d.between(origin + 1, origin + 28).all(), "Snapshot boundaries")
            check(not labels.duplicated(["series_id", "target_d"]).any() and labels.series_id.nunique() == size, "Snapshot coverage")
            check(labels.units.notna().eq(labels.target_d <= 1941).all(), "Future label availability")
            counts.append({"origin_d": origin, "scope": scope, "rows": len(labels)})
        features = pd.read_parquet(out / "final" / "features" / f"origin_{origin}.parquet", filters=[("series_id", "in", sample_ids)])
        sample_labels = pd.read_parquet(out / "final" / "targets" / f"origin_{origin}.parquet", filters=[("series_id", "in", sample_ids)])
        for row in sample_labels.itertuples(index=False):
            expected = sample_sales.loc[row.series_id, f"d_{row.target_d}"] if row.target_d <= 1941 else np.nan
            check(np.allclose(row.units, expected, equal_nan=True), "Independent target alignment")
        for sid, group in features.groupby("series_id"):
            history = sample_sales.loc[sid, [f"d_{d}" for d in range(1, origin + 1)]].to_numpy(dtype=float)
            for lag in [1, 7, 14, 28, 56]:
                check(np.allclose(group[f"sales_lag_{lag}"], history[-lag]), "Independent lag check")
            for window in [7, 28, 56, 90]:
                check(np.allclose(group[f"sales_mean_{window}"], history[-window:].mean(), rtol=1e-6), "Independent rolling mean")
                check(np.allclose(group[f"sales_std_{window}"], history[-window:].std(ddof=0), rtol=1e-6), "Independent rolling std")
            past_prices = price_sample[(price_sample.series_id == sid) & (price_sample.week_end <= origin)].sort_values("week_end")
            last = np.nan if past_prices.empty else past_prices.sell_price.iloc[-1]
            check(np.allclose(group.price_last, last, equal_nan=True), "Independent past-only price")
            state = sample_sales.loc[sid, "state_id"]
            check(group.snap.tolist() == calendar.loc[group.target_d, f"snap_{state}"].tolist(), "State SNAP alignment")
    boundaries = []
    for split in manifest["splits"]:
        origin = split["origin_d"]
        check(split["train_end_d"] == origin and split["target_start_d"] == origin + 1 and split["target_end_d"] == origin + 28, "Split endpoints")
        check(calendar.loc[split["target_end_d"], "date"] == split["target_end_date"], "Split dates")
        training_targets = [t for o in manifest["protocol"]["training_origins"] if o < origin
                            for t in range(o + 1, o + 29) if split["train_start_d"] <= t <= origin]
        check(len(training_targets) == 365 and len(set(training_targets)) == 365, "Training coverage exactly365 days")
        boundaries.append({"split_id": split["split_id"], "train_days": len(training_targets),
                           "train_end_d": max(training_targets), "target_start_d": split["target_start_d"]})
    report = {"status": "passed", "series": len(sales), "development_series": len(dev),
              "artifact_checksums_verified": len(artifact_manifest),
              "sales_days": 1941, "sales_end": calendar.loc[1941, "date"],
              "hierarchy": hierarchy_checks, "snapshots": counts, "split_checks": boundaries,
              "independent_feature_sample_series": sample_ids,
              "independent_feature_checks": "all19 origins; lag1/7/14/28/56, rolling7/28/56/90 mean/std, past-only last price, target-state SNAP"}
    (out / "verification_report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({k: report[k] for k in ["status", "series", "development_series", "sales_days", "sales_end"]}))


if __name__ == "__main__":
    main()
