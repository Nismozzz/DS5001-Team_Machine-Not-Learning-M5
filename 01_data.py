#!/usr/bin/env python3
"""M5 module 01: frozen sources, auditable splits and origin-safe features."""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

VERSION = "m5-module01-v1"
META = ["id", "item_id", "dept_id", "cat_id", "store_id", "state_id"]
FILES = ["calendar.csv", "sales_train_evaluation.csv", "sales_train_validation.csv",
         "sell_prices.csv", "sample_submission.csv"]
LEVELS = [[], ["state_id"], ["store_id"], ["cat_id"], ["dept_id"],
          ["state_id", "cat_id"], ["state_id", "dept_id"], ["store_id", "cat_id"],
          ["store_id", "dept_id"], ["item_id"], ["state_id", "item_id"],
          ["item_id", "store_id"]]
COUNTS = [1, 3, 10, 3, 7, 9, 21, 30, 70, 3049, 9147, 30490]
ORIGINS = [1829, 1857, 1885, 1913, 1941]
HORIZON = 28
TRAIN_ORIGINS = list(range(1437, 1942, 28))


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True,
                               allow_nan=False) + "\n", encoding="utf-8")


def parquet(frame, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(path, index=False, compression="zstd")


def canonical_ids(frame):
    frame = frame.copy()
    frame["series_id"] = frame.item_id + "_" + frame.store_id
    require(frame.series_id.is_unique, "Duplicate item-store series")
    require(frame[META].notna().all().all(), "Null hierarchy key")
    return frame.sort_values("series_id").reset_index(drop=True)


def sample_ids(meta, state, size, seed):
    candidates = meta.loc[meta.state_id.eq(state), "series_id"].tolist()
    require(0 < size <= len(candidates), "Invalid development sample size/state")
    ranked = sorted(candidates, key=lambda s: (hashlib.sha256(
        f"{seed}|{s}".encode("utf-8")).hexdigest(), s))
    return sorted(ranked[:size])


def make_manifest(meta, calendar, sources, state, size, seed):
    selected = sample_ids(meta, state, size, seed)
    date = calendar.set_index("d_num").date.to_dict()
    splits = []
    for i, origin in enumerate(ORIGINS):
        splits.append({"split_id": ["validation_1", "validation_2", "validation_3",
                                    "test", "forecast"][i],
                       "role": "validation" if i < 3 else ("test" if i == 3 else "forecast"),
                       "origin_d": origin, "origin_date": date[origin],
                       "train_start_d": origin - 364, "train_end_d": origin,
                       "target_start_d": origin + 1, "target_end_d": origin + 28,
                       "target_start_date": date[origin + 1], "target_end_date": date[origin + 28],
                       "labels_available": origin < 1941})
    return {"schema_version": VERSION, "owner": "Xiaofeng ZHU (xiaofezhu5)",
            "sources": sources, "sales_version": "sales_train_evaluation.csv",
            "observations": {"series": 30490, "items": 3049, "stores": 10, "states": 3,
                             "days": 1941, "start_date": date[1], "end_date": date[1941],
                             "hierarchy_levels": 12, "hierarchy_nodes": 42840},
            "development": {"state": state, "size": size, "seed": seed,
                            "algorithm": "lowest SHA256(UTF-8(seed|series_id)), tie by series_id; output sorted",
                            "series_ids": selected,
                            "series_ids_sha256": hashlib.sha256(("\n".join(selected) + "\n").encode()).hexdigest()},
            "final": {"size": 30490, "selection": "all item-store series, sorted series_id"},
            "resource_limits": {"series_batch_size": 500, "threads": 2,
                                "recommended_ram_gib": 6, "recommended_free_disk_gib": 4,
                                "feature_rows_final": 30490 * len(TRAIN_ORIGINS) * 28,
                                "feature_rows_development": size * len(TRAIN_ORIGINS) * 28},
            "protocol": {"horizon": 28, "training_window_days": 365,
                         "training_origins": TRAIN_ORIGINS, "training_origin_stride": 28,
                         "training_row_filter": "origin_d < split.origin_d AND target_d >= train_start_d AND target_d <= train_end_d",
                         "feature_availability": "sales <= row.origin_d; prices from completed weeks <= row.origin_d; future calendar assumed known",
                         "test_policy": "Tune using validation_1..3 only; freeze configuration before test; forecast has no labels",
                         "hierarchy_development": "induced aggregates over sampled leaves only, never full-M5 aggregates"},
            "splits": splits}


def select_training_rows(frame, split):
    """Works for feature/label keys; must be applied separately for every fold."""
    return frame[(frame.origin_d < split["origin_d"]) &
                 frame.target_d.between(split["train_start_d"], split["train_end_d"])]


def origin_statistics(y, origin):
    """Never examines y[:, origin:] (origin is 1-based last observed day)."""
    history = y[:, :origin]
    n = len(history)
    result = {}
    for lag in [1, 7, 14, 28, 56]:
        result[f"sales_lag_{lag}"] = history[:, -lag].astype("float32")
    for window in [7, 28, 56, 90]:
        values = history[:, -window:]
        result[f"sales_mean_{window}"] = values.mean(axis=1).astype("float32")
        result[f"sales_std_{window}"] = values.std(axis=1, ddof=0).astype("float32")
    pos = history > 0
    first = np.where(pos.any(axis=1), pos.argmax(axis=1) + 1, 0)
    last = np.where(pos.any(axis=1), origin - pos[:, ::-1].argmax(axis=1), 0)
    age = np.where(first > 0, origin - first + 1, 0)
    result["age_since_first_sale"] = age.astype("int16")
    result["days_since_last_sale"] = np.where(last > 0, origin - last, -1).astype("int16")
    window = history[:, -365:]
    count = (window > 0).sum(axis=1)
    active_days = np.minimum(age, window.shape[1])
    mean = window.mean(axis=1)
    adi = np.divide(active_days, count, out=np.full(n, np.nan), where=count > 0)
    nzmean = np.divide(window.sum(axis=1), count, out=np.zeros(n), where=count > 0)
    nzvar = np.divide((window.astype("float64") ** 2).sum(axis=1), count,
                      out=np.zeros(n), where=count > 0) - nzmean ** 2
    cv2 = np.divide(np.maximum(nzvar, 0), nzmean ** 2,
                    out=np.full(n, np.nan), where=nzmean > 0)
    intermittent = np.where(adi < 1.32, np.where(cv2 < 0.49, "smooth", "erratic"),
                            np.where(cv2 < 0.49, "intermittent", "lumpy"))
    intermittent = np.where(count == 0, "no_demand", intermittent)
    recent = history[:, -90:].mean(axis=1)
    previous = history[:, -180:-90].mean(axis=1)
    lifecycle = np.select([first == 0, (first > 0) & (age <= 90),
                           (history[:, -90:].sum(axis=1) == 0), recent < 0.5 * previous],
                          ["not_yet_sold", "new", "dormant", "declining"], default="established")
    strata = pd.DataFrame({"mean_daily_365": mean.astype("float32"),
                           "zero_fraction_365": (window == 0).mean(axis=1).astype("float32"),
                           "active_days_365": active_days.astype("int16"),
                           "positive_days_365": count.astype("int16"),
                           "adi": adi.astype("float32"), "cv2": cv2.astype("float32"),
                           "volume": np.select([mean == 0, mean < 1, mean < 5],
                                               ["zero", "low", "medium"], default="high"),
                           "intermittency": intermittent, "lifecycle": lifecycle})
    return pd.DataFrame(result), strata


def price_features(price_matrix, week_ends, origin):
    """Use completed weeks only; missing prices stay missing until first quote."""
    past = price_matrix[:, week_ends <= origin]
    n = len(past)
    last = np.full(n, np.nan, dtype="float32")
    previous = last.copy()
    latest_end = np.full(n, -1, dtype="int16")
    past_ends = week_ends[week_ends <= origin]
    for j in range(past.shape[1]):
        mask = np.isfinite(past[:, j])
        previous[mask] = last[mask]
        last[mask] = past[mask, j]
        latest_end[mask] = past_ends[j]
    trailing = past[:, -52:]
    maximum = np.max(np.where(np.isfinite(trailing), trailing, -np.inf), axis=1)
    maximum[~np.isfinite(maximum)] = np.nan
    discount = 1 - last / maximum
    return pd.DataFrame({"price_last": last, "price_change": last / previous - 1,
                         "price_max_52w": maximum, "discount_proxy": discount,
                         "promotion_proxy": np.where(np.isfinite(discount), (discount > 0.01).astype(float), np.nan).astype("float32"),
                         "price_missing": (~np.isfinite(last)).astype("int8"),
                         "price_age_days": np.where(latest_end >= 0, origin - latest_end, -1).astype("int16")})


def build_features(meta, y, calendar, price_matrix, week_ends, origin):
    stats, _ = origin_statistics(y, origin)
    prices = price_features(price_matrix, week_ends, origin)
    base = pd.concat([meta[["series_id", "item_id", "dept_id", "cat_id", "store_id", "state_id"]].reset_index(drop=True),
                      stats, prices], axis=1)
    repeated = base.iloc[np.repeat(np.arange(len(base)), 28)].reset_index(drop=True)
    repeated["origin_d"] = np.int16(origin)
    repeated["horizon"] = np.tile(np.arange(1, 29, dtype="int8"), len(base))
    repeated["target_d"] = (origin + repeated.horizon.astype("int16")).astype("int16")
    cal = calendar.set_index("d_num")
    wanted = ["date", "wm_yr_wk", "wday", "month", "year", "event_name_1", "event_type_1",
              "event_name_2", "event_type_2", "snap_CA", "snap_TX", "snap_WI"]
    future = cal.loc[repeated.target_d, wanted].reset_index(drop=True)
    for col in ["event_name_1", "event_type_1", "event_name_2", "event_type_2"]:
        future[col] = future[col].fillna("None")
    repeated = pd.concat([repeated, future], axis=1)
    repeated["snap"] = np.select([repeated.state_id.eq(s) for s in ["CA", "TX", "WI"]],
                                 [repeated[f"snap_{s}"] for s in ["CA", "TX", "WI"]]).astype("int8")
    repeated = repeated.drop(columns=["snap_CA", "snap_TX", "snap_WI"])
    repeated["is_event"] = repeated.event_name_1.ne("None").astype("int8")
    return repeated


def audit_fields(frame, name, records):
    for col in frame:
        series = frame[col]
        records.append({"table": name, "field": col, "rows": len(series),
                        "missing": int(series.isna().sum()), "dtype": str(series.dtype),
                        "zeros": int(series.eq(0).sum()) if pd.api.types.is_numeric_dtype(series) else None})


def hierarchy(meta):
    parts, summary = [], []
    for level, keys in enumerate(LEVELS, 1):
        if keys:
            nodes = meta[keys].apply(lambda row: json.dumps(row.tolist(), separators=(",", ":")), axis=1)
        else:
            nodes = pd.Series(["[]"] * len(meta))
        nodes = "L" + str(level).zfill(2) + ":" + nodes
        parts.append(pd.DataFrame({"series_id": meta.series_id, "level": np.int8(level), "node_id": nodes}))
        summary.append({"level": level, "keys": "+".join(keys) or "total", "nodes": nodes.nunique(), "leaves": len(meta)})
    return pd.concat(parts, ignore_index=True), pd.DataFrame(summary)


def dictionary(meta, sales, calendar, prices, feature_sample, strata):
    records = []
    descriptions = {
        "series_id": "Canonical item_id + '_' + store_id; no validation/evaluation suffix",
        "id": "Original Kaggle identifier; suffix denotes file version, not a different SKU",
        "item_id": "Product identifier; 3049 distinct products, each present at all10 stores",
        "dept_id": "Department identifier; 7 departments; each product has one department",
        "cat_id": "Category identifier: FOODS/HOBBIES/HOUSEHOLD; each department has one category",
        "store_id": "Store identifier: CA_1..CA_4, TX_1..TX_3, WI_1..WI_3",
        "state_id": "State identifier: CA/TX/WI; store belongs to exactly one state",
        "date": "ISO calendar date; sales d_1=2011-01-29, d_1941=2016-05-22",
        "d": "Original M5 calendar day key d_1..d_1969",
        "d_num": "Integer day key extracted from d; 1-based",
        "wm_yr_wk": "M5 retail-week identifier; joins calendar to weekly prices",
        "weekday": "English weekday label",
        "wday": "M5 weekday: Saturday=1, Sunday=2, ..., Friday=7",
        "month": "Calendar month1..12 of target date",
        "year": "Calendar year of target date",
        "sell_price": "Weekly selling price in USD; observed quotes positive; absent quote is not zero",
        "price_last": "Last observed price in a completed calendar week at origin; past-only carry forward",
        "price_change": "Last observed quote / preceding observed quote - 1; null without two quotes",
        "price_max_52w": "Maximum observed price in last 52 completed calendar weeks",
        "discount_proxy": "1 - price_last / price_max_52w; descriptive proxy, not a verified promotion",
        "promotion_proxy": "discount_proxy > 0.01; null when reference unavailable",
        "price_missing": "1 if no price observed by origin; no future backfill",
        "price_age_days": "Days since end of last completed week with price; -1 if unknown",
        "origin_d": "Last observed day, inclusive; all sales/price features bounded here",
        "target_d": "origin_d + horizon, 1-based M5 day index",
        "horizon": "Direct forecast step, 1..28; lags refer to origin, not target",
        "age_since_first_sale": "Origin minus first positive-sale day plus 1; 0 if no positive history",
        "days_since_last_sale": "Origin minus latest positive-sale day; -1 if no positive history",
        "adi": "min(days since first positive sale,365) / positive days in trailing365; null if no demand",
        "cv2": "Population variance of positive trailing365 sales / squared positive-sales mean; null if no demand",
        "volume": "Trailing365 mean: zero=0, low=(0,1), medium=[1,5), high>=5",
        "intermittency": "ADI cutoff1.32/CV2 cutoff0.49: smooth/erratic/intermittent/lumpy; no_demand if zero positives",
        "lifecycle": "Priority: not_yet_sold, new(age<=90), dormant(last90 sum=0), declining(last90 mean<0.5 prior90 mean), established",
        "units": "Observed nonnegative unit sales; null for targets after d_1941; label only",
        "node_id": "Lxx:JSON ordered hierarchy-key array; induced totals for development",
        "level": "Official M5 aggregation level1..12",
        "event_name_1": "Scheduled event1; raw null means no event; feature value None",
        "event_name_2": "Scheduled event2; raw null means no second event; feature value None",
        "event_type_1": "Event1 type; raw null means no event; feature value None",
        "event_type_2": "Event2 type; raw null means no second event; feature value None",
        "is_event": "1 if target date has an event_name_1",
        "snap_CA": "Scheduled California SNAP eligibility indicator0/1",
        "snap_TX": "Scheduled Texas SNAP eligibility indicator0/1",
        "snap_WI": "Scheduled Wisconsin SNAP eligibility indicator0/1",
        "snap": "Target-date SNAP flag for the series state; calendar assumed known at origin",
        "mean_daily_365": "Mean sales in last365 calendar days through origin, including observed leading zeros",
        "zero_fraction_365": "Fraction of zero-sale days in last365 calendar days through origin",
        "active_days_365": "min(origin - first positive sale day + 1,365), or0 if no positive history",
        "positive_days_365": "Number of positive-sale days in last365 calendar days through origin",
    }
    tables = {"series": meta, "calendar": calendar, "prices": prices,
              "features": feature_sample, "demand_strata": strata,
              "hierarchy_map": pd.DataFrame({"series_id": [""], "level": [1], "node_id": [""]}),
              "targets": pd.DataFrame({"series_id": [""], "origin_d": [1], "target_d": [2], "units": [1.]})}
    for table, frame in tables.items():
        for col in frame:
            desc = descriptions.get(col, col)
            if col.startswith("sales_lag_"):
                lag = int(col.rsplit("_", 1)[1])
                desc = f"units[origin_d-{lag}+1]; lag1 is the last observed day"
            elif col.startswith("sales_mean_") or col.startswith("sales_std_"):
                desc = "Trailing " + col.rsplit("_", 1)[1] + " days ending at origin inclusive; std ddof=0"
            role = "label" if table == "targets" and col == "units" else "metadata"
            if table == "features" and col not in ["series_id", "origin_d", "target_d", "date"]:
                role = "predictor"
            records.append({"table": table, "field": col, "dtype": str(frame[col].dtype),
                            "role": role, "description": desc,
                            "availability": "known calendar assumption" if table == "calendar" else
                            ("observed only through origin" if table in ["features", "demand_strata"] else "source/structural")})
    for table in ["sales_development", "sales_final"]:
        for col in meta:
            records.append({"table": table, "field": col, "dtype": str(meta[col].dtype),
                            "role": "identifier", "description": descriptions.get(col, col), "availability": "structural"})
        records.append({"table": table, "field": "d_1..d_1941", "dtype": "int16",
                        "role": "label_history", "description": "Daily unit sales, wide; zero is observed zero, never missing; no censoring/stockout correction",
                        "availability": "day-specific; never read beyond origin"})
    return pd.DataFrame(records)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-dir", type=Path, default=Path("m5-forecasting-accuracy"))
    parser.add_argument("--output-dir", type=Path, default=Path("data/processed"))
    parser.add_argument("--lock-file", type=Path, default=Path("raw_data_manifest.json"))
    parser.add_argument("--state", default="CA")
    parser.add_argument("--sample-size", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=5001)
    parser.add_argument("--manifest-only", action="store_true")
    args = parser.parse_args()
    start = time.monotonic()
    pa.set_cpu_count(2)
    out = args.output_dir
    out.mkdir(parents=True, exist_ok=True)
    sources = {name: {"bytes": (args.raw_dir / name).stat().st_size,
                      "sha256": sha256(args.raw_dir / name)} for name in FILES}
    if args.lock_file.exists():
        require(json.loads(args.lock_file.read_text())["files"] == sources,
                "Raw data differs from frozen lock; investigate, do not silently replace the lock")
    else:
        write_json(args.lock_file, {"schema_version": VERSION, "files": sources,
                                   "provenance": "User-supplied local Kaggle-format files; original download/account acceptance not independently verified",
                                   "source_url": "https://www.kaggle.com/competitions/m5-forecasting-accuracy/data",
                                   "rules_url": "https://www.kaggle.com/competitions/m5-forecasting-accuracy/rules",
                                   "terms": "Access via Kaggle account, join competition and accept applicable rules; no independent redistribution license assumed. Raw and derived data excluded from git."})
    calendar = pd.read_csv(args.raw_dir / "calendar.csv")
    calendar["d_num"] = calendar.d.str[2:].astype("int16")
    require(calendar.d_num.tolist() == list(range(1, 1970)), "Unexpected calendar days")
    require(pd.to_datetime(calendar.date).diff().dropna().eq(pd.Timedelta(days=1)).all(), "Non-daily calendar")
    meta = canonical_ids(pd.read_csv(args.raw_dir / "sales_train_evaluation.csv", usecols=META))
    require(len(meta) == 30490 and meta.item_id.nunique() == 3049 and meta.store_id.nunique() == 10
            and meta.state_id.nunique() == 3, "Unexpected M5 cardinalities")
    manifest = make_manifest(meta, calendar, sources, args.state, args.sample_size, args.seed)
    write_json(out / "split_manifest.json", manifest)
    if args.manifest_only:
        print(f"Rebuilt manifest: {out / 'split_manifest.json'}", flush=True)
        return
    print("Sources frozen; loading and auditing sales/prices", flush=True)
    missing = []
    dtype = {f"d_{d}": "int32" for d in range(1, 1942)}
    sales = canonical_ids(pd.read_csv(args.raw_dir / "sales_train_evaluation.csv", dtype=dtype))
    daycols = [f"d_{d}" for d in range(1, 1942)]
    require([c for c in sales if c.startswith("d_")] == daycols, "Sales dates incomplete/unordered")
    y32 = sales[daycols].to_numpy()
    require(np.isfinite(y32).all() and y32.min() >= 0 and y32.max() <= 32767, "Invalid unit sales")
    y = y32.astype("int16")
    del y32
    sales[daycols] = sales[daycols].astype("int16")
    audit_fields(sales, "sales_train_evaluation", missing)
    validation = canonical_ids(pd.read_csv(args.raw_dir / "sales_train_validation.csv",
                                           dtype={k: v for k, v in dtype.items() if int(k[2:]) <= 1913}))
    require(validation.series_id.equals(meta.series_id), "Validation identifiers differ")
    require(np.array_equal(validation[daycols[:1913]].to_numpy(), y[:, :1913]), "Validation/evaluation overlap differs")
    require(validation[[c for c in META if c != "id"]].equals(meta[[c for c in META if c != "id"]]), "Hierarchy drift between versions")
    require(sales.id.eq(sales.series_id + "_evaluation").all() and
            validation.id.eq(validation.series_id + "_validation").all(), "Unexpected source IDs")
    audit_fields(validation, "sales_train_validation", missing)
    del validation
    prices = pd.read_csv(args.raw_dir / "sell_prices.csv", dtype={"sell_price": "float32", "wm_yr_wk": "int32"})
    audit_fields(prices, "sell_prices", missing)
    require(not prices.duplicated(["store_id", "item_id", "wm_yr_wk"]).any(), "Duplicate price key")
    require(prices.notna().all().all() and np.isfinite(prices.sell_price).all() and prices.sell_price.gt(0).all(), "Invalid price")
    prices["series_id"] = prices.item_id + "_" + prices.store_id
    require(prices.series_id.isin(meta.series_id).all(), "Unknown price item-store")
    weeks = calendar.groupby("wm_yr_wk", sort=False).d_num.agg(["min", "max"])
    require(prices.wm_yr_wk.isin(weeks.index).all(), "Unknown price week")
    price_matrix = prices.pivot(index="series_id", columns="wm_yr_wk", values="sell_price").reindex(index=meta.series_id, columns=weeks.index).to_numpy(dtype="float32")
    # Boundary weeks in supplied calendar may be incomplete; never pretend the last
    # calendar fragment is a completed week (the final 1969-day week is one day).
    week_ends = (weeks["min"] + 6).to_numpy(dtype="int16")
    audit_fields(calendar, "calendar", missing)
    submission = pd.read_csv(args.raw_dir / "sample_submission.csv")
    audit_fields(submission, "sample_submission", missing)
    require(submission.id.is_unique and len(submission) == 60980 and list(submission.columns) == ["id"] + [f"F{i}" for i in range(1, 29)], "Unexpected submission schema")
    require(set(submission.id) == set(meta.series_id + "_validation") | set(meta.series_id + "_evaluation"), "Submission IDs differ")
    del submission
    for child, parent in [("item_id", "dept_id"), ("dept_id", "cat_id"), ("store_id", "state_id")]:
        require(meta.groupby(child)[parent].nunique().eq(1).all(), f"Inconsistent {child}->{parent}")
    hmap, hsummary = hierarchy(meta)
    require(hsummary.nodes.tolist() == COUNTS and hsummary.nodes.sum() == 42840, "Hierarchy counts differ")
    selected = meta.series_id.isin(manifest["development"]["series_ids"]).to_numpy()
    devmap, devsummary = hierarchy(meta.loc[selected].reset_index(drop=True))
    parquet(meta, out / "series.parquet")
    parquet(sales, out / "sales_final.parquet")
    parquet(sales.loc[selected], out / "sales_development.parquet")
    parquet(calendar, out / "calendar.parquet")
    parquet(prices, out / "prices.parquet")
    parquet(hmap, out / "hierarchy_map_final.parquet")
    parquet(devmap, out / "hierarchy_map_development.parquet")
    hsummary.to_csv(out / "hierarchy_audit.csv", index=False)
    devsummary.to_csv(out / "hierarchy_development_audit.csv", index=False)
    cardinality = [{"field": col, "distinct": int(meta[col].nunique()), "missing": int(meta[col].isna().sum())}
                   for col in ["series_id", "item_id", "dept_id", "cat_id", "store_id", "state_id"]]
    pd.DataFrame(cardinality).to_csv(out / "cardinality_audit.csv", index=False)
    pd.DataFrame(missing).to_csv(out / "missing_audit.csv", index=False)
    frequency = [
        {"scope": "sales_final", "frequency": "daily", "series": 30490, "expected_cells": y.size, "observed_cells": y.size, "missing_cells": 0, "start": calendar.date.iloc[0], "end": calendar.date.iloc[1940]},
        {"scope": "calendar", "frequency": "daily", "series": 1, "expected_cells": 1969, "observed_cells": len(calendar), "missing_cells": 0, "start": calendar.date.iloc[0], "end": calendar.date.iloc[-1]},
        {"scope": "prices", "frequency": "weekly", "series": 30490, "expected_cells": price_matrix.size, "observed_cells": int(np.isfinite(price_matrix).sum()), "missing_cells": int(np.isnan(price_matrix).sum()), "start": int(weeks.index.min()), "end": int(weeks.index.max())}]
    pd.DataFrame(frequency).to_csv(out / "frequency_audit.csv", index=False)
    daily_price_present = np.isfinite(price_matrix[:, weeks.index.get_indexer(calendar.wm_yr_wk.iloc[:1941])])
    price_coverage = pd.DataFrame({"series_id": meta.series_id,
                                   "missing_price_days": (~daily_price_present).sum(axis=1),
                                   "positive_sales_without_weekly_price": ((y > 0) & ~daily_price_present).sum(axis=1),
                                   "zero_sales_days": (y == 0).sum(axis=1)})
    price_coverage.to_csv(out / "price_coverage_audit.csv", index=False)
    del daily_price_present, sales, prices, hmap, devmap
    print("Cardinalities, hierarchy and overlap verified; writing origin-safe feature snapshots", flush=True)
    feature_audit, strata_parts = [], []
    sample_feature = None
    for origin in TRAIN_ORIGINS:
        _, strata = origin_statistics(y, origin)
        strata.insert(0, "series_id", meta.series_id)
        strata.insert(1, "origin_d", np.int16(origin))
        strata_parts.append(strata)
        writers = {}
        try:
            for batch_start in range(0, len(meta), 500):
                batch_end = min(batch_start + 500, len(meta))
                m = meta.iloc[batch_start:batch_end].reset_index(drop=True)
                f = build_features(m, y[batch_start:batch_end], calendar,
                                   price_matrix[batch_start:batch_end], week_ends, origin)
                target_index = f.target_d.to_numpy() - 1
                rows = np.repeat(np.arange(batch_start, batch_end), 28)
                units = np.full(len(f), np.nan, dtype="float32")
                known = target_index < 1941
                units[known] = y[rows[known], target_index[known]]
                labels = f[["series_id", "origin_d", "target_d"]].copy()
                labels["units"] = units
                require(not {"units", "volume", "intermittency", "lifecycle"} & set(f), "Labels leaked into features")
                if sample_feature is None:
                    sample_feature = f.iloc[:2].copy()
                for scope, mask in [("final", np.ones(len(f), dtype=bool)),
                                    ("development", np.repeat(selected[batch_start:batch_end], 28))]:
                    if not mask.any():
                        continue
                    for kind, frame in [("features", f), ("targets", labels)]:
                        key = (scope, kind)
                        table = pa.Table.from_pandas(frame.loc[mask], preserve_index=False)
                        if key not in writers:
                            path = out / scope / kind / f"origin_{origin}.parquet"
                            path.parent.mkdir(parents=True, exist_ok=True)
                            writers[key] = pq.ParquetWriter(path, table.schema, compression="zstd", use_dictionary=True)
                        writers[key].write_table(table)
                for col in f:
                    nnull = int(f[col].isna().sum())
                    feature_audit.append({"origin_d": origin, "field": col, "rows": len(f), "missing": nnull})
        finally:
            for writer in writers.values():
                writer.close()
        print(f"  origin {origin}: {len(meta) * 28:,} final / {int(selected.sum()) * 28:,} development rows", flush=True)
    all_strata = pd.concat(strata_parts, ignore_index=True)
    parquet(all_strata, out / "demand_strata_final.parquet")
    parquet(all_strata[all_strata.series_id.isin(manifest["development"]["series_ids"])], out / "demand_strata_development.parquet")
    if feature_audit:
        pd.DataFrame(feature_audit).groupby(["origin_d", "field"], as_index=False)[["rows", "missing"]].sum().to_csv(out / "feature_missing_audit.csv", index=False)
    else:
        pd.DataFrame(columns=["origin_d", "field", "rows", "missing"]).to_csv(out / "feature_missing_audit.csv", index=False)
    # Demand tables include descriptive labels for evaluation; predictors have their own explicit allow-list.
    model_features = [c for c in sample_feature if c not in ["series_id", "origin_d", "target_d", "date"]]
    write_json(out / "feature_list.json", {"predictors": model_features, "keys": ["series_id", "origin_d", "target_d"],
                                           "label_file": "targets/origin_{origin_d}.parquet", "label": "units"})
    dictionary(meta, None, calendar, pd.read_parquet(out / "prices.parquet").iloc[:1],
               sample_feature, all_strata.iloc[:1]).to_csv(out / "data_dictionary.csv", index=False)
    artifacts = {str(p.relative_to(out)): {"bytes": p.stat().st_size, "sha256": sha256(p)}
                 for p in sorted(out.rglob("*")) if p.is_file() and p.name not in ["artifact_manifest.json", "run_report.json", "verification_report.json"]}
    write_json(out / "artifact_manifest.json", artifacts)
    write_json(out / "run_report.json", {"schema_version": VERSION, "python": platform.python_version(),
                                         "pandas": pd.__version__, "numpy": np.__version__, "pyarrow": pa.__version__,
                                         "elapsed_seconds": round(time.monotonic() - start, 3),
                                         "checks": {"raw_hash_lock": True, "validation_overlap_equal": True,
                                                    "sales_daily_contiguous": True, "hierarchy_counts": COUNTS,
                                                    "sales_last_date": calendar.date.iloc[1940],
                                                    "future_labels_null": True},
                                         "artifact_count": len(artifacts)})
    print(f"Complete: {out}, elapsed {time.monotonic()-start:.1f}s", flush=True)


if __name__ == "__main__":
    main()
