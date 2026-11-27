# Lightweight acceptance artifacts

This directory contains only small publishable results and contract files; it holds no full predictions.

| File | Contents |
|---|---|
| `hierarchy_nodes_final.parquet` | Node and level catalogue |
| `leaf_metadata_final.parquet` | Leaf-series hierarchy keys |
| `mint_weights.parquet` | Daily and period error variances and MinT weights |
| `daily_forecast_metrics.csv` | Daily forecast diagnostics |
| `period_forecast_metrics.csv` | 28-day cumulative forecast diagnostics |
| `coherence_diagnostics.csv` | Hierarchical summation diagnostics |
| `constraint_checks.csv` | Non-negativity and monotonicity checks |
| `mint_cg_diagnostics.csv` | CG convergence and active-set records |
| `representative_samples.csv` | Representative forecasts per level |
| `runtime_complexity.csv` | Runtime by scheme |
| `verification_report.json` | Independent recomputation summary of the full prediction output |
| `run_context.sanitized.json` | Run metadata with local paths and environment fingerprints removed |

`top_down_shares/` keeps leaf-level historical shares by split and by horizon.

The full hierarchy map sits at the repository root as `hierarchy_map_final.parquet` under the assignment naming; the whole-repository checksum list sits at the root as `ARTIFACTS.sha256`.


