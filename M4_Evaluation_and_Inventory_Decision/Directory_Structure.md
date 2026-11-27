# Directory structure (Module 04)

## 1. Directory tree

```
Module4/
├── README.md
├── Directory_Structure.md
├── 01_code/
│   ├── 04_evaluate.ipynb
│   ├── 04_evaluate.py
│   ├── make_period_pinball.py
│   ├── make_hierarchy_offline.py
│   └── check_bottom_up_call.py
├── 02_outputs_final/
│   ├── tables/
│   │   ├── forecast_metrics.csv
│   │   ├── inventory_summary.csv
│   │   ├── inventory_summary_by_window.csv
│   │   ├── coverage_report.csv
│   │   ├── decision_map.csv
│   │   ├── crosscheck_equal_coverage.csv
│   │   ├── bootstrap_summary.csv
│   │   ├── tau_selection.csv
│   │   └── input_audit.csv
│   ├── json/
│   │   ├── run_report.json
│   │   ├── artifact_manifest.json
│   │   ├── handoff_summary.json
│   │   ├── simulation_parameters.json
│   │   ├── input_signoff.json
│   │   └── replay_log.json
│   ├── figures/
│   │   ├── fig_cost_curve.png
│   │   ├── fig_error_cost.png
│   │   ├── fig_heterogeneity.png
│   │   └── fig_order_granularity.png
│   └── supplement/
│       ├── forecast_metrics_period_pinball.csv
│       ├── summary.json
│       ├── high_level_quantile_calibration.csv
│       ├── order_granularity_compare.csv
│       ├── quantile_additivity_gap.csv
│       ├── hierarchy_recon_summary.json
│       ├── bottom_up_api_check.csv
│       └── README_Offline_Crosscheck_Notes.md
├── 03_docs/
│   ├── Module04_Task_Brief_and_Stage_Conclusions.docx
│   └── Module04_Task_Brief_and_Stage_Conclusions.md
└── 04_contribution/
    └── Contribution_Xiang_LI.md
```

> `Module4/` in the tree above is this deliverable directory, `M4_Evaluation_and_Inventory_Decision/`. `05_outputs_dev/` was a scratch area for local development outputs; it is empty, is not shipped, and is therefore not listed. The pack contains 37 files: 35 deliverables plus `README.md` and `Directory_Structure.md`.

## 2. File descriptions

### Root

| File | Description |
| --- | --- |
| README.md | Pack guide: overview, environment and versions, run commands, main results, reproducibility, data dependencies, scope and limitations |
| Directory_Structure.md | This file: directory tree and file-by-file description |

### 01_code

| File | Description |
| --- | --- |
| 04_evaluate.ipynb | Main pipeline notebook, 44 cells (20 text + 24 code), with the full run output and 3 figures |
| 04_evaluate.py | Command-line version of the main pipeline, 2,450 lines, same source as the notebook |
| make_period_pinball.py | Offline script that recomputes the 23-point period-level pinball |
| make_hierarchy_offline.py | Script for the hierarchy reconciliation check and the order granularity comparison |
| check_bottom_up_call.py | Script that calls the Module 03 quantile reconciliation interface live and checks equivalence |

### 02_outputs_final/tables

| File | Description |
| --- | --- |
| forecast_metrics.csv | Accuracy metrics for four models × four windows × multiple levels × 23 τ, 972 rows |
| inventory_summary.csv | Item level: inventory results over the cost-ratio and service-level grid, 368 rows |
| inventory_summary_by_window.csv | Same as above, split by window, 1,472 rows |
| coverage_report.csv | Item level: nominal against realised coverage, 368 rows |
| decision_map.csv | Model choice and cost advantage under three splits (intermittency, sales scale, life cycle), item level only, 1,288 rows |
| crosscheck_equal_coverage.csv | Cost comparison at equal coverage, including the interpolation to nominal 0.95 |
| bootstrap_summary.csv | Paired bootstrap results, 3 rows |
| tau_selection.csv | Optimal service level under the four cost ratios, 16 rows |
| input_audit.csv | Self-check log of the input inventory, 20 rows |

### 02_outputs_final/json

| File | Description |
| --- | --- |
| run_report.json | Acceptance results: 11 checks, 17 artifacts, 8 open items |
| artifact_manifest.json | Hash manifest of the 17 official artifacts (16 stored in Git; the 170 MB per-series cost table is flagged `in_git: false`) |
| handoff_summary.json | Handoff summary |
| simulation_parameters.json | Simulation conventions and alternatives |
| input_signoff.json | Sign-off record of the Module 01 input recomputation, 95/95 |
| replay_log.json | Cost recomputation and independent rerun records |

### 02_outputs_final/figures

| File | Description |
| --- | --- |
| fig_cost_curve.png | Cost against service level and cost ratio |
| fig_error_cost.png | Scatter comparison of accuracy and cost |
| fig_heterogeneity.png | Model heterogeneity across demand segments |
| fig_order_granularity.png | Cost comparison between SKU-level and store-total ordering (supplementary) |

### 02_outputs_final/supplement

Eight items, produced by Module 04 and shipped with the pack; there is also one supplementary figure, `figures/fig_order_granularity.png`. None of them counts towards the 17 official artifacts or the hash manifest; the conclusions appear in Sections 4.7 and 7.1 of the report.

| File | Description |
| --- | --- |
| forecast_metrics_period_pinball.csv | 23-point pinball at SKU / store / total level (4 windows × 3 levels × 23 τ × 4 models), 1,104 rows |
| summary.json | Self-check record of the period-level pinball recomputation (1,104 rows, alignment error against coverage_report) |
| high_level_quantile_calibration.csv | Quantile calibration at total, state and store level, 276 rows |
| order_granularity_compare.csv | Cost comparison of the two order granularities, 184 rows |
| quantile_additivity_gap.csv | Quantile additivity gap, 23 rows |
| hierarchy_recon_summary.json | Conclusions of the hierarchy check and comparison |
| bottom_up_api_check.csv | Equivalence check of the live reconciliation interface call, 7 rows |
| README_Offline_Crosscheck_Notes.md | Conventions, file guide and limitations of the supplementary items |

### 03_docs

| File | Description |
| --- | --- |
| Module04_Task_Brief_and_Stage_Conclusions.docx | Module report (official version), including the 6.1 conclusion table and the 7.2 hierarchy reconciliation check |
| Module04_Task_Brief_and_Stage_Conclusions.md | Read-only mirror of the same report, for direct reading on GitHub |

### 04_contribution

| File | Description |
| --- | --- |
| Contribution_Xiang_LI.md | Individual contribution statement |
