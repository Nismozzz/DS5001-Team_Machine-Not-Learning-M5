# DS5001 · Module 4 — Evaluation, Inventory Simulation and Decision Map

**Team Machine Not Learning** · Member 4 · Xiang LI (xli3487)

This directory is the complete deliverable pack of Module 04, covering forecast evaluation, inventory simulation and the decision map: 35 deliverables plus this README and `Directory_Structure.md`. Every number comes from all 30,490 item-store series over four scorable windows; no sampled or simulated data are used.

## Overview



* Models evaluated: seasonal naive, exponential smoothing, TSB and LightGBM, primarily at SKU level on a 28-day cumulative basis.

* Simulation conventions: zero lead time, no backlog of unfilled demand, no intermediate replenishment within the 28-day period; the first period starts from zero opening inventory, and each later period opens with the previous period's carry-over.

* Cost convention: cost ratios only (holding : stockout = 1:1, 2:1, 5:1, 9:1), so no absolute prices are needed and missing prices do not matter.

* 17 official artifacts, all 11 acceptance checks pass; in addition there are 8 offline supplementary items (`02_outputs_final/supplement/`) and 1 supplementary figure, shipped with the pack but outside the official list and the hash check.

## Contents



| Path                             | Content                                 |
| ------------------------------ | ---------------------------------- |
| `01_code/`                     | Main pipeline notebook and command-line script, plus 3 offline recomputation scripts   |
| `02_outputs_final/tables/`     | 9 official result tables (accuracy, inventory, coverage, decision map, selection and self-check)      |
| `02_outputs_final/json/`       | 6 run and handoff metadata files (acceptance, hashes, parameters, sign-off, recomputation)        |
| `02_outputs_final/figures/`    | 4 figures (3 official + 1 supplementary)               |
| `02_outputs_final/supplement/` | 8 supplementary items (period-level pinball, hierarchy check, order granularity, interface check) |
| `03_docs/`                     | Module report (official docx + md reading mirror)  |
| `04_contribution/`             | Individual contribution statement                             |
| `README.md`                    | This file: overview, environment, run commands, results and limitations           |
| `Directory_Structure.md`                     | File-by-file description                              |


## Environment



| Component         | Version                         |
| ---------- | -------------------------- |
| Python     | 3.12.16                    |
| pandas     | 2.2.3                      |
| numpy      | 2.2.6                      |
| pyarrow    | 19.0.1                     |
| matplotlib | 3.11.2                     |
| scipy      | 1.18.1 (only for calling the Module 03 quantile reconciliation interface) |

Other dependencies: python-docx, nbformat, nbclient, ipykernel.

## Quick start



```
# Main pipeline (full data, serial, about 24 minutes)
python 04_evaluate.py

# Main pipeline notebook (with full output, about 21 minutes)
jupyter nbconvert --to notebook --execute --inplace 04_evaluate.ipynb

# Supplementary items
python make_period_pinball.py      # 23-point period-level pinball, about 5 minutes
python make_hierarchy_offline.py   # hierarchy check and order granularity, about 15 seconds
python check_bottom_up_call.py     # live call of the Module 03 interface, about 1 minute
```

The last two supplementary scripts need the Module 03 handoff package: point `M3_HANDOFF` at `Module3_Full_Handoff_Module4`; if it is unset the script exits with an explicit error. The project root can be set with `M5_ROOT`; by default it is found by walking up from the script location to the directory that contains `data/` or `M5/`, so no path is hard-coded.

Data are read through relative paths:



* Truth: `data/processed/final/targets/origin_{1829,1857,1885,1913}.parquet`

* Forecasts and residuals: `data/processed_final/member2/{test,validation_1..3,calibration_1745,1773,1801}/`

* Hierarchy mapping: `data/processed/hierarchy_map_final.parquet`

* Module 03 handoff package: path configured separately inside the scripts

## Main results



| Metric                        | Result                                                 |
| ------------------------- | -------------------------------------------------- |
| SKU-level WAPE (mean of four windows)         | LightGBM 0.7399, exponential smoothing 0.7406, TSB 0.7466, seasonal naive 0.8752 |
| Cost (τ=0.95, 5:1, per unit of demand, mean of four windows) | LightGBM 0.694, TSB 0.879, exponential smoothing 0.920, seasonal naive 1.062     |
| Lead (5:1)                 | 21.0% below TSB, 24.5% below exponential smoothing                         |
| Coverage (τ=0.95, LightGBM, four windows)  | 0.9182 / 0.9259 / 0.9283 / 0.9200, maximum deviation 0.0318      |
| Optimal service level (5:1)               | LightGBM 0.86, TSB 0.85, exponential smoothing 0.86, seasonal naive 0.87         |
| Paired bootstrap (against seasonal naive, 5:1, τ=0.95, B=200) | LightGBM is on average 57.48 lower per series (four windows pooled, Co=1), interval \[−59.45, −55.39], significant |
| Mean of the three decision-map splits                  | 15.86 / 31.61 / 26.21, equal weight 24.56%                    |

Supplementary conclusions:



* On the 28-day cumulative SKU-level 23-point pinball (supplementary table), LightGBM is lowest of the four models at all 23 service levels; at store and total level it is lowest at only 6 levels. The pinball in the main table `forecast_metrics.csv` is on a daily basis, where LightGBM shares only 4 common levels, so the two must not be compared directly.

* Quantiles are not additive: the top level and the sum over leaves differ by 35.4% at τ=0.95, and in the test window the top level is 1.48 times realised demand. That panel is an extended diagnostic only and does not enter the headline results; it does not affect the leaf-level cost conclusions. Node-level detail is in `02_outputs_final/supplement/hierarchy_recon_summary.json`.

* When the store only sets a total and spreads it to SKUs by point-forecast share, cost rises from 0.6943 to 0.8620 (+24.1%) and the fill rate falls from 0.9772 to 0.9453.

## Reproducibility



* All 11 acceptance checks in the run report pass; the artifact manifest lists 17 items with hashes in `artifact_manifest.json`. Sixteen of them are stored in Git; the per-series cost table (about 170 MB) is flagged `in_git: false` and regenerated by `01_code/04_evaluate.py`.

* Independent rerun cross-check: 9 data files and the input sign-off file are byte-identical; only the 6 metadata files that carry timestamps differ.

* Module 01 inputs were rehashed one by one, 95/95 match, recorded in `input_signoff.json`.

* Randomness appears only in the paired bootstrap, with seed fixed at 5001; the rest of the pipeline uses no random numbers.

## Data dependencies



| Source              | Use                         |
| --------------- | -------------------------- |
| Module 01 (Xiaofeng) | Truth values, split list, long-run sales, demand segments, hierarchy mapping     |
| Module 02 (Jiaheng)  | LightGBM daily and 28-day cumulative forecasts, historical residuals |
| Module 03 (Junze)    | Hierarchy catalogue, reconciled point forecasts, reconciliation interface          |

The 28-day cumulative quantiles of the three baseline models are built inside this module from historical rolling residuals, as noted in the report.

## Scope and limitations



* The sample is all 30,490 series of the M5 dataset (CA 4 stores, TX 3, WI 3); costs are normalised by total demand, so absolute amounts cannot be extrapolated to other data sources.

* Daily quantiles offer only four common levels (0.5 / 0.9 / 0.95 / 0.99), so the four-model comparison uses the 28-day cumulative 23-point grid.

* Quantiles are not additive, so segmented quantile conclusions stop at leaf level; high-level segment diagnostics use the point-forecast convention, and high-level cumulative quantiles do not enter the headline results.

* The quantile panel delivered by Module 03 is unreconciled; this module builds its own catalogue for option A and calls the bottom-up scheme live, with output matching the leaf sums level by level. The item-level quantiles obtained from that live call are used only for interface verification and do not enter the cost or decision results.

* Only one model has 28-day cumulative quantiles on both sides, so the order granularity comparison covers that model only.

## Documents



* `03_docs/Module04_Task_Brief_and_Stage_Conclusions.docx`: module report with purpose, inputs, conventions, results, review, boundaries and handoff items; the `.md` in the same directory is a read-only mirror of the same content for online reading.

* `02_outputs_final/supplement/README_Offline_Crosscheck_Notes.md`: method, conclusions and limitations of the supplementary items.

* `Directory_Structure.md`: file-by-file description.
