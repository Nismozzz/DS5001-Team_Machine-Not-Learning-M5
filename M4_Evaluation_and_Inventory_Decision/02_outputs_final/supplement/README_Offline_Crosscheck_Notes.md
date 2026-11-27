# Offline supplementary cross-check: hierarchy reconciliation usability and order granularity

This directory holds the supplementary items of Module 04. It does not change any official deliverable (the seventeen items in `02_outputs_final` stay exactly as they are).
Three generation scripts: `01_code/make_period_pinball.py` (period-level pinball, about 5 minutes), `01_code/make_hierarchy_offline.py` (hierarchy verification and order granularity, about 15 seconds), and `01_code/check_bottom_up_call.py` (live call of the Module 03 interface, about 1 minute). None uses randomness, and all results can be recomputed bit for bit.

## 1. Why this exists

Once the full Module 03 handoff package arrived, three questions had to be settled: was reconciliation actually performed, can the quantiles be fed straight into the inventory simulation, and how large is the gap between "ordering at SKU level" and "ordering at store-total level". None of the three requires rerunning the main pipeline; all can be answered offline.

## 2. File guide

| File | Rows | Content |
|---|---|---|
| `high_level_quantile_calibration.csv` | 276 = 4 evaluation windows × 3 levels × 23 τ | Total / state / store levels: coverage of realised aggregate demand and the stock multiple at the 28-day cumulative quantiles; inputs are LightGBM's 23 quantiles (there is no model column, so do not cite as a four-model result) |
| `order_granularity_compare.csv` | 184 = 4 cost ratios × 23 τ × 2 modes | Cost per unit demand, fill rate, order units, end inventory; LightGBM only |
| `quantile_additivity_gap.csv` | 23 | Top-node value against the sum over all leaf quantiles, relative gap by τ; also based on LightGBM quantiles |
| `summary.json` | — | Self-check metadata for the period-level pinball (row count, bit-level alignment error against `coverage_report.csv`) |
| `forecast_metrics_period_pinball.csv` | 1,104 = 4 windows × 3 levels × 23 τ × 4 models | 28-day cumulative pinball loss for the four models on the same grid |
| `hierarchy_recon_summary.json` | — | Key numbers and limitations for the hierarchy verification and order granularity |
| `bottom_up_api_check.csv` | 7 | Results of a live call to the Module 03 bottom-up interface (top level and two sample SKUs) |
| `fig_order_granularity.png` | — | Cost against τ under both modes at 5:1 (LightGBM, four windows pooled) |

## 3. Core findings

1. **Point forecasts are genuinely reconciled**: the top-level 28-day cumulative point forecast moves from 1,146,624 to 1,189,844 (+3.8%) and exactly equals the sum over 30,490 leaves; the aggregation error is 0.
2. **Cumulative quantiles are not reconciled**: the `bottom_up` quantile panel is identical row by row to `unreconciled` (L01/L03/L12, 701,523 rows in total, 0 differences).
3. **Quantiles are not additive**: the relative gap between the top node and the sum over all leaves is 2.63% at τ=0.50, 35.45% at τ=0.95, and 49.81% at the maximum. The store CA_1 node sits 14.2% below the sum of its leaves.
4. **Store-level quantiles are systematically high and the total level moves the other way**: at τ=0.95 the store level covers 9–10 of 10 stores but stocks 1.45–1.50 times realised demand; at τ=0.50 only 2 of 10 stores are covered (0.20). A single total node moves the other way, reaching only 0.96–0.99 of realised demand at τ=0.95. Node-level detail is in `hierarchy_recon_summary.json`.

Items 2 to 4 are extended diagnostics and do not enter the headline results; the headline cost, coverage and decision map all use item-store (SKU) leaf-level quantiles, so the conclusions are unaffected.

5. **Order granularity comparison (5:1, τ=0.95, four windows)**: SKU-level ordering costs 0.6943 per unit demand with a fill rate of 0.9772; store-total ordering costs 0.8620 (+24.1%) with a fill rate of 0.9453 (−3.2pp). Order units fall from 7.28 million to 7.07 million, while end inventory rises from 675,000 to 701,000.
   The reason is not that the total is too small, but that spreading the store total across SKUs by point-forecast share overstocks fast movers and understocks slow movers.

## 4. Limitations

- Only LightGBM has 28-day cumulative quantiles on both the Module 02 and the Module 03 side; the cumulative quantiles of the three baselines are built inside Module 04 from "point forecast + empirical error quantiles" and are excluded from this comparison.
- Mode B spreads the store total using only SKU point-forecast shares and does not compare better allocation rules, so 24.1% is the result of the "most naive allocation", not a lower bound for that mode.
- Store-level quantile coverage is counted node by node (10 stores × 4 windows); the sample is small, so tail τ values are indicative only.
- Conclusions are bounded by all 30,490 series of the M5 dataset (CA 4 stores, TX 3 stores, WI 3 stores).

## 5. Relationship to the official deliverables

- Official numbers are unchanged; the headline still uses SKU-level ordering: at 5:1 and τ=0.95, LightGBM 0.694, TSB 0.879, ETS 0.920 and seasonal naive 1.062.
- This directory is referenced only by Section 4.7 of the report and by the sign-off note; it is not part of the seventeen-item list in `artifact_manifest.json`.

## 6. How to rerun

The two scripts carry no absolute paths; they walk up from the script location to find the project root that contains `data/` or `M5/`. The Module 03 handoff package is supplied separately through an environment variable:

```
$env:M5_ROOT   = "<project root containing data/ and M5/>"      # optional, auto-detected by default
$env:M3_HANDOFF = "<Module3_Full_Handoff_Module4>"  # required, exits with a clear error if absent
python 01_code\make_hierarchy_offline.py
python 01_code\check_bottom_up_call.py
```

`make_period_pinball.py` depends only on the Module 02 deliverables and the truth files under the project root, and is located in the same way.
