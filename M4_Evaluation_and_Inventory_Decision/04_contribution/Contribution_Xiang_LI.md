# Contribution statement — Xiang LI (xli3487)

Team: Machine Not Learning　·　Project: DS5001 group assignment　·　Module: 04

## 1. Scope of responsibility

Within Module 04 I owned three parts: forecast evaluation (accuracy and quantile loss), inventory simulation (cost and service level), and the decision map (model choice segmented by demand characteristics). I also handled the input interface and convention sign-off with Modules 01/02/03, as well as this module's handoff documents and the packaging of the deliverable set.

## 2. What was delivered

34 deliverables in total, plus 2 explanatory files (`README.md` and `Directory_Structure.md`).

| Category | Count | Content |
| --- | --- | --- |
| Code | 5 | Main pipeline notebook and command-line version, 3 offline recomputation scripts |
| Result tables | 9 | Accuracy, inventory, coverage, decision map, equal-coverage cross-check, bootstrap, optimal service level, input self-check |
| Metadata | 6 | Run report, artifact hash manifest, handoff summary, simulation parameters, input sign-off, recomputation log |
| Figures | 4 | Cost curve, accuracy-cost scatter, segment heterogeneity, order granularity comparison |
| Supplementary items | 8 | Period-level pinball, three-level quantile calibration, order granularity, additivity gap, interface check, explanatory notes |
| Documents | 2 | Module report, this statement |

## 3. Main work and conclusions

1. **Evaluation grid and conventions**: fixed the four scorable windows, 30,490 series, and the period-level convention with 23 service levels, and handled the upstream limit that daily level offers only four quantile levels, so the four models are never ranked at incomparable levels.
2. **Full sweep over cost and service level**: ran the inventory simulation over the four cost ratios and 23 service levels, obtaining the optimal service levels (5:1, LightGBM 0.86, TSB 0.85) and the cost curves; the headline setting is τ=0.95 with 5:1.
3. **Decision map**: segmented along three dimensions — demand type, sales scale and life cycle — producing 1,288 rows of segmented results, with an equal-weight three-dimension advantage of 24.56%.
4. **Locating and handling the coverage problem**: confirmed the upstream 23 points are raw uncalibrated quantiles, with realised coverage at nominal τ=0.95 of 0.918–0.928; on that basis the evaluation side supplies an equivalent-quantile mapping (interpolation to nominal 0.95 coverage) without requiring a rerun on the forecasting side, and both cost sets are reported side by side.
5. **Hierarchy reconciliation panel check**: verified the Module 03 handoff package row by row, giving three conclusions — point forecasts are genuinely reconciled (the top level moves from 1,146,624 to 1,189,844 and equals the leaf sum), cumulative quantiles are not reconciled (701,523 rows, zero difference), and quantiles are not additive (35.4% gap at τ=0.95).
6. **Order granularity comparison**: under the same simulation core, compared ordering by SKU with ordering by store total: cost 0.6943 against 0.8620 (+24.1%), fill rate 0.9772 against 0.9453 — the store total is sufficient in volume but mismatched in composition.
7. **Interface evidence**: installed the dependencies as agreed in integration, built the directories from the handoff package's hierarchy mapping, and called the Module 03 quantile reconciliation interface live; the output matches this side's leaf sum bit for bit, confirming the convention gap is resolved.
8. **Documents and handoff**: wrote the module report (97 paragraphs), organised the delivery directory and the file-by-file description, and committed every supplementary item as a rerunnable script.

## 4. Interfaces with others

| Counterpart | Content |
| --- | --- |
| Module 01 (Xiaofeng) | Received truth values, splits, the sales wide table, demand segments and the hierarchy mapping; recomputed every hash, 95/95 match, and left a sign-off record |
| Module 02 (Jiaheng) | Used its daily and cumulative forecasts and historical residuals; confirmed 8 conventions including the attribution of baseline quantiles, the daily quantile levels and the calibration method |
| Module 03 (Junze) | Checked its full handoff package and established what is usable and what is not; interpreted or aligned on output layout, reconciliation scope, metric reuse and the cumulative WAPE convention, and executed the reconciliation interface call as agreed |
| Module 05 (Jianfeng) | Supplied the figures, tables and result conventions the report needs |

I did not modify any file owned by others; issues found were reported through a checklist table and written notes, never by editing their artifacts directly.

## 5. Reproducibility evidence

- All 11 acceptance checks in the run report pass; 17 official artifacts, with hashes written into the manifest.
- Independent rerun cross-check: 9 data files and the sign-off file are byte-identical; differences appear only in metadata files that carry timestamps.
- All three offline recomputation scripts rerun standalone: pinball about 5 minutes, hierarchy check about 15 seconds, interface check about 1 minute.
- Randomness appears only in the paired bootstrap, seed 5001.

## 6. Boundaries and what was not done

- The sample is all 30,490 series of the M5 dataset (CA 4 stores, TX 3, WI 3); conclusions are not extrapolated to other datasets.
- Quantiles are not additive, so segmented quantile conclusions stop at leaf level and high-level diagnostics switch to the point-forecast convention.
- The MinT against bottom-up cost comparison was not run: it needs a quantile baseline for each upper node, and the available sources contradict the leaf level, so it would only illustrate the method; it is left for later.
- The 8 supplementary items are not merged into the official artifact list, and the report notes their independent status.
