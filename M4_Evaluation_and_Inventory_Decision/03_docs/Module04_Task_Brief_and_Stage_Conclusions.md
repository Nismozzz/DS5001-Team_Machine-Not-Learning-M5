> This file is a read-only mirror of `Module04_Task_Brief_and_Stage_Conclusions.docx`, provided for direct reading on GitHub; the official version and layout follow the docx.

Module 04 Evaluation, Inventory Simulation and Decision Map: Task Brief and Stage Conclusions

DSC5001 · Team Machine Not Learning · Lead: Xiang LI (xli3487)

Full-data conventions (30,490 series · 23-point quantile grid) · 16 November 2026

Contents


# Summary

This part converts forecast results into inventory cost and answers one operational question: given a setting for replenishment and service level, does the extra modelling cost buy a lower inventory cost? The conclusions rest on the full Module 02 deliverable package, covering 30,490 series (CA 4 stores, TX 3, WI 3) and four scorable windows (d_1829, d_1857, d_1885, d_1913), on a 23-point quantile grid (0.50, 2/3, 0.80 to 0.99 in steps of 0.01, 5/6), with cost ratios 1:1, 2:1, 5:1 and 9:1, simulating four consecutive periods with end-of-period carry-over. One official run produces 17 files and takes 1,408.6 seconds (Python 3.12.16 with pandas 2.2.3, single machine, serial); all 11 acceptance checks pass.

Three core findings. First, the accuracy gaps are smaller than expected: item-level WAPE is LightGBM 0.7399, exponential smoothing 0.7406, TSB 0.7466 and seasonal naive 0.8752, the first three within 0.01 of each other. Second, the cost gaps are markedly larger: at service level 0.95 the cost per unit of demand is lowest for LightGBM, at 0.603, 0.626, 0.694 and 0.785 across the four cost ratios, 12.7% to 26.4% below TSB and 14.7% to 30.7% below exponential smoothing, the advantage growing with the cost ratio. Third, coverage is the only systematic weakness: at nominal 0.95 the realised figure is 0.9231 to 0.9434, the LightGBM band is too narrow, and τ must be raised to 0.9617 to reach 0.94 coverage; of the 368 rows on the full grid, 39 deviate by more than 0.05, concentrated at low quantiles.

Reproducibility is demonstrated by independent recomputation: without reusing any module-internal function and recomputing directly from the delivered files, WAPE differs from the archived values by at most 4.4×10⁻¹⁶, coverage by 1.1×10⁻¹⁶, and the relative error of the cost totals is 3.96×10⁻¹⁶; all three bootstrap point estimates match exactly. In another independent run, 9 data files are byte-identical.

The boundaries are equally explicit: the sample is all 30,490 series of the M5 dataset (10 stores), costs are normalised by total demand and absolute amounts cannot be extrapolated; the Module 03 panel was verified row by row, with point forecasts self-consistent but cumulative quantiles unreconciled; accuracy metrics cover item, store and total level, whereas inventory cost, coverage and the decision map (segmented model choice) stop at item level; the simulation convention fixes opening inventory as the previous period's carry-over and schedules no intermediate replenishment within the 28 days.


# 1　Purpose of this part


## 1.1　Position in the overall study

The project is a chain from forecasting to operational decisions: Module 01 assembles reproducible inputs, Module 02 produces point and quantile forecasts, Module 03 handles the hierarchy and reconciliation, and Module 05 owns reporting and presentation. The first three modules stay on the forecasting side and can only say which model has the smaller error.

Module 04 is the only link that converts forecasts into inventory cost, and the only one that can judge whether the modelling effort is worth it. Without it the team would only have an accuracy ranking and could not answer which model to choose operationally.


## 1.2　Three questions to answer

First, how large is the accuracy gap. WAPE, MASE and quantile loss are computed on the four scorable windows at item, store and total level, so that no one reports only the flattering fine-grained numbers.

Second, how much cost difference the accuracy difference buys. Inventory is simulated over four periods under a given review cycle, service level and cost ratio, and converted into total cost per unit of demand.

Third, within what range the differences hold. Segment-optimal models are found by intermittency, life cycle and sales scale, and the uncertainty is estimated with a paired bootstrap.


# 2　Inputs and dependencies


## 2.1　Inputs from Module 01

The Module 01 deliverables are the only source of real data for this part, and five kinds of artifact are used: the split list, realised sales at each rolling origin, the long-run sales wide table, the demand segment table and the hierarchy mapping. They were rebuilt locally in full following the repository instructions and rehashed one by one; all 95 artifacts match and are recorded in the input sign-off file (95/95), which also fulfils the team requirement that each member substantively review one module they do not own.


## 2.2　Inputs from Module 02

Module 02 delivers a forecast panel organised by rolling origin, with fields for series, origin date, target date, model, level, quantile level, forecast type and forecast value. This work uses the daily point forecasts and the 28-day cumulative quantiles of the four scorable windows: LightGBM's period quantiles are read directly, while the three baselines have point forecasts only and their period-cumulative quantiles are constructed inside this module — this must be stated plainly and not blurred into "Module 02 supplied all quantiles".

Quantile loss must be reported under two conventions. Daily: LightGBM uses its native daily quantiles, which exist only at 0.5, 0.9, 0.95 and 0.99; the three baselines are converted from point forecasts through a period-level scale ratio into daily quantiles with all 23 levels, so the four models are comparable only at the four common levels, and at the other 19 levels only the baselines have values and no cross-model ranking is possible. The 28-day cumulative level is recomputed offline from the same truth file on one grid for all four models: at 0.95 the mean loss per series is LightGBM 2.076, TSB 3.110, exponential smoothing 3.374 and seasonal naive 3.785, i.e. 33.2%, 38.5% and 45.2% lower; averaged over all 23 levels the figures are 3.247, 4.022, 4.272 and 4.982, with the same ordering and 19.3%, 24.0% and 34.8% lower. These numbers come from the supplementary script with the simulation untouched, and the coverage and mean_S align with the report digit by digit.


## 2.3　Inputs from Module 03

Module 03 delivers the hierarchical structure and reconciliation: 12 levels, 42,840 nodes, 30,490 leaf series and 365,880 membership relations, plus reconciliation methods and diagnostic and validation reports (2,160 forecast files, 2,192 manifest entries, additivity error zero).

Two points need stating. First, the panel is in place and verified: point forecasts were indeed reconciled bottom-up, with the top-level 28-day cumulative total moving from 1,146,624 to 1,189,844, exactly equal to the sum of leaf point forecasts; for the 28-day cumulative quantile block `bottom_up` passes the upstream values through unchanged (identical row by row to the unreconciled panel — the scheme's definition, since summing quantiles cannot restore additivity), and the panel itself does not add up — store node CA_1 is 14.2% below its leaf sum, the top level is 35.4% low at τ=0.95, and the largest node gap is 49.8%. Hierarchical accuracy metrics are therefore still reported at item, store and total level, some store and state levels switch to the point-forecast convention, and the reconciled panel is not used for quantiles. Second, the data contract requires calling its quantile reconciliation interface before simulation: this part has done so under option A, installing the dependencies and building the catalogue from the handoff package's hierarchy mapping, then calling the bottom-up scheme live on this side's item-level quantiles, with output matching this side's level-by-level sums digit for digit and a maximum difference of zero; every number for SKU-level ordering is unchanged and the convention gap is resolved. That interface check does not replace fixing the Module 03 deliverable panel; the interface only guarantees that nodes and leaves add up consistently, and does not correct the high store-level quantiles — see Section 4.7.


# 3　Structure and method


## 3.1　Pipeline composition

The twenty-four code cells fall into six blocks by responsibility: input and window locking, forecast ingestion and quality checks, accuracy evaluation, inventory simulation, presentation of conclusions, and freezing and handoff.


## 3.2　Key conventions

Simulation conventions: review period 28 days, zero lead time, opening inventory zero in the first period and equal to the previous period's carry-over thereafter, no backlog of unfilled demand, closing remainder carried into the next period; the order target is the chosen quantile of that period's 28-day cumulative demand, and the order quantity is the target minus opening on-hand, floored at zero; cost is closing remainder times holding unit cost plus unmet demand times stockout unit cost; the fill rate is the filled quantity over actual demand (equal-weight average over series). The four windows are contiguous and cover d_1830 to d_1941, 112 days in total, which allows four consecutive periods; windows without truth labels are used for forecast display only and do not enter the cost simulation.

The sweep covers 23 service levels, four cost ratios and four models, each run over the four windows: inventory summary 368 rows (4 models × 23 levels × 4 cost ratios), per-window summary 1,472 rows, decision map 1,288 rows. The bootstrap fixes seed 5001, 200 draws and a paired design, with the same sample used for both compared models to cancel sampling-scale noise.


## 3.3　Construction of baseline cumulative quantiles

The period-cumulative quantiles of the three baselines are not directly available; the final scheme takes, for each model, the empirical quantile of the 28-day cumulative error from an earlier window, adds it to the current point forecast and truncates at zero. Three earlier attempts (global scaling, single-window additive, pooled per model with clipping) were all unstable. Under nominal 0.95 this scheme gives the three baselines a realised coverage of 0.937 to 0.943, a shortfall of 0.007 to 0.013; baseline quantile quality depends on the number of look-back windows rather than on the functional form, so it can be tightened further if Module 02 supplies more windows.


# 4　Main results


## 4.1　Forecast accuracy: gaps in the third decimal place

Item-level WAPE over the four windows: LightGBM 0.7481, 0.7531, 0.7377, 0.7209; exponential smoothing 0.7438, 0.7455, 0.7512, 0.7220; TSB 0.7481, 0.7540, 0.7501, 0.7339; seasonal naive 0.8710, 0.8788, 0.8889, 0.8622; four-window means 0.7399, 0.7406, 0.7466, 0.8752. The first three differ by no more than 0.016 in any window, and their ordering flips between windows, so accuracy alone cannot give a stable ranking.

MASE is an equal-weight per-series average: exponential smoothing 1.1106, 1.1634, 1.3380, 1.1974; TSB 1.1087, 1.1514, 1.3150, 1.2032; LightGBM 1.1589, 1.2024, 1.3563, 1.2345; seasonal naive 1.2280, 1.2895, 1.4755, 1.3527. TSB is lowest in three windows, showing that once each series is divided by its own scale the simpler model has the edge.

Aggregate-level ordering differs from item level: store-level WAPE is exponential smoothing 0.0868, LightGBM 0.0725, TSB 0.1509; at total level 0.0649, 0.0475, 0.1289. Aggregation lowers the error by an order of magnitude across the board (item level 0.74, store level 0.07 to 0.15, total level 0.05 to 0.13), and LightGBM is no worse than the baselines at all three levels; TSB is only 0.007 behind LightGBM at item level but twice it at store level, showing that its errors do not cancel well within a store. This does not change the cost conclusion (replenishment is executed per SKU), but it has direct implications for hierarchical reconciliation: reconciliation can fix systematic node-level bias, and cannot fix errors that are independent across items.


## 4.2　Cost: the model gap widens with the cost ratio

At service level 0.95, cost per unit of demand (four-window mean, cost ratios 1:1 / 2:1 / 5:1 / 9:1): LightGBM 0.603, 0.626, 0.694, 0.785; TSB 0.691, 0.738, 0.879, 1.068; exponential smoothing 0.707, 0.760, 0.920, 1.133; seasonal naive 0.827, 0.886, 1.062, 1.297. LightGBM is lowest under all four cost ratios, 12.7%, 15.2%, 21.0% and 26.4% below TSB and 14.7%, 17.7%, 24.5% and 30.7% below exponential smoothing.

The gradient shows that the heavier the stockout penalty, the more the quantile bandwidth matters. LightGBM's advantage does not come from point forecasts (only 0.0007 ahead at item level) but from a quantile shape that fits the demand distribution better; at the same level the fill rate of the four models is 0.982 to 0.985 (equal-weight per-series average), so the cost gap is not bought by sacrificing service.


## 4.3　Coverage: the only systematic weakness

Realised coverage at nominal 0.95: exponential smoothing 0.9414, TSB 0.9434, seasonal naive 0.9373, LightGBM 0.9231, maximum deviation 0.0318. Of the 368 rows across the 23 levels, 39 deviate by more than 0.05, the largest being 0.0933, concentrated at low quantiles; the LightGBM deviation has a consistent sign (realised coverage too low, i.e. the band is too narrow), and reaching 0.94 coverage requires raising τ to 0.9617.

There are three layers of cause: demand intermittency and unstable empirical quantiles in small samples; the additive-error assumption being tied to sales scale, which squeezes the quantiles of high-volume items; and LightGBM's training objective leaning towards the median, with sparse samples in the upper tail, producing a systematic downward bias near 0.95.

Four remedies are available: calibrating residuals by quantile level, conservatively widening high quantiles, calibrating by sales-scale stratum, or switching to the reconciliation interface — at the cost, respectively, of giving up part of the quantile shape, higher inventory, needing more historical windows, and depending on another team's deliverable. None of them was used to change the forecast-side quantiles; instead, as agreed with Module 02, an equivalent mapping is applied on the evaluation side: after interpolation to nominal 0.95, at 5:1 the costs are LightGBM 0.794, TSB 0.941, exponential smoothing 0.997 and seasonal naive 1.171. The main line still reports raw quantiles and no rerun on the forecasting side is needed.


## 4.4　Decision map: the segment-optimal model is not unique

Splitting along three dimensions — intermittency, life cycle and sales scale — at the 5:1, 0.95 slice the five intermittency segments fall to: lumpy TSB 258,263.72, intermittent LightGBM 1,674,658.0, lumpy high-variance TSB 621,866.1, no-demand LightGBM 11,222.34, smooth LightGBM 685,822.05. Segmented model choice holds, but the split appears only between the lumpy and intermittent types.

The three-dimension saving against the simple scheme is 15.86%, 31.61% and 26.21%, equal weight 24.56% (average within each dimension first, then equal-weight the dimensions; averaging the 14 segments directly gives 24.44%, and the two must not be mixed). In none of the 1,288 rows is the optimal cost higher than the simple or reference scheme.


## 4.5　Uncertainty and the equal-coverage check

Paired bootstrap (against the seasonal naive baseline, 5:1, 0.95, B=200, in units of the four-window cost difference per series, Co=1): exponential smoothing −22.19 (−22.92 to −21.41), TSB −28.55 (−29.60 to −27.51), LightGBM −57.48 (−59.45 to −55.39). None of the three intervals crosses zero, and the LightGBM interval does not overlap the other two.

The equal-coverage check recalibrates quantiles to realised coverage of 0.92, 0.94 and 0.96 before comparing: at 5:1 LightGBM is still lowest throughout (0.688, 0.745, 0.843), against TSB 0.793, 0.865, 1.016 and exponential smoothing 0.837, 0.913, 1.081, the lead narrowing to 13% to 22%; at 1:1 and the low-coverage end TSB is marginally better (0.569 against 0.594). Interpolating the three points to nominal 0.95: LightGBM 0.9689 with 0.794, TSB 0.9558 with 0.941, exponential smoothing 0.9572 with 0.997, seasonal naive 0.9597 with 1.171, a lead of 15.6%, 20.3% and 32.2%; the price is that LightGBM's own cost rises 14.4% from 0.694.


## 4.6　Extending the quantile grid and the selection rule

The theoretically optimal quantile equals the stockout cost divided by stockout plus holding: 0.50 at 1:1, 0.67 at 2:1, 0.83 at 5:1 and 0.90 at 9:1, the first two falling outside the task card's 0.80 to 0.99 grid. This version extends the grid down to 0.50 and adds 2/3 and 5/6; the three new points are taken from native Module 02 values, so no new data are required.

After the extension the optimal quantile at 1:1 and 2:1 lands on 0.50 and 0.6667, with zero deviation from theory and no longer "the lower edge of the grid"; at 5:1 it is TSB 0.85, exponential smoothing 0.86, LightGBM 0.86 and seasonal naive 0.87, and at 9:1 it is 0.91 to 0.92, all slightly above the theoretical 0.83 and 0.90 — the quantile band is too narrow and needs extra buffer to hold stockouts down. The headline 0.95 numbers are digit-for-digit the same as before the extension.

This is one deliberate deviation from the task card parameters (the original grid 0.80 to 0.99): without the extension, 1:1 and 2:1 could only be described as "the measured optimum is the lower edge of the grid", and the conclusions for those two cost ratios would lose their meaning; with it, all four cost ratios have a genuine optimum and the headline is unchanged.


## 4.7　Verification of the hierarchy reconciliation panel and the order granularity comparison

Three checks on the Module 03 handoff package. First, point forecasts were indeed reconciled bottom-up: the top-level 28-day cumulative total moves from 1,146,624 to 1,189,844 (+3.8%), exactly equal to the sum of the 30,490 leaf point forecasts. Second, the 28-day cumulative quantile block is passed through unchanged under `bottom_up`: its panel is identical row by row to the unreconciled one, 701,523 rows at top, store and leaf level in total with zero difference — the scheme's definition, not a missing step. Third, the quantile panel does not add up: store node CA_1 is 106,884 against a leaf sum of 124,569, 14.2% low; across the whole system τ=0.95 is 35.4% low, with a maximum of 49.8%.

This follows from quantiles not being additive: the sum of the 95% quantiles of 30,490 series is "the stock needed if every series is stocked at 95%", not "the 95% quantile of total demand", and the gap widens as the level rises — only 2.6% at τ=0.50, 35.4% at 0.95. Store-level quantiles are systematically high while the total level moves the other way; node-level coverage and stock multiples are detailed in 02_outputs_final/supplement/hierarchy_recon_summary.json, and the high-level quantile shape is too wide, with the median too low and the tail too high. These limitations apply only to the high-level cumulative quantile convention; cost, coverage and the decision map all use item-store leaf-level quantiles, so the headline conclusions are unaffected.

The order granularity comparison reuses the same simulation core and leaves the main results unchanged. A orders per SKU (the current headline); B lets the store set a total and spreads it by SKU point-forecast share. At 5:1, 0.95, four windows pooled: cost A 0.6943, B 0.8620 (+24.1%); the fill rate pooled by total demand is A 0.9772 against B 0.9453 (3.2 percentage points lower; as an equal-weight per-series average A is 0.9824, and the two conventions must not be mixed); order units fall from 7.28 million to 7.07 million while closing inventory rises from 675,000 to 701,000 — "the total is saved, the structure is wasted". The store-level 95% quantile is itself adequate; the problem lies in the allocation: fast movers are overstocked and slow movers understocked, so stockouts and leftovers appear at the same time.

This set is supplementary, not a replacement: the 17 official artifacts are unchanged and the conclusions still rest on SKU-level ordering. Two limitations: only LightGBM has 28-day cumulative quantiles on both sides, so the three baselines are not included; and mode B allocates by point-forecast share only, so 24.1% is the result under the "most naive allocation", not a lower bound for that convention.

One further live call was made as agreed: the Module 03 dependencies were installed locally, its catalogue was rebuilt from the hierarchy mapping (42,840 nodes, 365,880 membership relations), and its quantile reconciliation function (bottom-up) was called with this side's 23 quantiles for 30,490 series as input. Three results: the output matches this side's leaf sums with a maximum difference of zero and the leaf values are unchanged, so the SKU-level cost, coverage and decision map need no rerun; the panel's top level also equals the leaf sum, at 1,826,608.6 in the test window at τ=0.95 (1.4829 times the realised 1,231,764) and 1,140,163.2 at τ=0.50 (0.9256 times); which shows that the interface only guarantees nodes and leaves add up and cannot fix the systematically high store-level quantiles, so high-level diagnostics still use the point-forecast convention.


# 5　Review and reproducibility


## 5.1　Results of the independent recomputation

The review calls no module-internal function and recomputes everything from the delivered files. The sixteen WAPE values differ from the archived ones by at most 4.441×10⁻¹⁶; LightGBM's 0.95 coverage across the four windows is 0.918203, 0.925877, 0.928272, 0.920007, differing from the report by 1.11×10⁻¹⁶, with a mean of 0.9231 in agreement.

On the cost side the maximum relative error of the column-wise totals against the inventory summary is 3.96×10⁻¹⁶; the three identity checks err by 4.08×10⁻¹⁶, 4.07×10⁻¹⁶ and 2.89×10⁻¹⁴. Rerunning the bootstrap with the same seed reproduces the point estimates exactly, with interval endpoints differing by no more than 0.392 (about 1%, from the order in which random numbers are consumed).

All fifteen Module 02 file fingerprints match; the 11 acceptance checks pass; none of the 1,288 decision-map rows shows "switching model costs more", and there are no duplicate unique keys. In another independent run, 9 data files and the sign-off file are byte-identical, with only the 6 timestamped metadata files differing.


## 5.2　Reproduction tolerance

The bootstrap interval endpoints are not reproduced digit for digit (maximum 0.392, about 1%): the point estimates match while the endpoints shift. This does not affect the direction (all three intervals are far from zero), but the strict statement should be "same order of magnitude, same conclusion"; digit-for-digit agreement holds only for the point estimates.


# 6　Conclusions and boundaries


## 6.1　Conclusions that hold now

| Item (headline convention) | Seasonal naive | ETS | TSB | LightGBM |
|---|---|---|---|---|
| Cost per unit of demand (τ=0.95, 5:1) | 1.062 | 0.920 | 0.879 | 0.694 |
| Item-level WAPE | 0.8752 | 0.7406 | 0.7466 | 0.7399 |
| Realised coverage (τ=0.95) | 0.937 | 0.941 | 0.943 | 0.923 |
| Equal-coverage cost (0.94, 5:1) | 1.080 | 0.913 | 0.865 | 0.745 |
| Optimal service level τ* (cost ratio 5:1) | 0.87 | 0.86 | 0.85 | 0.86 |

First, the conventions are closed and reproducible. Cost can be fully recomputed from the raw forecasts plus the parameter table, the independent recomputation error is at floating-point level, and any cost figure can be checked by a third party.

Second, the cost ranking is determined by quantile quality, not by point-forecast accuracy. The three modern models differ by only 0.007 at item level (0.135 against seasonal naive), yet their cost per unit of demand differs by 12.7% to 30.7%, and the gap comes from quantile bandwidth and shape.

Third, LightGBM is lowest under all four cost ratios at nominal 0.95, but the advantage is not independent of coverage: once coverage is calibrated to 0.92, it still leads by 13% to 28% at 5:1 and 9:1, while at 1:1 and the low-coverage end TSB is 4.3% lower; the advantage grows as the stockout penalty rises.

Fourth, segmented model choice holds but only for some demand types: lumpy goes to TSB, intermittent to LightGBM, and smooth and no-demand still to LightGBM; the three-dimension equal-weight saving is 24.56%.

Fifth, coverage is a known weakness and has been quantified: 39 of 368 rows deviate beyond tolerance, concentrated at low quantiles, and LightGBM needs τ=0.9617 to reach 0.94 coverage; the handling path and its cost are recorded.

Sixth, cost is highly sensitive to the cost ratio, so any decision must be tied to a cost structure. From 1:1 to 9:1, LightGBM's cost rises from 0.603 to 0.785, its gap to TSB widens from 12.7% to 26.4%, and the optimal service level moves from 0.50 to 0.92; the heavier the stockout penalty, the more quantile quality is worth.

Seventh, the optimal service level follows directly from the cost ratio. The theoretical value equals stockout cost divided by stockout plus holding; measured, 1:1 and 2:1 land exactly on 0.50 and 0.6667, 5:1 gives 0.85 to 0.87 and 9:1 gives 0.91 to 0.92, so replenishment parameters need not depend on modelling detail.

Eighth, quantiles are not additive, so hierarchical conclusions stop at leaf level. At τ=0.95 the top level is 35.4% below the leaf sum, store CA_1 14.2% below, and node gaps reach 49.8%; the interface call only guarantees that sums are self-consistent; high-level comparisons switch to the point-forecast convention and the quantile convention is not extrapolated to department stores and state level; that panel does not enter the headline results.

Ninth, order granularity affects the cost structure, not the total volume. With the store setting only a total and spreading it to SKUs, cost is 24.1% higher and the fill rate 3.2 percentage points lower, while order quantity is 2.9% lower and closing inventory 3.8% higher: the total is sufficient, the composition is mismatched.

Tenth, reconciling point forecasts changes hierarchical consistency, not item-level conclusions. After the live interface call the leaf values are unchanged digit for digit with a maximum difference of zero, showing that this part's cost, coverage and decision-map conclusions still hold once reconciliation is complete.


## 6.2　Boundaries of the conclusions

There are five boundaries. First, the sample is all 30,490 series of the M5 dataset (CA 4 stores, TX 3, WI 3): complete coverage, but not representative of broader data sources. Second, cost is normalised per unit of demand, so absolute amounts vary with sample size and only cross-group percentages are readable. Third, the period-cumulative quantiles of the three baselines are constructed inside this module; if Module 02 supplies official quantiles the baseline costs will change and the lead may narrow. Fourth, the Module 03 panel is in place but its cumulative quantiles are unreconciled, so hierarchical quantile conclusions stop at leaf level and department stores and state level use the point-forecast convention only; reconciling that panel belongs to the Module 03 deliverable, this module has avoided it by working at leaf level, and it is registered as a handoff item (see Section 7.2). Fifth, daily quantiles offer only four common levels and the order granularity comparison covers LightGBM only, so those two conclusions have limited coverage.


## 6.3　Implications for model choice and further work

On model choice: if replenishment is executed per SKU and the stockout penalty lies between 1:1 and 9:1, LightGBM is the default; only the lumpy demand segment is cheaper with TSB, and the gain is concentrated in a few types, so a wholesale switch is not advisable.

On parameters: the service level can be set directly from the cost ratio — 0.50 when holding and stockout cost the same, 0.6667 when stockout costs twice holding, 0.86 at five times and 0.92 at nine times, with no need to default to 0.95 by rule of thumb.

On priorities: first fix the quantile bandwidth — LightGBM's coverage shortfall is 0.027, and reaching equivalent 0.95 coverage costs a 14.4% uplift that must enter any benefit comparison; then consider a finer quantile grid; hierarchical reconciliation is worth the effort only where node-level bias is genuinely systematic (TSB's store-level WAPE being twice LightGBM's is such a case), and random item-level error lies outside its reach.


# 7　Handoff notes


## 7.1　Artifacts handed to Module 05

The 17 official artifacts are 9 result tables (accuracy, inventory summary and per-window summary, coverage, decision map, bootstrap, equal coverage, quantile selection, input audit) + 1 per-series cost base table (large, not shipped, regenerable by the main script) + 4 metadata files (simulation parameters, handoff summary, replay log, sign-off) + 3 figures; the run report and the artifact manifest are stored separately in the json directory and are not counted among the 17. There are also 8 supplementary items, all in 02_outputs_final/supplement: the period-level 23-point pinball table and its self-check, the hierarchy check conclusions, the 276-row high-level quantile calibration, the 184-row order granularity comparison, the 23-row quantile additivity gap, the 7-row interface check table and the explanatory notes; the supplementary figure fig_order_granularity.png sits in figures and, like these 8 items, is supplementary and outside the manifest and hash.

Four things downstream users should note: costs are normalised per unit of demand; apart from the accuracy metrics and the supplementary high-level calibration, cost, coverage, equal coverage, bootstrap, quantile selection and the decision map are all item-level, so they must not be cited as three levels; quantile loss must be reported under the right convention — at daily level only four common levels are comparable and LightGBM has only those four points, while the 23-point cumulative figures are in the supplementary table; the three figures must always be cited together with the parameter table.


## 7.2　Open threads back to Modules 02 and 03

Back to Module 02: daily quantiles exist at only four levels and only LightGBM is non-empty, which limits the coverage of daily quantile loss; LightGBM's high quantiles are too narrow and reaching 0.94 coverage needs τ=0.9617, caused by a median-leaning training objective and sparse tail samples.

Back to Module 03: cumulative quantiles are unreconciled, `bottom_up` matches the unreconciled panel row by row and the top level is 35.4% below the leaf sum, so comparing MinT with bottom-up requires a genuinely reconciled quantile version; the interface was called live under option A and its output matches the leaf sums digit for digit, so the convention gap is resolved; aggregate-level WAPE is far below item level (total 0.0475 against 0.7399), item errors largely cancel when summed, and the gain from hierarchical reconciliation can only come from systematic node-level bias.
