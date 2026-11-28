# Member 2 — Complete Delivery Notes

**The model code, all official predictions and the handover results are complete, and every stage has passed acceptance. This one package is the whole deliverable: unzip it once and everything is ready to use.**

## What the package delivers

- Four model families: Seasonal Naive, ETS, TSB and LightGBM.
- Point forecasts, daily quantiles, 28-day cumulative demand quantiles, and residuals for the stages that have ground truth.
- Development-set tuning, the three-fold full-data backtest, the independent test, the future forecast, and historical calibration under two configurations.
- Full coverage: 30,490 item–store series, 12 levels, 42,840 nodes.

## Results

Full-data independent test, item–store level, 28 days (d_1914–d_1941), point forecasts before reconciliation. Lower is better for both metrics.

| Model | WAPE | MASE |
|---|---:|---:|
| Seasonal Naive | 86.22% | 1.3527 |
| ETS | 72.20% | **1.1974** |
| TSB | 73.39% | 1.2032 |
| LightGBM | **72.09%** | 1.2345 |

LightGBM has the lowest WAPE and sits close to ETS; ETS has the lowest MASE. All four models are therefore kept for the downstream comparison.

## Where to look after unzipping

| Folder | Contents |
|---|---|
| `code/` | models, configurations, dependencies, tests and the metric summaries under `results/member2` |
| `predictions/full/` | the complete predictions and residuals used for the team's official experiments |
| `predictions/development/` | development-set tuning, test, forecast and calibration, for small-scale integration work |

Inside `predictions/full/`:

| Directory | Contents |
|---|---|
| `final_backtest` | the three rolling validation windows |
| `final_test` | the independent test |
| `final_forecast` | the future 28-day forecast; no ground truth, no residuals |
| `final_calibration` | historical calibration of the final selected configuration |
| `final_predeclared_calibration` | historical calibration rebuilt from the fixed original first-candidate configuration |

**Example:** open `predictions/full/final_test/predictions/test/L12/` to see `daily.parquet` (daily forecast), `period.parquet` (cumulative quantiles) and `residuals.parquet` (residuals). `L01`–`L11` are the other levels. These are directly readable data files, and nothing else has to be unzipped.

## How it hands over

- **Member 3:** use `daily.parquet` for the hierarchical reconciliation and `residuals.parquet` to estimate forecast error; residuals are actual minus forecast.
- **Member 4:** use `period.parquet` for LightGBM's 28-day cumulative demand quantiles and continue the inventory experiments.
- **GitHub:** merge the files under `code/` into the root of the team repository and keep Member 1's files; the complete predictions are read straight from this package and are handed over together with it.
- **Re-running:** needs `01_data.py`, `requirements.txt` and `data/processed` from the team project. Create a `.venv-models` environment, install `requirements-models.txt`, and use `./run_models.ps1 --help` for the run options. Existing predictions can be used without re-running anything.

## Notes

- This package delivers the leaf-level, per-window forecasts that the hierarchy step consumes: cross-level reconciliation is Module 3's deliverable, and it rebuilds coherent totals from these numbers. Cumulative-window quantiles are not additive, so they are not obtained by summing daily quantiles — for the three baselines they are constructed by the downstream stage, while LightGBM's `period_quantile` comes directly from the model.
- The final configuration is 63 leaves / 300 rounds; the fixed first-candidate calibration uses 31 leaves / 300 rounds and is an after-the-fact reconstruction. Residuals must be matched to the configuration they come from, and may only use labels that were already known before each forecast origin; the three-fold full-data backtest is a retrospective result measured after the configuration was selected.
- At some higher levels the cumulative quantiles are constant because the sample there is small; tail calibration and the inventory results are verified in the downstream modules. Interface review, sign-off and repository submission are carried out by the team.

This package contains the code and all ten official stage handovers. It does not contain the raw Kaggle data, the local Python environment or trained model weights: the weights stay in Member 2's original run directory, and the team can use the predictions in this package directly. `artifact_manifest.json` records the checksum of every file in the package, and each stage additionally carries its own `handoff_manifest.json` and acceptance report.
