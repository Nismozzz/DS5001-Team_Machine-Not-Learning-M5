# DS5001 · Machine Not Learning

This project uses the M5 dataset to study hierarchical demand forecasting and inventory decisions across 30,490 item–store series and 12 aggregation levels.

## Team Contributions

| Member | Module | Main contributions | Details |
| --- | --- | --- | --- |
| Xiaofeng ZHU | 01 · Data Processing and Feature Engineering | Locked and audited the raw data, built a 5,000-series development set and a full dataset, and delivered rolling splits, features computed at each forecast origin, separate target labels, 12-level hierarchy mappings, and demand segments. Completed independent local validation. | [Module 01 documentation and reproduction steps](https://github.com/Nismozzz/DS5001-Team_Machine-Not-Learning-M5/blob/d8806ce49241b8bde6e296488521cdf333404625/README.md) |
| Jiaheng | 02 · Demand Forecasting | Implemented Seasonal Naive, ETS, TSB, and LightGBM. Tuned models on the development set and completed three-fold backtesting on the full dataset, holdout testing, 28-day forecasting, and historical calibration. Delivered point forecasts, daily quantiles, 28-day cumulative demand quantiles, and residuals for periods with observed demand. | [Module 02 handoff guide](Member2_%E5%AE%8C%E6%95%B4%E4%BA%A4%E4%BB%98%E5%8C%85/%E4%BA%A4%E6%8E%A5%E8%AF%B4%E6%98%8E.md) |
| Junze LU | 03 · Hierarchical Forecast Reconciliation | Compared unreconciled forecasts, bottom-up, top-down, and nonnegative MinT across the 12-level M5 hierarchy. Verified that reconciled point forecasts were nonnegative and summed consistently across levels. Delivered reconciliation code, data contracts, and diagnostics. | [Module 03 README](M3_Hierarchical_Forecast_Reconciliation/README.md) |
| Xiang LI | 04 · Evaluation and Inventory Decisions | Compared forecast accuracy and quantile loss across four evaluation windows for the full set of series. Simulated inventory outcomes under different cost ratios and service levels, and built a model selection decision map by demand pattern, sales volume, and product life cycle. | [Module 04 README](M4_Evaluation_and_Inventory_Decision/README.md) |
| Lin Jianfeng | 05 · Reporting and Presentation | Aligned the scope of the Module 03 and 04 documentation and clarified that the report uses item–store quantiles for inventory costs, coverage, and the decision map. | [Module 05 documentation commit](https://github.com/Nismozzz/DS5001-Team_Machine-Not-Learning-M5/commit/1f3c817) |

## Data and Reproduction

Download the raw M5 data from Kaggle and follow the Module 01 reproduction steps to generate inputs for the downstream modules. See the linked module documentation for dependencies, run commands, and deliverables. [M5 dataset on Kaggle](https://www.kaggle.com/competitions/m5-forecasting-accuracy/data)
