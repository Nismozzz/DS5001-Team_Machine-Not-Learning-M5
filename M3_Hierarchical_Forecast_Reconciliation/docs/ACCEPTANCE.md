# Acceptance and handoff checklist

## Technical acceptance

- [x] 12 levels, 42,840 nodes and 30,490 leaf series map consistently.
- [x] The four schemes cover 4 models, a 28-day horizon and five forecast scenarios.
- [x] Point forecasts and direct cumulative quantiles are handled in separate pipelines.
- [x] Hierarchical summation error is zero for every reconciled scheme.
- [x] All output point forecasts are non-negative.
- [x] LightGBM daily quantiles and 28-day cumulative quantiles are all monotone.
- [x] Every MinT point and quantile solve converged.
- [x] Independent recomputation passes: 2,160 prediction Parquet files, 2,192 manifest records.
- [x] A unified data contract, method notes, representative samples, runtimes and unit tests are provided.
- [x] An interface for reconciling external baseline quantiles is provided.

## Handoff acceptance

| Item | Owner | Status |
|---|---|---|
| Hierarchy mapping and method notes | Module 3 | Done |
| H2 schema, origins and sample coverage | Module 2 / Module 3 | Rebuilt from the inputs and verified |
| Hierarchical coherence | Module 3 | Done |
| Forecast usability and method sign-off | Module 4 | Signed off in Module 4 (§4.7) |
| Inventory input interface | Module 3 / Module 4 | Provided; live call verified in Module 4 (§4.7), additivity error 0 |
| Final number freeze | Module 5 | Outside this handoff; the frozen numbers are those in this package |

The signature block should record only roles and dates within the course project; if the repository is public, do not add student EIDs or private email addresses to public documents.

## Before release

- [x] No raw Kaggle data.
- [x] No 901 MB full prediction tree.
- [x] No `__pycache__`, `.pyc` or `.DS_Store`.
- [x] No local absolute paths, names, EIDs or private email addresses.
- [x] Publication constraints are recorded in `docs/SECURITY_AND_PUBLICATION.md`; no personal contact details appear in the package documents.
- [x] Commit author addresses sit in the Git history only; they are not part of the package content.


