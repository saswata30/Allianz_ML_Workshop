# Allianz ML Workshop — the Databricks ML pattern with Genie Code

Hands-on workshop for the **Allianz Data Platform (ADP)** on Azure Databricks. Attendees build a **claim-case
escalation model** end to end, using **Genie Code for ML** as their AI pair-programmer at every step, and
touch every layer of the Databricks Data Science & Machine Learning platform:

| Platform layer | Products | Module |
|---|---|---|
| **EDA & development** | Genie Code for ML · Notebooks & IDE integration | `01`, `01b` |
| **Data** | Lakeflow · Feature Store | `02`, `pipelines/` |
| **Training** | AI Runtime (serverless) · ML Runtime (classic compute) | `03` |
| **Serving** | Batch inference · Model Serving · Feature Serving | `04`, `05` |
| **Monitoring** | Inference tables · Data Quality Monitoring for ML | `05`, `06` |
| **MLOps** | Lakeflow Jobs · MLflow (champion / challenger) | `03`, `07` |
| **Governance & storage** | Unity Catalog · Delta Lake | all modules |

> **Why Genie Code instead of AutoML?** Databricks now positions **Genie Code for ML** for customised model
> development (and AI Functions such as `ai_forecast` for simple cases) in place of AutoML. Every module gives
> attendees a Genie Code prompt first, then a tested reference solution so nobody gets stuck.

---

## Repository layout

```
notebooks/
  _config.py                    shared names (catalog, schema, tables, model, endpoint) — run by every module
  00_setup_data.py              personal schema + synthetic claim cases (60k labelled, 3k open)
  01_eda_genie_code.py          EDA with Genie Code
  01b_genie_code_end_to_end.py  one prompt → explore, features, XGBoost, predictions (mirrors the Databricks demo)
  02_feature_engineering.py     features → Unity Catalog feature table
  03_train_and_track.py         training set, 3 candidates, MLflow tracking, UC registry, @champion
  04_batch_inference.py         Spark-UDF batch scoring → case_predictions
  05_model_serving.py           serverless endpoint + AI Gateway inference table; Genie Code endpoint health
  06_monitoring.py              inference-log monitor, simulated drift, alerting
  07_mlops_jobs.py              creates the training job, nightly scoring job and Lakeflow pipeline (SDK or UI)
pipelines/case_features_pipeline.sql   Lakeflow Declarative Pipeline with expectations
docs/lab-guide.md                       step-by-step attendee instructions with checkpoints
docs/facilitator-guide.html             agenda, prerequisites and platform map for the facilitator
docs/genie-code-instructions.md         suggested Genie Code instructions for the workshop
```

## Before the workshop (admin / facilitator)

1. **Genie Code:** *Partner-powered AI features* enabled for the Azure Databricks account and workspace.
2. **Unity Catalog:** a catalog for the workshop (default name `adp_workshop`). Grant attendees
   `USE CATALOG` and `CREATE SCHEMA` on it — each attendee gets their own schema `ml_<user>`.
3. **Compute:** serverless notebooks and jobs enabled (or a shared cluster on **Databricks Runtime 16.4 LTS ML** or later).
4. **Model Serving** available in the workspace region; attendees may create endpoints.
5. **Data Quality Monitoring** (Lakehouse Monitoring) available.
6. A small **serverless SQL warehouse** for the monitoring dashboard and alerts.
7. Dry-run `00` → `06` once with a test user (about 45 minutes, mostly endpoint creation).

To run on real ADP data instead of the synthetic set, set the `source_table` widget to the curated table
(for example `adp.claims.cases`) — it needs the same columns, including the `escalated` label.

## Attendee quick start

1. In the workspace: **Workspace → Create → Git folder**, URL `https://github.com/saswata30/Allianz_ML_Workshop`.
2. Open `notebooks/00_setup_data`, attach **Serverless**, and **Run all**.
3. Work through the modules in order, following [`docs/lab-guide.md`](docs/lab-guide.md).

## Agenda (3.5 hours)

| Time | Module | Format |
|---|---|---|
| 0:00–0:15 | Welcome, the escalation use case, the platform map | Talk |
| 0:15–0:25 | `00` Setup | Hands-on |
| 0:25–0:50 | `01` EDA with Genie Code | Hands-on |
| 0:50–1:10 | `01b` Genie Code challenge: one prompt to a model | Hands-on |
| 1:10–1:35 | `02` Feature engineering & Feature Store | Hands-on |
| 1:35–1:50 | ☕ Break | |
| 1:50–2:30 | `03` Train, track & register | Hands-on |
| 2:30–2:50 | `04` Batch inference | Hands-on |
| 2:50–3:00 | ☕ Break (endpoint builds) | |
| 3:00–3:15 | `05` Model Serving | Hands-on |
| 3:15–3:30 | `06` Monitoring · `07` Jobs & lifecycle · recap | Demo + discussion |

## Run it as jobs (no bundle or CLI needed)

Module `07_mlops_jobs` creates everything from inside the workspace with the Databricks SDK, one set per attendee:

| Resource | Runs |
|---|---|
| `case-escalation-training-<user>` | `02_feature_engineering` → `03_train_and_track` |
| `case-escalation-scoring-<user>` | `04_batch_inference`, daily 05:00 (created paused) |
| `case-features-pipeline-<user>` | `pipelines/case_features_pipeline.sql` on serverless |

Each step also has a click-through **UI alternative** (Jobs & Pipelines → Create).

## Clean up

```sql
DROP SCHEMA IF EXISTS adp_workshop.ml_<your_user> CASCADE;
```
Then delete your serving endpoint (**Serving** → endpoint → **Delete**), the monitor (table → **Quality** → **Delete monitor**), and the jobs and pipeline (last cell of `07_mlops_jobs`).

## References

* Genie Code for data science — https://learn.microsoft.com/azure/databricks/notebooks/ds-agent
* Model serving observability with Genie Code — https://learn.microsoft.com/azure/databricks/machine-learning/model-serving/model-serving-genie-code
* Feature Store (feature engineering in Unity Catalog) — https://learn.microsoft.com/azure/databricks/machine-learning/feature-store/
* Model Serving — https://learn.microsoft.com/azure/databricks/machine-learning/model-serving/
* Data Quality Monitoring — https://learn.microsoft.com/azure/databricks/lakehouse-monitoring/
* Lakeflow Jobs — https://learn.microsoft.com/azure/databricks/jobs/
* Lakeflow Declarative Pipelines — https://learn.microsoft.com/azure/databricks/dlt/
* Managed MLflow — https://www.databricks.com/product/managed-mlflow
* Demo: Genie Code for machine learning — https://www.youtube.com/watch?v=XxhyoA95pwY

All data in this repo is **synthetic**. No Allianz customer data is included.
