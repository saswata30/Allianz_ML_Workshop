# Databricks notebook source
# MAGIC %md
# MAGIC # 07 · MLOps — bundles, jobs and the model lifecycle
# MAGIC
# MAGIC **Platform layers:** MLOps — *Declarative Automation Bundles (DABs) and MLOps Stacks* · *MLflow*
# MAGIC
# MAGIC Everything you built by hand today is also defined **as code** in this repo, so it can be reviewed in Git
# MAGIC and promoted dev → test → prod by CI/CD:
# MAGIC
# MAGIC | File | Defines |
# MAGIC |---|---|
# MAGIC | `databricks.yml` | The bundle: name, variables (`catalog`, `schema`) and `dev` / `prod` targets |
# MAGIC | `resources/ml_jobs.yml` | **Training job** (features → train & register) and a nightly **batch-scoring job** |
# MAGIC | `resources/feature_pipeline.yml` | A serverless **Lakeflow Declarative Pipeline** that refreshes cleaned cases with data-quality expectations |
# MAGIC | `pipelines/case_features_pipeline.sql` | The pipeline's SQL |
# MAGIC
# MAGIC **Time:** 20 minutes (demo + discussion).

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 1 · Deploy the bundle (from a terminal or the workspace)
# MAGIC
# MAGIC **From your laptop** (Databricks CLI ≥ 0.250):
# MAGIC ```bash
# MAGIC git clone https://github.com/saswata30/Allianz_ML_Workshop.git && cd Allianz_ML_Workshop
# MAGIC databricks auth login --host https://<your-workspace>.azuredatabricks.net
# MAGIC databricks bundle validate
# MAGIC databricks bundle deploy -t dev --var="schema=ml_<your_user>"
# MAGIC databricks bundle run case_escalation_training -t dev --var="schema=ml_<your_user>"
# MAGIC ```
# MAGIC
# MAGIC **From the workspace:** open this Git folder → the **Deployments** (rocket) icon → **Deploy** to `dev`.
# MAGIC
# MAGIC > 🧞 **Genie Code prompt**
# MAGIC > ```
# MAGIC > Explain what the bundle in this Git folder deploys. Read databricks.yml and the resources folder,
# MAGIC > and list each job, its tasks and schedule, and the pipeline.
# MAGIC > ```

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 2 · The champion / challenger lifecycle
# MAGIC
# MAGIC The training job registers a **new version** each run. Promotion is a deliberate step:
# MAGIC
# MAGIC 1. The new version gets alias `@challenger`.
# MAGIC 2. An evaluation task compares challenger and champion on the same holdout (`mlflow.models.evaluate` or the
# MAGIC    metrics logged in module 03).
# MAGIC 3. If the challenger wins **and** a reviewer approves, move `@champion` to it. Batch scoring (module 04) and
# MAGIC    the endpoint (module 05) pick it up — batch scoring immediately, the endpoint on its next config update.
# MAGIC
# MAGIC Run the cell below to see the aliases on your model.

# COMMAND ----------

# MAGIC %run ./_config

# COMMAND ----------

import mlflow
from mlflow import MlflowClient

mlflow.set_registry_uri("databricks-uc")
client = MlflowClient()
for mv in client.search_model_versions(f"name = '{MODEL_NAME}'"):
    info = client.get_model_version(MODEL_NAME, mv.version)
    print(f"v{mv.version:>3}  aliases={info.aliases}  run={mv.run_id}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 3 · Start from MLOps Stacks for a new use case
# MAGIC
# MAGIC For the next ADP use case, generate a production-ready project skeleton (CI/CD, tests, staging/prod targets,
# MAGIC monitoring) rather than starting from scratch:
# MAGIC ```bash
# MAGIC databricks bundle init mlops-stacks
# MAGIC ```
# MAGIC Choose Azure DevOps or GitHub Actions for CI/CD, and Unity Catalog for the model registry.
# MAGIC
# MAGIC For an agent-first start, the **MLOps Quickstart** reference repo
# MAGIC ([databricks-solutions/mlops-quickstart](https://github.com/databricks-solutions/mlops-quickstart)) ships with
# MAGIC **Genie Code skills**, so Genie Code can build governed tables, tracked models and a serving endpoint
# MAGIC following the same pattern you used today.
# MAGIC
# MAGIC ## ✅ Checkpoint & wrap-up
# MAGIC You have walked every layer of the Databricks Data Science & ML platform:
# MAGIC
# MAGIC | Layer | What you used |
# MAGIC |---|---|
# MAGIC | EDA & development | Genie Code for ML, notebooks |
# MAGIC | Data | Feature Store (`case_features`); Lakeflow pipeline in the bundle |
# MAGIC | Training | Serverless compute, MLflow tracking |
# MAGIC | Serving | Batch inference, Model Serving, Feature Serving (stretch) |
# MAGIC | Monitoring | Inference tables, Data Quality Monitoring |
# MAGIC | MLOps | Bundles, jobs, MLflow aliases, MLOps Stacks |
# MAGIC | Governance & storage | Unity Catalog, Delta Lake |
