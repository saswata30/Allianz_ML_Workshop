# Databricks notebook source
# MAGIC %md
# MAGIC # 07 · MLOps — jobs, a Lakeflow pipeline and the model lifecycle
# MAGIC
# MAGIC **Platform layers:** MLOps — *Lakeflow Jobs* · *MLflow* · Data — *Lakeflow Declarative Pipelines*
# MAGIC
# MAGIC So far you ran every step by hand. In production the same notebooks run unattended. In this module you
# MAGIC create, **from this notebook with the Databricks SDK** (or by clicking through the UI):
# MAGIC
# MAGIC | Resource | What it does |
# MAGIC |---|---|
# MAGIC | **Training job** `case-escalation-training-<you>` | `02_feature_engineering` → `03_train_and_track`: refreshes features, retrains, registers a new version (promoted to `@champion` only if it is better) |
# MAGIC | **Batch-scoring job** `case-escalation-scoring-<you>` | `04_batch_inference` every morning at 05:00 (created paused) |
# MAGIC | **Lakeflow pipeline** `case-features-pipeline-<you>` | `pipelines/case_features_pipeline.sql`: cleaned cases and features with data-quality expectations, on serverless |
# MAGIC
# MAGIC Everything stays in this Git folder, so changes to the notebooks are reviewed in Git and the jobs always run
# MAGIC the committed code.
# MAGIC
# MAGIC **Time:** 20 minutes.
# MAGIC
# MAGIC ---
# MAGIC ### 🗺️ How to follow this notebook
# MAGIC * **Compute:** at the top right, attach **Serverless**.
# MAGIC * **Run cells one at a time, top to bottom** — click inside a cell and press `Shift + Enter`.
# MAGIC * **Symbols:** 🧞 = ask Genie Code first · ▶ = run the next cell · 👀 = what you should see · ✅ = check before moving on.
# MAGIC * **Stuck?** Run the cell marked `# Reference` under each prompt — it always works, so you can never fall behind.
# MAGIC
# MAGIC ### Your path (20 minutes)
# MAGIC 1. ▶ Run the config cell and the setup cell → 👀 the folder path of these notebooks.
# MAGIC 2. Step 1: create the two jobs → 👀 two links: the training job and the (paused) scoring job.
# MAGIC 3. Step 2: create and start the Lakeflow pipeline → 👀 a pipeline link. Open it to see the graph and the data-quality results.
# MAGIC 4. Step 3: start a training run → 👀 a run link. Wait about 5 minutes, then run the next cell → 👀 version 2 with alias `challenger` (or `champion` if it scored better).
# MAGIC 5. Step 4: read how drift, promotion and code changes close the loop.
# MAGIC 6. ✅ **Jobs & Pipelines** in the left bar shows your two jobs and one pipeline.

# COMMAND ----------

# MAGIC %run ./_config

# COMMAND ----------

import os
from databricks.sdk import WorkspaceClient
from databricks.sdk.service import jobs, pipelines

w = WorkspaceClient()

NOTEBOOK_DIR = os.path.dirname(
    dbutils.notebook.entry_point.getDbutils().notebook().getContext().notebookPath().get())
REPO_ROOT = os.path.dirname(NOTEBOOK_DIR)
PARAMS = {"catalog": CATALOG, "schema": SCHEMA}
SUFFIX = USER_TAG.replace("_", "-")

print("Notebooks:", NOTEBOOK_DIR)
print("Parameters:", PARAMS)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 1 · Create the training and scoring jobs
# MAGIC
# MAGIC > 🧞 **Genie Code prompt**
# MAGIC > ```
# MAGIC > Using the Databricks SDK, create a job named case-escalation-training-<my user> with two notebook
# MAGIC > tasks in this folder: 02_feature_engineering, then 03_train_and_track, both with parameters
# MAGIC > catalog and schema. Use serverless compute. If a job with that name exists, update it instead.
# MAGIC > ```
# MAGIC
# MAGIC ▶ **Next:** if you used Genie Code and it worked, skip ahead. Otherwise run the `# Reference` cell below.
# MAGIC
# MAGIC **UI alternative:** **Jobs & Pipelines → Create → Job**. Add a *Notebook* task for `02_feature_engineering`,
# MAGIC then a second task for `03_train_and_track` that depends on it. Set the parameters `catalog` and `schema`,
# MAGIC and leave compute as **Serverless**. Create the scoring job the same way with `04_batch_inference` and a schedule.

# COMMAND ----------

# Reference
def notebook_task(key, notebook, depends_on=()):
    return jobs.Task(
        task_key=key,
        notebook_task=jobs.NotebookTask(notebook_path=f"{NOTEBOOK_DIR}/{notebook}", base_parameters=PARAMS),
        depends_on=[jobs.TaskDependency(task_key=d) for d in depends_on],
    )

def upsert_job(name, tasks, schedule=None, description=""):
    """Create the job, or overwrite its settings if a job with this name already exists."""
    common = dict(name=name, tasks=tasks, schedule=schedule, description=description,
                  max_concurrent_runs=1, tags={"workshop": "allianz_ml"})
    existing = next(iter(w.jobs.list(name=name)), None)
    if existing:
        w.jobs.reset(job_id=existing.job_id, new_settings=jobs.JobSettings(**common))
        return existing.job_id
    return w.jobs.create(**common).job_id

training_job_id = upsert_job(
    f"case-escalation-training-{SUFFIX}",
    tasks=[notebook_task("features", "02_feature_engineering"),
           notebook_task("train_register", "03_train_and_track", depends_on=["features"])],
    description="Refresh features, retrain and register the case-escalation model (promotes only if it beats @champion).",
)

scoring_job_id = upsert_job(
    f"case-escalation-scoring-{SUFFIX}",
    tasks=[notebook_task("score_open_cases", "04_batch_inference")],
    schedule=jobs.CronSchedule(quartz_cron_expression="0 0 5 * * ?", timezone_id="America/Chicago",
                               pause_status=jobs.PauseStatus.PAUSED),
    description="Nightly scoring of open cases with the @champion model.",
)

print(f"Training job: {w.config.host}/jobs/{training_job_id}")
print(f"Scoring job : {w.config.host}/jobs/{scoring_job_id}  (schedule paused — unpause it in the UI when ready)")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 2 · Create the Lakeflow Declarative Pipeline
# MAGIC
# MAGIC The pipeline turns `cases` into `cases_clean` (rows that fail the expectations are dropped and counted) and
# MAGIC `case_features_pipeline` (the same feature logic as module 02, in SQL).
# MAGIC
# MAGIC **UI alternative:** **Jobs & Pipelines → Create → ETL pipeline**. Choose your catalog and schema, select
# MAGIC **Serverless**, add the source file `pipelines/case_features_pipeline.sql` from this Git folder, and add the
# MAGIC configuration key `source_cases` = your `cases` table.

# COMMAND ----------

# Reference
PIPELINE_NAME = f"case-features-pipeline-{SUFFIX}"
pipeline_spec = dict(
    name=PIPELINE_NAME,
    catalog=CATALOG,
    serverless=True,
    configuration={"source_cases": CASES_TABLE},
    libraries=[pipelines.PipelineLibrary(file=pipelines.FileLibrary(
        path=f"{REPO_ROOT}/pipelines/case_features_pipeline.sql"))],
)

existing = next(iter(w.pipelines.list_pipelines(filter=f"name LIKE '{PIPELINE_NAME}'")), None)
if existing:
    pipeline_id = existing.pipeline_id
    print("Pipeline already exists.")
else:
    try:
        pipeline_id = w.pipelines.create(schema=SCHEMA, **pipeline_spec).pipeline_id
    except TypeError:  # older SDK versions call the target schema `target`
        pipeline_id = w.pipelines.create(target=SCHEMA, **pipeline_spec).pipeline_id

update = w.pipelines.start_update(pipeline_id=pipeline_id)
print(f"Pipeline: {w.config.host}/pipelines/{pipeline_id}  (update {update.update_id} started)")

# COMMAND ----------

# MAGIC %md
# MAGIC Open the pipeline link above: the graph shows `cases_clean → case_features_pipeline`, and the
# MAGIC **Data quality** tab shows how many rows passed each expectation.

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 3 · Retrain on demand and watch the lifecycle
# MAGIC
# MAGIC Run the training job once. Because module 03 only moves `@champion` when the new version is better, a
# MAGIC retrain on the same data registers a new version as `@challenger` and leaves the champion in place.

# COMMAND ----------

run = w.jobs.run_now(job_id=training_job_id)
print(f"Training run started: {w.config.host}/jobs/{training_job_id}/runs/{run.run_id}")
print("Takes about 5 minutes. Re-run the next cell when it finishes.")

# COMMAND ----------

import mlflow
from mlflow import MlflowClient

mlflow.set_registry_uri("databricks-uc")
client = MlflowClient()
for mv in sorted(client.search_model_versions(f"name = '{MODEL_NAME}'"), key=lambda m: int(m.version)):
    info = client.get_model_version(MODEL_NAME, mv.version)
    auc = client.get_run(mv.run_id).data.metrics.get("test_roc_auc")
    print(f"v{mv.version:>3}  aliases={info.aliases}  test_roc_auc={auc:.3f}" if auc else f"v{mv.version:>3}  aliases={info.aliases}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 4 · Close the loop
# MAGIC
# MAGIC * **Drift → retrain:** edit the SQL alert from module 06 and add the training job as the alert's action
# MAGIC   (or a job *trigger* on new data in `cases`), so drift starts a retrain automatically.
# MAGIC * **Promotion:** a reviewer compares `@challenger` with `@champion` in the MLflow UI and moves the alias.
# MAGIC   Batch scoring picks up the new champion on its next run; update the serving endpoint (module 05, Step 1) to
# MAGIC   roll it out to real-time traffic.
# MAGIC * **Code changes:** edit notebooks in this Git folder, commit and open a pull request. The jobs run whatever
# MAGIC   is on the branch the Git folder is checked out to.
# MAGIC
# MAGIC > 🧞 **Genie Code prompt**
# MAGIC > ```
# MAGIC > Show the last 5 runs of my case-escalation-training job with their status and duration, and tell me
# MAGIC > which model version each run registered.
# MAGIC > ```
# MAGIC
# MAGIC ▶ **Next:** if you used Genie Code and it worked, skip ahead. Otherwise run the `# Reference` cell below.
# MAGIC
# MAGIC ## ✅ Checkpoint & wrap-up
# MAGIC **Jobs & Pipelines** shows your two jobs and the pipeline. You have walked every layer of the Databricks
# MAGIC Data Science & ML platform:
# MAGIC
# MAGIC | Layer | What you used |
# MAGIC |---|---|
# MAGIC | EDA & development | Genie Code for ML, notebooks, Git folders |
# MAGIC | Data | Feature Store (`case_features`), Lakeflow Declarative Pipeline |
# MAGIC | Training | Serverless compute, MLflow tracking |
# MAGIC | Serving | Batch inference, Model Serving, Feature Serving (stretch) |
# MAGIC | Monitoring | Inference tables, Data Quality Monitoring, SQL alerts |
# MAGIC | MLOps | Lakeflow Jobs, MLflow aliases (champion / challenger) |
# MAGIC | Governance & storage | Unity Catalog, Delta Lake |
# MAGIC
# MAGIC ## Clean up (after the workshop)
# MAGIC Uncomment and run to delete the jobs and pipeline you created here.

# COMMAND ----------

# w.jobs.delete(job_id=training_job_id)
# w.jobs.delete(job_id=scoring_job_id)
# w.pipelines.delete(pipeline_id=pipeline_id)
