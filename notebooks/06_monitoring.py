# Databricks notebook source
# MAGIC %md
# MAGIC # 06 · Monitoring — Data Quality Monitoring for ML
# MAGIC
# MAGIC **Platform layers:** Monitoring — *Inference tables* · *Data Quality Monitoring for ML*
# MAGIC
# MAGIC A model that was accurate at launch drifts as claims patterns change. You attach an **inference-log monitor**
# MAGIC to the predictions table. Databricks then computes, on a schedule:
# MAGIC
# MAGIC * **Profile metrics** — distributions, nulls and summary statistics for every feature and the prediction.
# MAGIC * **Drift metrics** — how each window compares with the previous one (and with a baseline).
# MAGIC * **Model quality** — ROC AUC, precision, recall and more, wherever the true label has arrived.
# MAGIC
# MAGIC It also creates a **dashboard**, and you can set **SQL alerts** on the metric tables.
# MAGIC
# MAGIC **Time:** 25 minutes.

# COMMAND ----------

# MAGIC %run ./_config

# COMMAND ----------

from databricks.sdk import WorkspaceClient
from databricks.sdk.service.catalog import MonitorInferenceLog, MonitorInferenceLogProblemType
from pyspark.sql import functions as F

w = WorkspaceClient()
ASSETS_DIR = f"/Workspace/Users/{CURRENT_USER}/allianz_ml_workshop_monitoring"

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 1 · Create the monitor
# MAGIC
# MAGIC > 🧞 **Genie Code prompt**
# MAGIC > ```
# MAGIC > Create a Databricks data quality (Lakehouse) monitor on {PREDICTIONS_TABLE} using the inference log
# MAGIC > profile: classification problem, prediction column predicted_escalation, label column escalated,
# MAGIC > model id column model_version, timestamp column scored_at, granularity 1 day. Write the metric
# MAGIC > tables to {CATALOG}.{SCHEMA}.
# MAGIC > ```
# MAGIC
# MAGIC **UI alternative:** Catalog Explorer → `case_predictions` → **Quality** tab → **Get started**.

# COMMAND ----------

# Reference
try:
    monitor = w.quality_monitors.get(table_name=PREDICTIONS_TABLE)
    print("Monitor already exists — refreshing it.")
except Exception:
    monitor = w.quality_monitors.create(
        table_name=PREDICTIONS_TABLE,
        assets_dir=ASSETS_DIR,
        output_schema_name=FQ,
        inference_log=MonitorInferenceLog(
            problem_type=MonitorInferenceLogProblemType.PROBLEM_TYPE_CLASSIFICATION,
            prediction_col="predicted_escalation",
            label_col=LABEL_COL,
            model_id_col="model_version",
            timestamp_col="scored_at",
            granularities=["1 day"],
        ),
        slicing_exprs=["line_of_business", "channel"],
    )
    print("Monitor created. Dashboard:", monitor.dashboard_id)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 2 · Simulate a shift in the claims mix, then refresh
# MAGIC
# MAGIC A storm event brings a wave of large Home claims with more contacts and worse sentiment. We score a batch of
# MAGIC such cases (with a timestamp of tomorrow so it lands in a new window) and refresh the monitor.

# COMMAND ----------

shifted = (spark.table(PREDICTIONS_TABLE).where(F.col(LABEL_COL).isNotNull()).limit(1500)
           .withColumn("line_of_business", F.lit("Home"))
           .withColumn("claim_amount", F.col("claim_amount") * 2.5)
           .withColumn("num_contacts", F.col("num_contacts") + 3)
           .withColumn("sentiment_score", F.greatest(F.col("sentiment_score") - 0.5, F.lit(-1.0)))
           .withColumn("escalation_probability", F.least(F.col("escalation_probability") + 0.15, F.lit(1.0)))
           .withColumn("predicted_escalation", (F.col("escalation_probability") >= 0.5).cast("int"))
           .withColumn("scored_at", F.col("scored_at") + F.expr("INTERVAL 1 DAY"))
           .withColumn(KEY_COL, F.concat(F.lit("SHIFT-"), F.col(KEY_COL))))
shifted.write.mode("append").saveAsTable(PREDICTIONS_TABLE)

refresh = w.quality_monitors.run_refresh(table_name=PREDICTIONS_TABLE)
print("Refresh started:", refresh.refresh_id, "— takes a few minutes on serverless.")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 3 · Read the metric tables
# MAGIC Wait for the refresh to finish (Quality tab shows *Succeeded*), then run this cell.
# MAGIC
# MAGIC > 🧞 **Genie Code prompt**
# MAGIC > ```
# MAGIC > From {PREDICTIONS_TABLE}_drift_metrics, show the columns with the largest population stability
# MAGIC > index or Jensen-Shannon distance in the most recent window, and explain what changed.
# MAGIC > ```

# COMMAND ----------

drift_table = f"{PREDICTIONS_TABLE}_drift_metrics"
profile_table = f"{PREDICTIONS_TABLE}_profile_metrics"
if spark.catalog.tableExists(drift_table):
    display(spark.sql(f"""
      SELECT window.start AS window_start, column_name, slice_key, slice_value,
             ROUND(js_distance, 3) AS js_distance, ROUND(population_stability_index, 3) AS psi
      FROM {drift_table}
      WHERE column_name IN ('claim_amount', 'num_contacts', 'sentiment_score', 'predicted_escalation')
        AND drift_type = 'CONSECUTIVE'
      ORDER BY window_start DESC, js_distance DESC NULLS LAST LIMIT 40"""))
else:
    print("Metric tables appear after the first refresh completes.")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 4 · Turn drift into an alert
# MAGIC 1. Open **SQL Editor**, and write a query over `case_predictions_drift_metrics` that returns the latest
# MAGIC    `js_distance` for `claim_amount`.
# MAGIC 2. **Save** → **Schedule** → **Add alert**: trigger when the value is above `0.2`; notify the model owner.
# MAGIC 3. In module 07 the same alert can start the retraining job.
# MAGIC
# MAGIC ## ✅ Checkpoint
# MAGIC * The **Quality** tab on `case_predictions` shows profile, drift and model-quality metrics, plus a dashboard.
# MAGIC * You can point to the drift you introduced in `claim_amount` and `num_contacts`.
# MAGIC
# MAGIC **Next:** `07_mlops_bundles`.
