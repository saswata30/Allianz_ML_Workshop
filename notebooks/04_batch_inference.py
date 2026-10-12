# Databricks notebook source
# MAGIC %md
# MAGIC # 04 · Serving — batch inference
# MAGIC
# MAGIC **Platform layer:** Serving — *Batch inference*
# MAGIC
# MAGIC Most claims use cases score on a schedule rather than per request. You load the `@champion` model straight
# MAGIC from Unity Catalog as a **Spark UDF**, score every open case in parallel, and write the results to a Delta
# MAGIC table that Power BI, a Genie space, or a case-management system can read.
# MAGIC
# MAGIC Because you load **by alias**, promoting a new model version to `@champion` changes what this job scores
# MAGIC with — no code change.
# MAGIC
# MAGIC **Time:** 20 minutes.

# COMMAND ----------

# MAGIC %pip install -q databricks-feature-engineering xgboost
# MAGIC %restart_python

# COMMAND ----------

# MAGIC %run ./_config

# COMMAND ----------

import mlflow
from mlflow import MlflowClient
from pyspark.sql import functions as F

mlflow.set_registry_uri("databricks-uc")
champion = MlflowClient().get_model_version_by_alias(MODEL_NAME, "champion")
MODEL_URI = f"models:/{MODEL_NAME}@champion"
print(f"Scoring with {MODEL_NAME} v{champion.version} (@champion)")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 1 · Score the open cases
# MAGIC
# MAGIC > 🧞 **Genie Code prompt**
# MAGIC > ```
# MAGIC > Load models:/{MODEL_NAME}@champion as a Spark UDF with mlflow.pyfunc.spark_udf (result type double).
# MAGIC > Join {OPEN_CASES_TABLE} to {FEATURE_TABLE} on case_id, score every row, and add columns
# MAGIC > escalation_probability, predicted_escalation (probability >= 0.5), model_version and scored_at.
# MAGIC > ```

# COMMAND ----------

# Reference
score_udf = mlflow.pyfunc.spark_udf(spark, MODEL_URI, result_type="double", env_manager="local")

input_cols = score_udf.metadata.get_input_schema().input_names()
features = spark.table(FEATURE_TABLE)

def score(keys_df, label_col=None):
    df = keys_df.join(features, KEY_COL, "inner")
    out = (df.withColumn("escalation_probability", score_udf(F.struct(*[F.col(c) for c in input_cols])))
             .withColumn("predicted_escalation", (F.col("escalation_probability") >= 0.5).cast("int"))
             .withColumn("model_version", F.lit(str(champion.version)))
             .withColumn("scored_at", F.current_timestamp()))
    keep = [KEY_COL, TIMESTAMP_COL, "escalation_probability", "predicted_escalation", "model_version", "scored_at",
            "line_of_business", "channel", "region", "claim_amount", "sentiment_score", "num_contacts", "days_open"]
    if label_col:
        keep.append(label_col)
    return out.select(*keep)

open_scored = score(spark.table(OPEN_CASES_TABLE).select(KEY_COL, TIMESTAMP_COL))
display(open_scored.orderBy(F.desc("escalation_probability")).limit(20))

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 2 · Write the predictions table
# MAGIC
# MAGIC We also score the **last 60 days of closed cases**, where the true outcome is known. Keeping the label next
# MAGIC to the prediction lets the monitor in module 06 measure accuracy, not just drift. The `escalated` column is
# MAGIC empty for open cases until they close.

# COMMAND ----------

recent_closed = (spark.table(CASES_TABLE)
                 .where(F.col(TIMESTAMP_COL) >= F.date_sub(F.current_date(), 75))
                 .select(KEY_COL, TIMESTAMP_COL, LABEL_COL))
closed_scored = score(recent_closed.select(KEY_COL, TIMESTAMP_COL), label_col=None).join(
    recent_closed.select(KEY_COL, LABEL_COL), KEY_COL)
open_scored_l = open_scored.withColumn(LABEL_COL, F.lit(None).cast("int"))

predictions = closed_scored.unionByName(open_scored_l)
(predictions.write.mode("overwrite").option("overwriteSchema", "true")
 .option("delta.enableChangeDataFeed", "true").saveAsTable(PREDICTIONS_TABLE))
print(f"Wrote {predictions.count():,} predictions to {PREDICTIONS_TABLE}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 3 · Who should the claims team call first?
# MAGIC
# MAGIC > 🧞 **Genie Code prompt**
# MAGIC > ```
# MAGIC > From {PREDICTIONS_TABLE}, list the 25 open cases (escalated is null) with the highest
# MAGIC > escalation_probability, with line_of_business, claim_amount and sentiment_score. Then show the
# MAGIC > count of high-risk open cases (probability >= 0.6) by line_of_business.
# MAGIC > ```

# COMMAND ----------

# Reference
display(spark.sql(f"""
  SELECT line_of_business, COUNT(*) AS high_risk_open_cases, ROUND(AVG(escalation_probability), 3) AS avg_probability
  FROM {PREDICTIONS_TABLE}
  WHERE {LABEL_COL} IS NULL AND escalation_probability >= 0.6
  GROUP BY line_of_business ORDER BY high_risk_open_cases DESC"""))

# COMMAND ----------

# MAGIC %md
# MAGIC ### Optional · Score in SQL with `ai_query`
# MAGIC After module 05 creates the serving endpoint, analysts can score from SQL without Python:
# MAGIC ```sql
# MAGIC SELECT case_id, ai_query('<your-endpoint-name>', named_struct(<feature columns>), 'DOUBLE') AS escalation_probability
# MAGIC FROM <open cases joined to features>;
# MAGIC ```
# MAGIC
# MAGIC ## ✅ Checkpoint
# MAGIC * `case_predictions` exists, with probabilities for open and recently closed cases.
# MAGIC * Its **Lineage** tab shows the model and the source tables.
# MAGIC
# MAGIC **Next:** `05_model_serving`.
