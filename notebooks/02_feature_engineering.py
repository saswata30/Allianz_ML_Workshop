# Databricks notebook source
# MAGIC %md
# MAGIC # 02 · Data — feature engineering & Feature Store
# MAGIC
# MAGIC **Platform layer:** Data — *Lakeflow* · *Feature Store*
# MAGIC
# MAGIC You turn the raw case table into a governed **feature table** in Unity Catalog. A feature table is a Delta
# MAGIC table with a primary key; once registered, Databricks tracks lineage from the feature table to every model
# MAGIC trained on it, and the same features can be reused for batch scoring and online serving.
# MAGIC
# MAGIC * **Features** go to `case_features` (one row per `case_id`, for both closed and open cases).
# MAGIC * **Labels** go to `case_labels` (closed cases only). Keeping them separate prevents label leakage.
# MAGIC
# MAGIC > **Lakeflow in production:** here we compute features in a notebook for speed. In production the same
# MAGIC > logic runs as a **Lakeflow Declarative Pipeline** (see `pipelines/case_features_pipeline.sql`) with
# MAGIC > data-quality expectations, scheduled or triggered as new cases land.
# MAGIC
# MAGIC **Time:** 25 minutes.

# COMMAND ----------

# MAGIC %pip install -q databricks-feature-engineering
# MAGIC %restart_python

# COMMAND ----------

# MAGIC %run ./_config

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 1 · Build the features
# MAGIC
# MAGIC > 🧞 **Genie Code prompt**
# MAGIC > ```
# MAGIC > Write a PySpark function build_features(df) that takes the cases table and returns case_id plus
# MAGIC > these features: log_claim_amount, contacts_per_week, is_high_value, negative_sentiment,
# MAGIC > new_customer, and the raw columns line_of_business, channel, region, claim_amount,
# MAGIC > policy_tenure_years, prior_claims_12m, customer_age, num_contacts, days_open,
# MAGIC > sentiment_score, has_attorney. Cast the flags to int. Apply it to both {CASES_TABLE} and
# MAGIC > {OPEN_CASES_TABLE} and union the results.
# MAGIC > ```

# COMMAND ----------

# Reference
from pyspark.sql import DataFrame, functions as F

RAW_COLS = ["line_of_business", "channel", "region", "claim_amount", "policy_tenure_years",
            "prior_claims_12m", "customer_age", "num_contacts", "days_open", "sentiment_score"]

def build_features(df: DataFrame) -> DataFrame:
    return df.select(
        KEY_COL,
        *RAW_COLS,
        F.col("has_attorney").cast("int").alias("has_attorney"),
        F.log1p("claim_amount").alias("log_claim_amount"),
        (F.col("num_contacts") / F.greatest(F.col("days_open") / 7, F.lit(1.0))).alias("contacts_per_week"),
        (F.col("claim_amount") > 50_000).cast("int").alias("is_high_value"),
        (F.col("sentiment_score") < -0.25).cast("int").alias("negative_sentiment"),
        (F.col("policy_tenure_years") < 1).cast("int").alias("new_customer"),
    )

features_df = build_features(spark.table(CASES_TABLE))
if spark.catalog.tableExists(OPEN_CASES_TABLE):
    features_df = features_df.unionByName(build_features(spark.table(OPEN_CASES_TABLE)))
display(features_df.limit(5))

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 2 · Register the feature table in Unity Catalog
# MAGIC
# MAGIC > 🧞 **Genie Code prompt**
# MAGIC > ```
# MAGIC > Use the Databricks FeatureEngineeringClient to create a feature table {FEATURE_TABLE}
# MAGIC > from features_df with primary key case_id and a description. If it already exists,
# MAGIC > merge the new rows into it instead.
# MAGIC > ```

# COMMAND ----------

# Reference
from databricks.feature_engineering import FeatureEngineeringClient

fe = FeatureEngineeringClient()

if spark.catalog.tableExists(FEATURE_TABLE):
    fe.write_table(name=FEATURE_TABLE, df=features_df, mode="merge")
    print(f"Merged features into {FEATURE_TABLE}")
else:
    fe.create_table(
        name=FEATURE_TABLE,
        primary_keys=[KEY_COL],
        df=features_df,
        description="Claim-case features for escalation-risk models (Allianz ML workshop)",
    )
    print(f"Created feature table {FEATURE_TABLE}")

# COMMAND ----------

# MAGIC %md ## Step 3 · Write the label table

# COMMAND ----------

labels = spark.table(CASES_TABLE).select(KEY_COL, TIMESTAMP_COL, F.col(LABEL_COL).cast("int").alias(LABEL_COL))
labels.write.mode("overwrite").option("overwriteSchema", "true").saveAsTable(LABEL_TABLE)
print(f"Wrote {labels.count():,} labels to {LABEL_TABLE}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## ✅ Checkpoint
# MAGIC * In **Catalog Explorer**, open `case_features`: it has a primary-key constraint on `case_id`.
# MAGIC * Open the **Features** page (left navigation, under *Machine Learning*): your feature table is listed and searchable.
# MAGIC
# MAGIC **Next:** `03_train_and_track`.
