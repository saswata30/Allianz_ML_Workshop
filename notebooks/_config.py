# Databricks notebook source
# MAGIC %md
# MAGIC # Shared configuration
# MAGIC
# MAGIC Every workshop notebook runs this file first with `%run ./_config`, so all modules agree on the same
# MAGIC catalog, schema, table, model and endpoint names.
# MAGIC
# MAGIC * **catalog** – the Unity Catalog catalog for the workshop (default `adp_workshop`).
# MAGIC * **schema** – leave blank to get a personal schema `ml_<your_user>`, so attendees never collide.
# MAGIC * **source_table** – leave blank to use the synthetic case data from `00_setup_data`; set it to a curated
# MAGIC   ADP table (for example `adp.claims.cases`) to run the same pattern on real data.

# COMMAND ----------

import re

dbutils.widgets.text("catalog", "adp_workshop", "Catalog")
dbutils.widgets.text("schema", "", "Schema (blank = ml_<user>)")
dbutils.widgets.text("source_table", "", "Source table (blank = synthetic)")

CATALOG = dbutils.widgets.get("catalog").strip()
CURRENT_USER = spark.sql("SELECT current_user()").first()[0]
USER_TAG = re.sub(r"[^a-z0-9]", "_", CURRENT_USER.split("@")[0].lower())
SCHEMA = dbutils.widgets.get("schema").strip() or f"ml_{USER_TAG}"

# Governance & storage — everything below lives in one Unity Catalog schema
FQ = f"{CATALOG}.{SCHEMA}"
CASES_TABLE = dbutils.widgets.get("source_table").strip() or f"{FQ}.cases"   # labelled history
OPEN_CASES_TABLE = f"{FQ}.open_cases"                                         # unlabelled, to score
FEATURE_TABLE = f"{FQ}.case_features"                                         # Feature Store table
LABEL_TABLE = f"{FQ}.case_labels"
PREDICTIONS_TABLE = f"{FQ}.case_predictions"                                  # batch inference output
MODEL_NAME = f"{FQ}.case_escalation"                                          # UC model registry
ENDPOINT_NAME = f"case-escalation-{USER_TAG}".replace("_", "-")[:63]           # Model Serving endpoint
INFERENCE_TABLE_PREFIX = "case_escalation_endpoint"                           # AI Gateway inference table
EXPERIMENT_PATH = f"/Users/{CURRENT_USER}/allianz_ml_workshop_case_escalation"

LABEL_COL = "escalated"
KEY_COL = "case_id"
TIMESTAMP_COL = "opened_at"

print(f"""Workshop configuration
  user            : {CURRENT_USER}
  schema          : {FQ}
  cases table     : {CASES_TABLE}
  feature table   : {FEATURE_TABLE}
  model (UC)      : {MODEL_NAME}
  endpoint        : {ENDPOINT_NAME}
  experiment      : {EXPERIMENT_PATH}""")
