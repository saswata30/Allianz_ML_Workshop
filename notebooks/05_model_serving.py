# Databricks notebook source
# MAGIC %md
# MAGIC # 05 · Serving — real-time Model Serving (+ Feature Serving)
# MAGIC
# MAGIC **Platform layers:** Serving — *Model Serving* · *Feature Serving* · Monitoring — *Inference tables*
# MAGIC
# MAGIC When a claims handler opens a case, the case-management app needs a risk score in milliseconds. You deploy
# MAGIC the `@champion` model to a **serverless Model Serving endpoint** (scale to zero, so it costs nothing when idle)
# MAGIC with **inference tables** turned on: every request and response is logged automatically to a Delta table
# MAGIC in your schema, ready for monitoring and retraining.
# MAGIC
# MAGIC **Time:** 30 minutes (endpoint creation takes about 10 minutes — keep going with the reading below while it builds).

# COMMAND ----------

# MAGIC %run ./_config

# COMMAND ----------

from databricks.sdk import WorkspaceClient
from databricks.sdk.service.serving import (
    AiGatewayConfig, AiGatewayInferenceTableConfig, EndpointCoreConfigInput, ServedEntityInput)
from mlflow import MlflowClient
import mlflow

mlflow.set_registry_uri("databricks-uc")
w = WorkspaceClient()
champion = MlflowClient().get_model_version_by_alias(MODEL_NAME, "champion")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 1 · Create (or update) the serving endpoint
# MAGIC
# MAGIC > 🧞 **Genie Code prompt**
# MAGIC > ```
# MAGIC > Using the Databricks SDK, create a Model Serving endpoint named {ENDPOINT_NAME} that serves the
# MAGIC > @champion version of {MODEL_NAME}, workload size Small, scale to zero enabled, with AI Gateway
# MAGIC > inference tables writing to {CATALOG}.{SCHEMA}. If the endpoint already exists, update it to the
# MAGIC > champion version instead.
# MAGIC > ```
# MAGIC
# MAGIC **UI alternative:** Catalog Explorer → your model → version → **Serve this model**.

# COMMAND ----------

# Reference
served = [ServedEntityInput(
    entity_name=MODEL_NAME,
    entity_version=str(champion.version),
    workload_size="Small",
    scale_to_zero_enabled=True,
)]

existing = {e.name for e in w.serving_endpoints.list()}
if ENDPOINT_NAME in existing:
    print(f"Updating {ENDPOINT_NAME} to v{champion.version} …")
    w.serving_endpoints.update_config_and_wait(name=ENDPOINT_NAME, served_entities=served)
else:
    print(f"Creating {ENDPOINT_NAME} with v{champion.version} … (about 10 minutes)")
    w.serving_endpoints.create_and_wait(
        name=ENDPOINT_NAME,
        config=EndpointCoreConfigInput(served_entities=served),
        ai_gateway=AiGatewayConfig(inference_table_config=AiGatewayInferenceTableConfig(
            catalog_name=CATALOG, schema_name=SCHEMA, table_name_prefix=INFERENCE_TABLE_PREFIX, enabled=True)),
    )
print(f"Endpoint ready: {w.config.host}/ml/endpoints/{ENDPOINT_NAME}")

# COMMAND ----------

# MAGIC %md
# MAGIC ### While it builds — what is happening?
# MAGIC * Databricks packages the model and its `requirements.txt` into a container, and provisions **serverless**
# MAGIC   compute that autoscales with traffic and scales to zero when idle.
# MAGIC * Access is governed by Unity Catalog permissions on the model plus endpoint permissions (`CAN_QUERY`).
# MAGIC * **AI Gateway** adds usage tracking, rate limits and the inference table `…_payload`.

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 2 · Query the endpoint
# MAGIC
# MAGIC > 🧞 **Genie Code prompt**
# MAGIC > ```
# MAGIC > Take 5 open cases joined to their features and send them to the endpoint {ENDPOINT_NAME} with
# MAGIC > w.serving_endpoints.query using dataframe_records. Show case_id with the returned probability.
# MAGIC > ```

# COMMAND ----------

# Reference
import json

input_cols = mlflow.models.get_model_info(f"models:/{MODEL_NAME}@champion").signature.inputs.input_names()
sample = (spark.table(OPEN_CASES_TABLE).select(KEY_COL).limit(5)
          .join(spark.table(FEATURE_TABLE), KEY_COL).toPandas())
records = json.loads(sample[input_cols].to_json(orient="records"))

resp = w.serving_endpoints.query(name=ENDPOINT_NAME, dataframe_records=records)
sample["escalation_probability"] = resp.predictions
display(sample[[KEY_COL, "line_of_business", "claim_amount", "sentiment_score", "escalation_probability"]])

# COMMAND ----------

# MAGIC %md
# MAGIC ### The same call over REST (for the case-management app)
# MAGIC ```bash
# MAGIC curl -X POST "$DATABRICKS_HOST/serving-endpoints/<endpoint>/invocations" \
# MAGIC   -H "Authorization: Bearer $TOKEN" -H "Content-Type: application/json" \
# MAGIC   -d '{"dataframe_records": [{"line_of_business": "Motor", "channel": "Broker", ...}]}'
# MAGIC ```
# MAGIC Use a **service principal** OAuth token for applications — never a personal token.

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 3 · Inspect the inference table
# MAGIC Requests take a few minutes to land. Re-run this cell after querying the endpoint a few times.

# COMMAND ----------

payload_table = f"{FQ}.{INFERENCE_TABLE_PREFIX}_payload"
if spark.catalog.tableExists(payload_table):
    display(spark.sql(f"SELECT request_time, status_code, execution_duration_ms, request, response FROM {payload_table} ORDER BY request_time DESC LIMIT 20"))
else:
    print(f"{payload_table} not created yet — it appears after the first logged requests (allow ~5–10 minutes).")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 4 · Ask Genie Code about endpoint health
# MAGIC 1. Open **Serving** → your endpoint, then click the sparkle **Genie Code** icon.
# MAGIC 2. Select **Agent** in the lower-right corner of the panel.
# MAGIC 3. Try: `Check the health of this endpoint`, `Is my endpoint configured correctly?`,
# MAGIC    `Show me recent requests to this endpoint`, or `/diagnose` if a deployment failed.
# MAGIC
# MAGIC Genie Code works **read-only** here: it diagnoses and recommends (for example scaling settings) but does
# MAGIC not change the endpoint.

# COMMAND ----------

# MAGIC %md
# MAGIC ## Stretch · Feature Serving
# MAGIC
# MAGIC Today the caller sends all feature values. With **Feature Serving**, the app sends only `case_id` and
# MAGIC Databricks looks the features up at request time from an **online feature store** that is synced from your
# MAGIC `case_features` table:
# MAGIC
# MAGIC 1. Publish `case_features` to an online store (Catalog Explorer → the feature table → **Create online table / Publish**).
# MAGIC 2. Log the model with `fe.log_model(..., training_set=training_set)` so it carries its feature lookups.
# MAGIC 3. Serve that model version — the endpoint then accepts `{"case_id": "CASE-0060001"}`.
# MAGIC
# MAGIC Docs: *Feature Serving* and *online feature stores* in the Databricks documentation for Azure.
# MAGIC
# MAGIC ## ✅ Checkpoint
# MAGIC * **Serving** page shows your endpoint as *Ready*, and it returns probabilities.
# MAGIC * The `…_payload` inference table appears in your schema.
# MAGIC
# MAGIC **Next:** `06_monitoring`.
