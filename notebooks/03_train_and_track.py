# Databricks notebook source
# MAGIC %md
# MAGIC # 03 · Training — train, track and register with MLflow
# MAGIC
# MAGIC **Platform layers:** Training — *AI Runtime (serverless)* / *ML Runtime (classic)* · MLOps — *MLflow* ·
# MAGIC Governance — *Unity Catalog model registry*
# MAGIC
# MAGIC You build a training set from the feature table, train three candidate models (including XGBoost), compare them in an
# MAGIC **MLflow experiment**, and register the winner to **Unity Catalog** with the alias `@champion`.
# MAGIC
# MAGIC **Compute:** run this on **serverless** notebook compute (the default). For large or deep-learning models,
# MAGIC switch to **AI Runtime** serverless GPU; teams that need custom libraries or init scripts can use a classic
# MAGIC cluster on an **ML Runtime** — the code is the same.
# MAGIC
# MAGIC **Time:** 40 minutes.
# MAGIC
# MAGIC ---
# MAGIC ### 🗺️ How to follow this notebook
# MAGIC * **Compute:** at the top right, attach **Serverless**.
# MAGIC * **Run cells one at a time, top to bottom** — click inside a cell and press `Shift + Enter`.
# MAGIC * **Symbols:** 🧞 = ask Genie Code first · ▶ = run the next cell · 👀 = what you should see · ✅ = check before moving on.
# MAGIC * **Stuck?** Run the cell marked `# Reference` under each prompt — it always works, so you can never fall behind.
# MAGIC
# MAGIC ### Your path (40 minutes)
# MAGIC 1. ▶ Run the `%pip install` cell (Python restarts), then the config cell, then the MLflow cell → 👀 the experiment path.
# MAGIC 2. Step 1: build the training set → 👀 `Train 48,000 · Test 12,000 · Positive rate ~18%`.
# MAGIC 3. Step 2: train the three models (about 3 minutes) → 👀 three lines, each with a ROC AUC of roughly 0.70–0.80.
# MAGIC 4. Step 3: click **Experiments** in the left bar → open `allianz_ml_workshop_case_escalation` → tick the three runs → **Compare**.
# MAGIC 5. Step 4: register the best model → 👀 `Registered … as @champion`.
# MAGIC 6. ✅ In **Catalog → your schema → Models**, `case_escalation` version 1 has the alias `champion`.

# COMMAND ----------

# MAGIC %pip install -q databricks-feature-engineering xgboost
# MAGIC %restart_python

# COMMAND ----------

# MAGIC %run ./_config

# COMMAND ----------

import mlflow
mlflow.set_registry_uri("databricks-uc")
mlflow.set_experiment(EXPERIMENT_PATH)
print("Experiment:", EXPERIMENT_PATH)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 1 · Create the training set from the Feature Store
# MAGIC
# MAGIC `FeatureLookup` joins features onto the label table by `case_id`. Because the features come from a
# MAGIC registered feature table, Unity Catalog records the lineage *feature table → model*.
# MAGIC
# MAGIC > 🧞 **Genie Code prompt**
# MAGIC > ```
# MAGIC > Using FeatureEngineeringClient, build a training set from {LABEL_TABLE} with a FeatureLookup
# MAGIC > on {FEATURE_TABLE} keyed by case_id, label escalated, excluding opened_at. Load it as a pandas
# MAGIC > DataFrame and split it 80/20 stratified on the label with random_state 42.
# MAGIC > ```
# MAGIC
# MAGIC ▶ **Next:** if you used Genie Code and it worked, skip ahead. Otherwise run the `# Reference` cell below.

# COMMAND ----------

# Reference
from databricks.feature_engineering import FeatureEngineeringClient, FeatureLookup
from sklearn.model_selection import train_test_split

fe = FeatureEngineeringClient()
training_set = fe.create_training_set(
    df=spark.table(LABEL_TABLE),
    feature_lookups=[FeatureLookup(table_name=FEATURE_TABLE, lookup_key=KEY_COL)],
    label=LABEL_COL,
    exclude_columns=[TIMESTAMP_COL],
)
train_pdf = training_set.load_df().toPandas()

CATEGORICAL = ["line_of_business", "channel", "region"]
NUMERIC = [c for c in train_pdf.columns if c not in CATEGORICAL + [KEY_COL, LABEL_COL]]
X = train_pdf[CATEGORICAL + NUMERIC]
y = train_pdf[LABEL_COL]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
print(f"Train {len(X_train):,} · Test {len(X_test):,} · Positive rate {y.mean():.1%}")
print("Numeric features:", NUMERIC)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 2 · Train three candidates, tracked in MLflow
# MAGIC
# MAGIC We wrap the scikit-learn pipeline in a small `pyfunc` model so that **both batch scoring and the serving
# MAGIC endpoint return the escalation probability** (not just 0/1).
# MAGIC
# MAGIC > 🧞 **Genie Code prompt**
# MAGIC > ```
# MAGIC > Train three scikit-learn pipelines on X_train/y_train: (1) one-hot encode the categorical columns and
# MAGIC > scale numerics, then LogisticRegression; (2) one-hot encode categoricals, then
# MAGIC > HistGradientBoostingClassifier; (3) one-hot encode categoricals, then XGBClassifier.
# MAGIC > Log each as its own MLflow run with ROC AUC, PR AUC, F1 and recall on X_test, and log the
# MAGIC > model with a signature and input example.
# MAGIC > ```
# MAGIC
# MAGIC ▶ **Next:** if you used Genie Code and it worked, skip ahead. Otherwise run the `# Reference` cell below.

# COMMAND ----------

# Reference
import pandas as pd
import sklearn
from mlflow.models import infer_signature
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, f1_score, recall_score, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
import xgboost
from xgboost import XGBClassifier


class EscalationModel(mlflow.pyfunc.PythonModel):
    """Returns the probability that a case is escalated."""
    def __init__(self, pipeline):
        self.pipeline = pipeline

    def predict(self, context, model_input, params=None):
        return pd.Series(self.pipeline.predict_proba(model_input)[:, 1], name="escalation_probability")


def make_pipeline(kind):
    if kind == "logistic_regression":
        pre = ColumnTransformer([("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), CATEGORICAL),
                                 ("num", StandardScaler(), NUMERIC)])
        clf = LogisticRegression(max_iter=1000, class_weight="balanced")
    elif kind == "hist_gradient_boosting":
        pre = ColumnTransformer([("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), CATEGORICAL)],
                                remainder="passthrough")
        clf = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.06, max_leaf_nodes=31,
                                             class_weight="balanced", random_state=42)
    else:  # xgboost
        pre = ColumnTransformer([("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), CATEGORICAL)],
                                remainder="passthrough")
        clf = XGBClassifier(n_estimators=400, learning_rate=0.05, max_depth=5, subsample=0.9,
                            colsample_bytree=0.9, eval_metric="auc", n_jobs=-1, random_state=42,
                            scale_pos_weight=float((y_train == 0).sum() / max((y_train == 1).sum(), 1)))
    return Pipeline([("prep", pre), ("clf", clf)])


results = {}
for kind in ["logistic_regression", "hist_gradient_boosting", "xgboost"]:
    with mlflow.start_run(run_name=kind) as run:
        pipe = make_pipeline(kind).fit(X_train, y_train)
        proba = pipe.predict_proba(X_test)[:, 1]
        pred = (proba >= 0.5).astype(int)
        metrics = {
            "test_roc_auc": roc_auc_score(y_test, proba),
            "test_pr_auc": average_precision_score(y_test, proba),
            "test_f1": f1_score(y_test, pred),
            "test_recall": recall_score(y_test, pred),
        }
        mlflow.log_params({"model_type": kind, "n_features": X.shape[1], "train_rows": len(X_train)})
        mlflow.log_metrics(metrics)
        model = EscalationModel(pipe)
        sample = X_test.head(5)
        mlflow.pyfunc.log_model(
            artifact_path="model",
            python_model=model,
            signature=infer_signature(sample, model.predict(None, sample)),
            input_example=sample,
            pip_requirements=[f"scikit-learn=={sklearn.__version__}", f"pandas=={pd.__version__}",
                              f"xgboost=={xgboost.__version__}"],
        )
        results[kind] = {"run_id": run.info.run_id, **metrics}
        print(f"{kind:24s} ROC AUC {metrics['test_roc_auc']:.3f} · PR AUC {metrics['test_pr_auc']:.3f} · recall {metrics['test_recall']:.3f}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 3 · Compare runs in the MLflow UI
# MAGIC 1. Click **Experiments** in the left navigation and open `allianz_ml_workshop_case_escalation`.
# MAGIC 2. Tick all three runs → **Compare**. Look at `test_roc_auc` and `test_recall`.
# MAGIC 3. Open the better run → **Artifacts** → `model`: note the signature and `requirements.txt`.
# MAGIC
# MAGIC > 🧞 **Genie Code prompt**
# MAGIC > ```
# MAGIC > Search the runs in experiment {EXPERIMENT_PATH} with mlflow.search_runs, show model_type and the
# MAGIC > test metrics in a table, and tell me which run has the best ROC AUC.
# MAGIC > ```
# MAGIC
# MAGIC ▶ **Next:** if you used Genie Code and it worked, skip ahead. Otherwise run the `# Reference` cell below.

# COMMAND ----------

# Reference
display(mlflow.search_runs(experiment_names=[EXPERIMENT_PATH],
                           order_by=["metrics.test_roc_auc DESC"])
        [["run_id", "params.model_type", "metrics.test_roc_auc", "metrics.test_pr_auc", "metrics.test_recall", "start_time"]])

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 4 · Register the best model to Unity Catalog and set `@champion`

# COMMAND ----------

from mlflow import MlflowClient

best_kind = max(results, key=lambda k: results[k]["test_roc_auc"])
best = results[best_kind]
mv = mlflow.register_model(f"runs:/{best['run_id']}/model", MODEL_NAME)

client = MlflowClient()
client.update_registered_model(MODEL_NAME, description="Predicts the probability that a claim case is escalated. Allianz ML workshop.")
client.update_model_version(MODEL_NAME, mv.version, description=f"{best_kind} · test ROC AUC {best['test_roc_auc']:.3f}")

# Promote only if this version beats the current champion (first run: no champion yet)
try:
    current = client.get_model_version_by_alias(MODEL_NAME, "champion")
    champ_auc = client.get_run(current.run_id).data.metrics.get("test_roc_auc", 0.0)
except Exception:
    current, champ_auc = None, -1.0

if best["test_roc_auc"] > champ_auc:
    client.set_registered_model_alias(MODEL_NAME, "champion", mv.version)
    print(f"Registered {MODEL_NAME} v{mv.version} ({best_kind}, AUC {best['test_roc_auc']:.3f}) as @champion")
else:
    client.set_registered_model_alias(MODEL_NAME, "challenger", mv.version)
    print(f"v{mv.version} (AUC {best['test_roc_auc']:.3f}) did not beat champion v{current.version} "
          f"(AUC {champ_auc:.3f}) — set as @challenger for review")

# COMMAND ----------

# MAGIC %md
# MAGIC ## ✅ Checkpoint
# MAGIC * **Catalog Explorer → your schema → Models → `case_escalation`** shows version 1 with alias `champion`.
# MAGIC * The model's **Lineage** tab links back to the run. The feature table's lineage shows the training set.
# MAGIC * ROC AUC should be about 0.70–0.80 on the synthetic data (the label is deliberately noisy, like real escalations).
# MAGIC
# MAGIC **Discuss:** what would make you promote a new version to `@champion` — a metric threshold, a review,
# MAGIC or both? (Module 07 runs this on a schedule.)
# MAGIC
# MAGIC **Next:** `04_batch_inference`.
