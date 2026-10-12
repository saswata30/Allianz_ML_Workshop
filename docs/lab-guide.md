# Lab guide — Allianz ML Workshop

Step-by-step instructions for attendees. Each module is one notebook in `notebooks/`. Every module follows the
same rhythm:

1. **Read** the markdown at the top — it names the platform layer you are working in.
2. **Ask Genie Code** using the 🧞 prompt in the notebook.
3. **Compare** with the *Reference* cell underneath (run it if Genie Code's version differs or fails).
4. **Check** the ✅ checkpoint before moving on.

---

## How to use Genie Code

| Action | How |
|---|---|
| Open it | Click the sparkle **Genie Code** icon to open the side panel in a notebook (or the SQL editor, or a serving endpoint page). |
| Let it run code | In the lower-right corner of the panel, select **Agent**. |
| Give it context | Type `@` and pick a table (for example `@cases`), or click **Add context**. |
| Approve steps | When it proposes code, click **Allow** or **Skip**. Turn on **Auto-approve** once you trust the flow. |
| Steer | Answer its clarifying questions; use **Continue** or **Reject** on its results. |
| Stop | Click the red stop icon. |
| Ask inline | Inside a cell, press `Cmd + I` (Mac) or `Ctrl + I` (Windows) to edit just that cell. |

Tips:
* Be specific about **tables, the label, the metric and the output** you want. Vague prompts get generic code.
* Ask it to **show its plan first** for anything multi-step.
* Keep the browser tab open while it works.
* Read every cell before you allow it to run. Genie Code only sees what your Unity Catalog permissions allow.
* Optional: paste the text in [`genie-code-instructions.md`](genie-code-instructions.md) into your Genie Code
  instructions so it follows the workshop conventions.

---

## Module 00 · Setup (10 min)

1. **Workspace → Create → Git folder.** Paste `https://github.com/saswata30/Allianz_ML_Workshop` and click **Create Git folder**.
2. Open `notebooks/00_setup_data`. In the compute selector, choose **Serverless**.
3. Leave the widgets as they are (catalog `adp_workshop`; schema blank → your personal `ml_<user>`).
4. **Run all.**

✅ **Checkpoint:** in **Catalog** (left navigation) open `adp_workshop` → `ml_<you>`; you see `cases` (60,000 rows)
and `open_cases` (3,000 rows). The printed escalation rate is about 15–25%.

**If it fails:** `PERMISSION_DENIED` on `CREATE SCHEMA` → ask the facilitator to grant `CREATE SCHEMA` on the catalog,
or to give you a pre-created schema and type its name into the `schema` widget.

---

## Module 01 · EDA with Genie Code (25 min) — *EDA & development*

1. Open `01_eda_genie_code`, attach **Serverless**, and run the first two cells (title + config).
2. Open the Genie Code panel → **Agent**.
3. **Step 1 — profile:** paste the prompt; replace `{CASES_TABLE}` with your table (or type `@cases`). Allow the cells it proposes.
4. **Step 2 — segments:** ask for escalation rate by line of business and channel, with a bar chart.
5. **Step 3 — signals:** ask it to compare distributions for escalated vs non-escalated cases.
6. **Step 4 — features:** ask it to propose 5 engineered features. Compare with the reference table in the notebook.

✅ **Checkpoint:** you can say which lines of business escalate most and name three predictive signals.

**Discuss:** Genie Code used the table and column comments set in module 00. What would better metadata on ADP tables do for its answers?

---

## Module 01b · Genie Code challenge — one prompt to a model (20 min)

This mirrors the Databricks demo *Genie Code for machine learning*.

1. Create a new, empty notebook `genie_code_scratch`; attach **Serverless**.
2. In Genie Code (**Agent**), paste the challenge prompt from `01b_genie_code_end_to_end`. Type `@` to attach
   `cases` and `open_cases`.
3. Review the **plan**; ask for changes if you want them; then **Allow** (or **Auto-approve**).
4. When it finishes, try the follow-up prompts (feature importance, leakage check, tuning, ROC curve).

✅ **Checkpoint:** you have an XGBoost model with ROC AUC and recall reported, an MLflow run, and the 10 highest-risk open cases.

**Debrief:** what is missing for production? (Reusable features, a governed registered model, serving,
monitoring, CI/CD) — modules 02–07 add each one.

---

## Module 02 · Feature engineering & Feature Store (25 min) — *Data*

1. Open `02_feature_engineering`, attach **Serverless**, run the `%pip` cell (Python restarts) and the config cell.
2. **Step 1:** ask Genie Code to write `build_features(df)` with the listed features; or run the reference.
3. **Step 2:** register `case_features` with primary key `case_id` using `FeatureEngineeringClient`.
4. **Step 3:** run the cell that writes `case_labels`.

✅ **Checkpoint:**
* `case_features` in Catalog shows a **primary key** on `case_id`.
* **Machine Learning → Features** lists your feature table.

**Lakeflow:** look at `pipelines/case_features_pipeline.sql`. The same logic runs as a serverless Lakeflow
Declarative Pipeline with data-quality **expectations** (bad rows dropped, others counted). Module 07 deploys it.

---

## Module 03 · Train, track & register (40 min) — *Training · MLflow · Unity Catalog*

1. Open `03_train_and_track`, attach **Serverless**, run the `%pip` cell, the config cell and the MLflow setup cell.
2. **Step 1:** build the training set with `FeatureLookup` (Genie Code prompt or reference).
3. **Step 2:** train three candidates — logistic regression, histogram gradient boosting, XGBoost. Each is its own MLflow run.
4. **Step 3:** open **Experiments** → `allianz_ml_workshop_case_escalation`. Select the three runs → **Compare**.
   Open the best run → **Artifacts** → `model` and look at the signature and `requirements.txt`.
5. **Step 4:** run the registration cell. The best model is registered to `adp_workshop.ml_<you>.case_escalation`
   and gets alias **`@champion`** (later runs only take `@champion` if they beat it; otherwise they get `@challenger`).

✅ **Checkpoint:** Catalog → your schema → **Models** → `case_escalation` shows version 1 with alias `champion`;
its **Lineage** tab shows the feature table. ROC AUC is roughly 0.70–0.80.

**Compute note:** serverless CPU is right for this model. For deep learning, choose **AI Runtime** (serverless
GPU) in the compute selector; classic **ML Runtime** clusters remain available when you need custom images.

---

## Module 04 · Batch inference (20 min) — *Serving*

1. Open `04_batch_inference`, attach **Serverless**, run the `%pip`, config and champion-lookup cells.
2. **Step 1:** score open cases with `mlflow.pyfunc.spark_udf` on `models:/…@champion`.
3. **Step 2:** write `case_predictions` (open cases plus the last 60 days of closed cases, which keep their label).
4. **Step 3:** ask Genie Code for the top-25 open cases to call first, and high-risk counts by line of business.

✅ **Checkpoint:** `case_predictions` exists; its Lineage tab shows the model and source tables.

---

## Module 05 · Model Serving (30 min, mostly waiting) — *Serving · Inference tables*

1. Open `05_model_serving` and run the config and SDK cells.
2. **Step 1:** create the endpoint `case-escalation-<you>` (about 10 minutes). Read the "while it builds" notes.
3. **Step 2:** query it with 5 open cases; you get escalation probabilities back.
4. **Step 3:** after a few queries, re-run the inference-table cell to see logged requests (they take ~5–10 minutes to land).
5. **Step 4:** on the endpoint page, open Genie Code → **Agent** → `Check the health of this endpoint`.

✅ **Checkpoint:** the endpoint is **Ready** on the **Serving** page and `case_escalation_endpoint_payload` appears in your schema.

**Stretch — Feature Serving:** publish `case_features` to an online store and serve a model logged with
`fe.log_model`, so the app sends only `case_id`.

---

## Module 06 · Monitoring (25 min) — *Monitoring*

1. Open `06_monitoring` and run the config cell.
2. **Step 1:** create the inference-log monitor on `case_predictions` (classification; prediction
   `predicted_escalation`, label `escalated`, model id `model_version`, timestamp `scored_at`, 1-day windows,
   sliced by line of business and channel).
3. **Step 2:** append a "storm surge" batch (bigger Home claims, more contacts, worse sentiment) and start a refresh.
4. **Step 3:** when the **Quality** tab shows the refresh succeeded, query the drift metrics (or ask Genie Code to explain them).
5. **Step 4:** create a SQL alert on `js_distance` for `claim_amount` above 0.2.

✅ **Checkpoint:** the Quality tab shows profile, drift and model-quality metrics and a dashboard; the drift you introduced is visible.

---

## Module 07 · Jobs & the model lifecycle (20 min) — *MLOps*

1. Open `07_mlops_jobs`, run the config cell and the setup cell.
2. **Step 1:** create the training job (`02` → `03`) and the nightly scoring job (`04`, created paused) — with the
   Genie Code prompt, the Reference cell, or by hand in **Jobs & Pipelines → Create → Job**.
3. **Step 2:** create and start the Lakeflow pipeline from `pipelines/case_features_pipeline.sql`; open it to see the
   graph and the data-quality results.
4. **Step 3:** start a training run, wait about 5 minutes, then list the model versions: the retrain registers v2
   as `@challenger` unless it beats the champion.
5. **Step 4:** discuss how drift alerts, reviewer promotion and Git pull requests close the loop.

✅ **Checkpoint:** **Jobs & Pipelines** shows `case-escalation-training-<you>`, `case-escalation-scoring-<you>` and
`case-features-pipeline-<you>`.

---

## Clean up after the workshop

```sql
DROP SCHEMA IF EXISTS adp_workshop.ml_<you> CASCADE;
```
Delete your serving endpoint and monitor, and run the clean-up cell at the end of `07_mlops_jobs` to remove your jobs and pipeline.

## Troubleshooting

| Symptom | Fix |
|---|---|
| Genie Code panel missing | Partner-powered AI features are off — ask the workspace admin. |
| `ModuleNotFoundError: databricks.feature_engineering` | Run the `%pip` cell at the top of the notebook, then continue after Python restarts. |
| Widgets reset after `%pip` | Expected — `%run ./_config` recreates them; re-enter a custom schema if you used one. |
| `RESOURCE_DOES_NOT_EXIST` for `@champion` | Run module 03 Step 4 first. |
| Endpoint stuck in *Not ready* | Open the endpoint → **Events** / **Logs**, or ask Genie Code `/diagnose`. |
| Inference table not found | It appears 5–10 minutes after the first logged request. |
| Monitor refresh slow | First refresh takes several minutes; check progress on the table's **Quality** tab. |
