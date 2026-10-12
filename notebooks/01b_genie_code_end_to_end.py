# Databricks notebook source
# MAGIC %md
# MAGIC # 01b · Genie Code challenge — from one request to a trained model
# MAGIC
# MAGIC **Platform layer:** EDA & development — *Genie Code for ML*
# MAGIC
# MAGIC This module mirrors the Databricks demo *"Genie Code for machine learning"*: you give Genie Code **one
# MAGIC request**, review its **plan**, and let it explore the data, analyse features, train an **XGBoost** model and
# MAGIC generate predictions — all from the Genie Code panel.
# MAGIC
# MAGIC Then, in modules 02–07, you rebuild the same model the governed, repeatable way (Feature Store, Unity
# MAGIC Catalog registry, serving, monitoring, jobs). The comparison is the lesson: **Genie Code gets you to a
# MAGIC first model in minutes; the platform pattern makes it production-grade.**
# MAGIC
# MAGIC **Time:** 20 minutes. Use a fresh, empty notebook for the agent's work, and keep this one open for the prompts.
# MAGIC
# MAGIC ---
# MAGIC ### 🗺️ How to follow this notebook
# MAGIC * **Compute:** at the top right, attach **Serverless**.
# MAGIC * **Run cells one at a time, top to bottom** — click inside a cell and press `Shift + Enter`.
# MAGIC * **Symbols:** 🧞 = ask Genie Code first · ▶ = run the next cell · 👀 = what you should see · ✅ = check before moving on.
# MAGIC * **Stuck?** Run the cell marked `# Reference` under each prompt — it always works, so you can never fall behind.
# MAGIC
# MAGIC ### Your path (20 minutes)
# MAGIC 1. ▶ Run the config cell.
# MAGIC 2. Create a new empty notebook: **+ New → Notebook**, name it `genie_code_scratch`, attach **Serverless**.
# MAGIC 3. In that notebook open **Genie Code → Agent**, paste the Step 1 prompt, and type `@cases` / `@open_cases` to attach the tables.
# MAGIC 4. Read the plan Genie Code shows you → answer its questions → click **Allow** for each step (or turn on **Auto-approve**).
# MAGIC 5. 👀 After a few minutes: data exploration, feature importance, an XGBoost model with ROC AUC and recall, and the 10 riskiest open cases.
# MAGIC 6. Try the Step 2 follow-up prompts, then discuss the Step 3 debrief table as a group.
# MAGIC 7. ✅ You have a working model built from one prompt — and a list of what is still missing for production.

# COMMAND ----------

# MAGIC %run ./_config

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 1 · Give Genie Code the whole task
# MAGIC
# MAGIC 1. Create a new notebook called `genie_code_scratch` and attach serverless compute.
# MAGIC 2. Open the **Genie Code** side panel, select **Agent** in the lower-right corner, and paste the prompt below.
# MAGIC    Type `@` and pick your `cases` and `open_cases` tables so Genie Code has the right context.
# MAGIC
# MAGIC > 🧞 **Genie Code prompt**
# MAGIC > ```
# MAGIC > I want to predict which claim cases will be escalated. Use @cases; the label is the column
# MAGIC > escalated. Explore the data, analyse which features matter most, train an XGBoost classifier
# MAGIC > with a proper train/test split, report ROC AUC and recall, and track the run in MLflow.
# MAGIC > Finally, generate predictions for the cases in @open_cases and show the 10 highest-risk cases.
# MAGIC > Show me your plan before you start.
# MAGIC > ```
# MAGIC
# MAGIC 3. **Review the plan.** Answer its clarifying questions and ask for changes if needed (for example *"also
# MAGIC    handle the categorical columns with one-hot encoding"*).
# MAGIC 4. Click **Allow** as each cell is proposed (or turn on **Auto-approve**). When a cell fails, notice how it
# MAGIC    reads the error and fixes its own code. Keep the tab open until it finishes.
# MAGIC
# MAGIC > This replaces the old AutoML step: Databricks positions **Genie Code for ML** for customised model
# MAGIC > development, and AI Functions in SQL (such as `ai_forecast`) for simple cases.

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 2 · Interrogate the result
# MAGIC Follow-up prompts to try in the same Genie Code conversation:
# MAGIC
# MAGIC * `Which features had the highest importance, and do they make business sense for claims?`
# MAGIC * `Is there any risk of data leakage in what you built? Check days_open in particular.`
# MAGIC * `Tune the XGBoost model with a small hyperparameter search and log every trial to MLflow.`
# MAGIC * `Plot the ROC curve and a confusion matrix at a 0.5 threshold.`
# MAGIC
# MAGIC ## Step 3 · Debrief (5 minutes, as a group)
# MAGIC | Question | Why it matters for ADP |
# MAGIC |---|---|
# MAGIC | Where did the features live? | Ad-hoc dataframes cannot be reused or served — module 02 fixes this |
# MAGIC | Is the model registered and governed? | Only Unity Catalog models can be served and audited — module 03 |
# MAGIC | How would a claims app call it? | Model Serving with inference tables — module 05 |
# MAGIC | Who notices when it degrades? | Data Quality Monitoring and alerts — module 06 |
# MAGIC | How does it reach production? | Scheduled jobs and a retraining loop — module 07 |
# MAGIC
# MAGIC **Next:** `02_feature_engineering`.
