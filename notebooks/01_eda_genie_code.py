# Databricks notebook source
# MAGIC %md
# MAGIC # 01 · EDA & development with Genie Code
# MAGIC
# MAGIC **Platform layer:** EDA & development — *Genie Code for ML* · *Notebooks & IDE integration*
# MAGIC
# MAGIC In this module you explore the case data **by asking Genie Code**, the AI coding agent built into
# MAGIC Databricks notebooks. Genie Code reads your Unity Catalog metadata (table and column comments), writes
# MAGIC and runs cells, reads the output, and fixes its own errors.
# MAGIC
# MAGIC ### How to use Genie Code in this workshop
# MAGIC 1. Open the **Genie Code** side panel (sparkle icon in the notebook). In the lower-right corner of the panel,
# MAGIC    select **Agent** so it can plan, write and run cells for you.
# MAGIC 2. Paste the prompt from each **🧞 Genie Code** box below. Reference tables with **`@table_name`** or the
# MAGIC    **Add context** button — Genie Code then reads the table and column comments from Unity Catalog.
# MAGIC 3. Answer any clarifying questions about its plan. When it wants to run code, click **Allow** (or **Skip**);
# MAGIC    once you trust the flow you can switch on **Auto-approve**. Use **Continue** / **Reject** to steer, and the
# MAGIC    red stop icon to halt.
# MAGIC 4. Compare its output with the **Reference** cell underneath. If they differ, ask Genie Code *why*.
# MAGIC 5. Keep this browser tab open while Genie Code works on a multi-step task.
# MAGIC
# MAGIC > **Requirement:** *Partner-powered AI features* must be enabled for the account and workspace (your admin
# MAGIC > will have done this before the workshop). Genie Code only sees data your Unity Catalog permissions allow.
# MAGIC
# MAGIC > **Human in the loop:** Genie Code proposes; you approve. Always read the code before accepting it —
# MAGIC > that is the habit we want for production ML.
# MAGIC
# MAGIC **Time:** 25 minutes.

# COMMAND ----------

# MAGIC %run ./_config

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 1 · Profile the table
# MAGIC
# MAGIC > 🧞 **Genie Code prompt**
# MAGIC > ```
# MAGIC > Profile the table {CASES_TABLE}. Show row count, null counts per column, and summary
# MAGIC > statistics for numeric columns. Then show the overall escalation rate.
# MAGIC > ```
# MAGIC > *(Replace `{CASES_TABLE}` with the name printed by the config cell above.)*

# COMMAND ----------

# Reference
cases = spark.table(CASES_TABLE)
print(f"Rows: {cases.count():,}")
display(cases.summary())
display(cases.groupBy().avg(LABEL_COL).withColumnRenamed(f"avg({LABEL_COL})", "escalation_rate"))

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 2 · Which segments escalate most?
# MAGIC
# MAGIC > 🧞 **Genie Code prompt**
# MAGIC > ```
# MAGIC > For each line_of_business and channel in this table, show the number of cases and the
# MAGIC > escalation rate, sorted by escalation rate. Plot escalation rate by line_of_business as a bar chart.
# MAGIC > ```

# COMMAND ----------

# Reference
from pyspark.sql import functions as F

seg = (cases.groupBy("line_of_business", "channel")
       .agg(F.count("*").alias("cases"), F.round(F.avg(LABEL_COL), 3).alias("escalation_rate"))
       .orderBy(F.desc("escalation_rate")))
display(seg)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 3 · Which numeric signals matter?
# MAGIC
# MAGIC > 🧞 **Genie Code prompt**
# MAGIC > ```
# MAGIC > Compare the distributions of claim_amount, sentiment_score, num_contacts and days_open
# MAGIC > for escalated vs non-escalated cases. Use a sample of 10,000 rows, show box plots, and
# MAGIC > summarise in two sentences which features look most predictive.
# MAGIC > ```

# COMMAND ----------

# Reference
import matplotlib.pyplot as plt

pdf = cases.sample(fraction=10_000 / max(cases.count(), 1), seed=7).toPandas()
cols = ["claim_amount", "sentiment_score", "num_contacts", "days_open"]
fig, axes = plt.subplots(1, len(cols), figsize=(18, 4))
for ax, c in zip(axes, cols):
    pdf.boxplot(column=c, by=LABEL_COL, ax=ax, showfliers=False)
    ax.set_title(c); ax.set_xlabel("escalated")
plt.suptitle("Numeric features by escalation outcome")
plt.tight_layout()
display(fig)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 4 · Ask Genie Code to propose features
# MAGIC
# MAGIC > 🧞 **Genie Code prompt**
# MAGIC > ```
# MAGIC > Based on this EDA, propose 5 engineered features for predicting escalation (for example
# MAGIC > ratios or flags). For each one give the PySpark expression and one sentence on why it should help.
# MAGIC > Do not create any tables yet.
# MAGIC > ```
# MAGIC
# MAGIC Keep the list Genie Code gives you — in module 02 we build the feature table. The reference feature set is:
# MAGIC
# MAGIC | Feature | Expression | Why |
# MAGIC |---|---|---|
# MAGIC | `log_claim_amount` | `log1p(claim_amount)` | Amounts are heavily skewed |
# MAGIC | `contacts_per_week` | `num_contacts / greatest(days_open/7, 1)` | Frequent chasing signals frustration |
# MAGIC | `is_high_value` | `claim_amount > 50000` | Large claims get more scrutiny |
# MAGIC | `negative_sentiment` | `sentiment_score < -0.25` | Explicit dissatisfaction |
# MAGIC | `new_customer` | `policy_tenure_years < 1` | Less loyalty, less patience |

# COMMAND ----------

# MAGIC %md
# MAGIC ## Optional · Work from your IDE
# MAGIC The same notebooks can be edited in VS Code or Cursor with the **Databricks extension** (or Databricks
# MAGIC Connect), keeping code in Git. Module 07 shows how the repo deploys as a bundle.
# MAGIC
# MAGIC ## ✅ Checkpoint
# MAGIC * You can explain which segments escalate most, and you have a list of candidate features.
# MAGIC
# MAGIC **Next:** `02_feature_engineering`.
