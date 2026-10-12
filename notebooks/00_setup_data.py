# Databricks notebook source
# MAGIC %md
# MAGIC # 00 · Setup — create your schema and the workshop data
# MAGIC
# MAGIC **Platform layer:** Governance & storage (Unity Catalog + Delta Lake)
# MAGIC
# MAGIC This notebook creates your personal schema and two Delta tables of **synthetic** Allianz-style claim
# MAGIC cases, so the workshop runs the same way in any workspace:
# MAGIC
# MAGIC | Table | Rows | What it holds |
# MAGIC |---|---|---|
# MAGIC | `cases` | 60,000 | Closed cases from the last 18 months, with the `escalated` outcome (the label) |
# MAGIC | `open_cases` | 3,000 | Cases opened in the last 14 days, still open — no label yet. We score these in module 04 |
# MAGIC
# MAGIC **Business question:** which new claim cases are likely to be **escalated** (complaint, ombudsman, or
# MAGIC senior-handler referral), so the team can intervene early?
# MAGIC
# MAGIC > If your facilitator has set the `source_table` widget to a curated ADP table, skip the data
# MAGIC > generation step — only the schema is created.
# MAGIC
# MAGIC **Run time:** about 1 minute on serverless compute.
# MAGIC
# MAGIC ---
# MAGIC ### 🗺️ How to follow this notebook
# MAGIC * **Compute:** at the top right, attach **Serverless**.
# MAGIC * **Run cells one at a time, top to bottom** — click inside a cell and press `Shift + Enter`.
# MAGIC * **Symbols:** 🧞 = ask Genie Code first · ▶ = run the next cell · 👀 = what you should see · ✅ = check before moving on.
# MAGIC * **Stuck?** Run the cell marked `# Reference` under each prompt — it always works, so you can never fall behind.
# MAGIC
# MAGIC ### Your path (about 5 minutes)
# MAGIC 1. ▶ Easiest option: click **Run all** at the top of the notebook, then read the results below.
# MAGIC 2. Config cell → 👀 prints your schema name, for example `adp_workshop.ml_jane_doe`. **Write it down.**
# MAGIC 3. *Create your schema* → 👀 `Schema ready: …`
# MAGIC 4. *Generate synthetic claim cases* (about 1 minute) → 👀 `Wrote 60,000 labelled cases … (escalation rate ~18%)` and `Wrote 3,000 open cases`.
# MAGIC 5. *Document the tables* → 👀 a preview of 10 claim cases.
# MAGIC 6. ✅ In **Catalog** (left bar) open your schema: you see `cases` and `open_cases`.

# COMMAND ----------

# MAGIC %run ./_config

# COMMAND ----------

# MAGIC %md ## 1 · Create your schema in Unity Catalog

# COMMAND ----------

spark.sql(f"CREATE SCHEMA IF NOT EXISTS {FQ} COMMENT 'Allianz ML workshop — personal schema for {CURRENT_USER}'")
print(f"Schema ready: {FQ}")

# COMMAND ----------

# MAGIC %md ## 2 · Generate synthetic claim cases
# MAGIC
# MAGIC The escalation label is drawn from a known logistic function of the features (large claims, attorney
# MAGIC involvement, negative sentiment, many contacts and slow handling all raise the risk), so the model has a
# MAGIC real signal to learn and the results are repeatable (`seed = 42`).

# COMMAND ----------

import numpy as np
import pandas as pd
from datetime import datetime, timedelta

USE_SYNTHETIC = not dbutils.widgets.get("source_table").strip()

def make_cases(n, start, end, rng, id_offset=0, labelled=True):
    lob = rng.choice(["Motor", "Home", "Commercial Property", "Liability", "Travel"], n, p=[.38, .27, .15, .12, .08])
    channel = rng.choice(["Broker", "Direct Online", "Call Centre", "Agent"], n, p=[.35, .25, .25, .15])
    region = rng.choice(["Midwest", "Northeast", "South", "West"], n)
    base_amount = {"Motor": 6_000, "Home": 12_000, "Commercial Property": 45_000, "Liability": 30_000, "Travel": 1_500}
    claim_amount = np.round(np.array([base_amount[l] for l in lob]) * rng.lognormal(0, 0.8, n), 2)
    tenure = np.clip(rng.gamma(2.0, 2.5, n), 0, 30).round(1)
    prior_claims = rng.poisson(0.6, n)
    customer_age = np.clip(rng.normal(47, 14, n), 18, 90).astype(int)
    num_contacts = 1 + rng.poisson(2.2, n)
    days_open = np.clip(rng.gamma(2.0, 9.0, n), 1, 180).astype(int)
    sentiment = np.clip(rng.normal(0.1, 0.45, n), -1, 1).round(3)
    has_attorney = rng.random(n) < (0.04 + 0.10 * (lob == "Liability"))
    seconds = rng.uniform(0, (end - start).total_seconds(), n)
    opened_at = [start + timedelta(seconds=float(s)) for s in seconds]

    df = pd.DataFrame({
        "case_id": [f"CASE-{id_offset + i:07d}" for i in range(n)],
        "opened_at": opened_at,
        "line_of_business": lob,
        "channel": channel,
        "region": region,
        "claim_amount": claim_amount,
        "policy_tenure_years": tenure,
        "prior_claims_12m": prior_claims,
        "customer_age": customer_age,
        "num_contacts": num_contacts,
        "days_open": days_open,
        "sentiment_score": sentiment,
        "has_attorney": has_attorney,
    })
    if labelled:
        logit = (-2.25
                 + 0.55 * np.log1p(claim_amount / 10_000)
                 + 1.6 * has_attorney
                 - 1.4 * sentiment
                 + 0.22 * (num_contacts - 3)
                 + 0.018 * (days_open - 20)
                 + 0.30 * prior_claims
                 - 0.05 * tenure
                 + 0.35 * (channel == "Call Centre"))
        df["escalated"] = (rng.random(n) < 1 / (1 + np.exp(-logit))).astype(int)
    return df

if USE_SYNTHETIC:
    rng = np.random.default_rng(42)
    now = datetime.now().replace(microsecond=0)
    hist = make_cases(60_000, now - timedelta(days=548), now - timedelta(days=15), rng)
    open_ = make_cases(3_000, now - timedelta(days=14), now, rng, id_offset=60_000, labelled=False)

    spark.createDataFrame(hist).write.mode("overwrite").option("overwriteSchema", "true").saveAsTable(CASES_TABLE)
    spark.createDataFrame(open_).write.mode("overwrite").option("overwriteSchema", "true").saveAsTable(OPEN_CASES_TABLE)
    print(f"Wrote {len(hist):,} labelled cases to {CASES_TABLE} (escalation rate {hist.escalated.mean():.1%})")
    print(f"Wrote {len(open_):,} open cases to {OPEN_CASES_TABLE}")
else:
    print(f"Using curated source table {CASES_TABLE} — no synthetic data generated.")

# COMMAND ----------

# MAGIC %md ## 3 · Document the tables (Unity Catalog)
# MAGIC Comments and tags make the data discoverable and give Genie Code better context in the next module.

# COMMAND ----------

if USE_SYNTHETIC:
    spark.sql(f"COMMENT ON TABLE {CASES_TABLE} IS 'Closed claim cases with escalation outcome (synthetic, workshop)'")
    spark.sql(f"COMMENT ON TABLE {OPEN_CASES_TABLE} IS 'Open claim cases awaiting an escalation-risk score (synthetic, workshop)'")
    spark.sql(f"ALTER TABLE {CASES_TABLE} ALTER COLUMN escalated COMMENT '1 = case was escalated (complaint, ombudsman or senior referral)'")
    spark.sql(f"ALTER TABLE {CASES_TABLE} SET TAGS ('workshop' = 'allianz_ml', 'data_class' = 'synthetic')")

display(spark.table(CASES_TABLE).limit(10))

# COMMAND ----------

# MAGIC %md
# MAGIC ## ✅ Checkpoint
# MAGIC * Catalog Explorer shows your schema with `cases` and `open_cases`.
# MAGIC * The escalation rate printed above is roughly 15–25%.
# MAGIC
# MAGIC **Next:** `01_eda_genie_code`.
