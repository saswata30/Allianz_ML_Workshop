# Suggested Genie Code instructions for the workshop

Genie Code can follow standing **user instructions** (and, for teams, shared skills). Add the text below as your
instructions from the Genie Code panel's settings — see *Customize Genie Code with instructions* in the Azure
Databricks docs if you can't find the option. It makes Genie Code's suggestions follow the ADP ML pattern used
in this workshop. Remove it afterwards if you don't want it for other work.

```text
You are helping with the Allianz ML workshop on Azure Databricks.

Conventions
- Use PySpark for data preparation and pandas only after sampling or aggregating.
- All tables, feature tables and models live in Unity Catalog under the catalog and schema printed by
  notebooks/_config.py (variables CATALOG, SCHEMA, FQ). Never write to other schemas.
- Register features with databricks.feature_engineering.FeatureEngineeringClient; primary key is case_id.
- The label is `escalated` (1 = escalated). Never use the label, or columns derived from it, as a feature.
- Track every training run in MLflow with ROC AUC, PR AUC, F1 and recall on a held-out test set,
  and log models with a signature and input example.
- Register models to Unity Catalog (mlflow.set_registry_uri("databricks-uc")) and use the aliases
  @champion and @challenger rather than stage names.
- Prefer serverless compute. Explain any cell that writes or overwrites a table before running it.
- Keep code readable: short functions, clear variable names, a one-line comment per cell.
```
