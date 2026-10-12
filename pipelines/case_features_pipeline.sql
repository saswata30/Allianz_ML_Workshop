-- Lakeflow Declarative Pipeline: cleaned cases and features with data-quality expectations.
-- The source table is set by the pipeline configuration key `source_cases`.

CREATE OR REFRESH MATERIALIZED VIEW cases_clean (
  CONSTRAINT valid_case_id   EXPECT (case_id IS NOT NULL)            ON VIOLATION DROP ROW,
  CONSTRAINT positive_amount EXPECT (claim_amount > 0)               ON VIOLATION DROP ROW,
  CONSTRAINT valid_sentiment EXPECT (sentiment_score BETWEEN -1 AND 1),
  CONSTRAINT valid_days_open EXPECT (days_open >= 0)
)
COMMENT 'Claim cases that pass basic quality checks'
AS SELECT * FROM ${source_cases};

CREATE OR REFRESH MATERIALIZED VIEW case_features_pipeline
COMMENT 'Engineered case features (same logic as notebook 02), refreshed by Lakeflow'
AS SELECT
  case_id,
  line_of_business, channel, region, claim_amount, policy_tenure_years, prior_claims_12m,
  customer_age, num_contacts, days_open, sentiment_score,
  CAST(has_attorney AS INT)                               AS has_attorney,
  LN(1 + claim_amount)                                    AS log_claim_amount,
  num_contacts / GREATEST(days_open / 7.0, 1.0)           AS contacts_per_week,
  CAST(claim_amount > 50000 AS INT)                       AS is_high_value,
  CAST(sentiment_score < -0.25 AS INT)                    AS negative_sentiment,
  CAST(policy_tenure_years < 1 AS INT)                    AS new_customer
FROM cases_clean;
