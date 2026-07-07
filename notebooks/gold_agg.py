# Databricks notebook source
# Gold aggregation layer

catalog = "xe_training_catalog"
schema = "capstone"
spark.sql(f"USE CATALOG {catalog}")
spark.sql(f"USE SCHEMA {schema}")

silver_table = f"{catalog}.{schema}.cap_silver"
gold_table = f"{catalog}.{schema}.cap_gold_carrier_delay"

spark.sql(f"""
CREATE OR REPLACE TABLE {gold_table}
CLUSTER BY (carrier, flight_date)
AS
SELECT
  flight_date, carrier, origin,
  COUNT(*) AS flight_count,
  ROUND(AVG(dep_delay_min), 2) AS avg_dep_delay_min,
  ROUND(AVG(arr_delay_min), 2) AS avg_arr_delay_min,
  SUM(CASE WHEN dep_delay_min > 60 THEN 1 ELSE 0 END) AS long_delay_count,
  ROUND(SUM(CASE WHEN dep_delay_min > 60 THEN 1 ELSE 0 END) * 100.0 / COUNT(*), 2) AS long_delay_pct
FROM {silver_table}
GROUP BY flight_date, carrier, origin
""")

print(f"Gold: {gold_table} ({spark.table(gold_table).count()} rows)")
