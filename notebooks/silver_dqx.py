# Databricks notebook source
# Silver layer with DQX quality checks + quarantine

from pyspark.sql import functions as F

catalog = "xe_training_catalog"
schema = "capstone"
spark.sql(f"USE CATALOG {catalog}")
spark.sql(f"USE SCHEMA {schema}")

bronze_table = f"{catalog}.{schema}.cap_bronze"
silver_table = f"{catalog}.{schema}.cap_silver"
quarantine_table = f"{catalog}.{schema}.cap_silver_quarantine"

bronze_df = spark.table(bronze_table)

quality_rules = [
    ("delay_null", F.col("dep_delay_min").isNull()),
    ("cancelled", F.lower(F.col("cancelled")).isin(["true", "1"])),
    ("missing_carrier", F.col("carrier").isNull()),
    ("extreme_delay", F.col("dep_delay_min") > 720),
]

flagged = bronze_df
for name, expr in quality_rules:
    flagged = flagged.withColumn(f"_dq_{name}", expr)

flagged = flagged.withColumn(
    "_dq_failed_rules",
    F.concat_ws(",", *[
        F.when(F.col(f"_dq_{name}"), F.lit(name)).otherwise(F.lit(None))
        for name, _ in quality_rules
    ])
)

clean = flagged.filter(F.col("_dq_failed_rules") == "") \
    .drop(*[f"_dq_{n}" for n, _ in quality_rules]).drop("_dq_failed_rules") \
    .withColumn("flight_date", F.to_date("flight_date")) \
    .withColumn("dep_delay_min", F.col("dep_delay_min").cast("int")) \
    .withColumn("arr_delay_min", F.col("arr_delay_min").cast("int"))

quarantined = flagged.filter(F.col("_dq_failed_rules") != "") \
    .withColumn("_quarantined_at", F.current_timestamp())

clean.write.mode("overwrite").option("overwriteSchema", "true").saveAsTable(silver_table)
quarantined.write.mode("overwrite").option("overwriteSchema", "true").saveAsTable(quarantine_table)

print(f"Silver: {silver_table} ({spark.table(silver_table).count()} rows)")
print(f"Quarantine: {quarantine_table} ({spark.table(quarantine_table).count()} rows)")
