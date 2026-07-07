# Databricks notebook source
# Bronze ingestion via Auto Loader

catalog = "xe_training_catalog"
schema = "capstone"
spark.sql(f"USE CATALOG {catalog}")
spark.sql(f"USE SCHEMA {schema}")

landing_volume = f"/Volumes/{catalog}/databricks/flights_landing"
bronze_table = f"{catalog}.{schema}.cap_bronze"
checkpoint = f"/Volumes/{catalog}/{schema}/checkpoints/cap_bronze"

(
    spark.readStream.format("cloudFiles")
    .option("cloudFiles.format", "csv")
    .option("cloudFiles.schemaLocation", f"{checkpoint}/_schema")
    .option("header", "true")
    .option("cloudFiles.inferColumnTypes", "true")
    .load(landing_volume)
    .writeStream
    .format("delta")
    .option("checkpointLocation", checkpoint)
    .trigger(availableNow=True)
    .toTable(bronze_table)
)

spark.streams.awaitAnyTermination()
print(f"Bronze: {bronze_table} ({spark.table(bronze_table).count()} rows)")
