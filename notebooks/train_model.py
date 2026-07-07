# Databricks notebook source
# MLflow-tracked RandomForest model on gold data

import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score

catalog = "xe_training_catalog"
schema = "capstone"
spark.sql(f"USE CATALOG {catalog}")
spark.sql(f"USE SCHEMA {schema}")

silver_table = f"{catalog}.{schema}.cap_silver"
model_name = f"{catalog}.{schema}.cap_delay_predictor"

mlflow.set_registry_uri("databricks-uc")

df = spark.sql(f"""
SELECT carrier, origin,
       DATE_FORMAT(flight_date, 'E') AS day_of_week,
       avg_dep_delay_min, flight_count,
       CASE WHEN long_delay_pct > 20 THEN 1 ELSE 0 END AS long_delay_flag
FROM (SELECT flight_date, carrier, origin,
             COUNT(*) AS flight_count,
             AVG(dep_delay_min) AS avg_dep_delay_min,
             SUM(CASE WHEN dep_delay_min > 60 THEN 1 ELSE 0 END) * 100.0 / COUNT(*) AS long_delay_pct
      FROM {silver_table}
      GROUP BY flight_date, carrier, origin)
""").toPandas()

df = pd.get_dummies(df, columns=["carrier", "origin", "day_of_week"], drop_first=True)
X = df.drop(columns=["long_delay_flag"])
y = df["long_delay_flag"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=30, stratify=y)

with mlflow.start_run(run_name="capstone_rf") as run:
    model = RandomForestClassifier(n_estimators=50, max_depth=6, random_state=30)
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    acc = accuracy_score(y_test, preds)
    f1 = f1_score(y_test, preds, zero_division=0)
    mlflow.log_metric("accuracy", acc)
    mlflow.log_metric("f1", f1)
    mlflow.sklearn.log_model(sk_model=model, artifact_path="model",
                             registered_model_name=model_name,
                             input_example=X_train.head(3))

print(f"Model: {model_name}")
print(f"Accuracy: {acc:.3f}, F1: {f1:.3f}")
