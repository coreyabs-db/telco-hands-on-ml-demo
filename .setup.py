# Databricks notebook source
import pyspark.sql.functions as F
import logging

# disable informational messages from prophet, etc...
logging.getLogger("py4j").setLevel(logging.WARNING)
logging.getLogger("cmdstanpy").setLevel(logging.WARNING)

# COMMAND ----------

# Setup
import re

current_user = (
    dbutils.notebook.entry_point.getDbutils()
    .notebook()
    .getContext()
    .tags()
    .apply("user"))

if current_user.rfind("@") > 0:
    current_user_no_at = current_user[: current_user.rfind("@")]
else:
    current_user_no_at = current_user

current_user_no_at = re.sub(r"\W+", "_", current_user_no_at)

# COMMAND ----------

catalog = "telco"
reliability_schema = "reliability"
churn_schema = "churn"
user_schema = current_user_no_at
schema = user_schema

spark.sql(f"USE CATALOG {catalog}")
spark.sql(f"CREATE SCHEMA IF NOT EXISTS {user_schema}")
spark.sql(f"USE SCHEMA {user_schema}")

print(f"using catalog {catalog}")
print(f"using user_schema {user_schema}")
print(f"using reliability_schema {reliability_schema}")
print(f"using churn_schema {churn_schema}")

# COMMAND ----------

source_path = "s3a://db-gtm-industry-solutions/data/CME/telco"

print(f"using source path {source_path}")

# COMMAND ----------

import mlflow
from mlflow.tracking.client import MlflowClient

# Use Databricks Unity Catalog to save our model
mlflow.set_registry_uri('databricks-uc')

print("using unity catalog for model registry")

# COMMAND ----------

# MAGIC %config InlineBackend.figure_format = "retina"
