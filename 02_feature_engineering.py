# Databricks notebook source
# MAGIC %run ./.setup

# COMMAND ----------

# MAGIC %md
# MAGIC # Load Data
# MAGIC We use the aggregated CDR data (`cdr_stream_hour_gold`) from our Unity Catalog to create hourly forecast

# COMMAND ----------

# DBTITLE 1,Read in gold table from Unity Catalog
hour_gold = spark.table('cdr_hour_gold')

# COMMAND ----------

# MAGIC %md
# MAGIC # Create feature table

# COMMAND ----------

# Filter data to only keep desired timeframe and drop columns we don't want to use in our model
start_date = "2023-05-01"
end_date = "2023-05-15"

hour_features = (
    hour_gold
    .filter(
        (F.col("datetime") >= start_date) &
        (F.col("datetime") < end_date))
    .drop("window"))

display(hour_features)

# COMMAND ----------

# MAGIC %md-sandbox
# MAGIC
# MAGIC # Write to Feature Store
# MAGIC
# MAGIC <img src="https://github.com/QuentinAmbard/databricks-demo/raw/main/product_demos/mlops-end2end-flow-feature-store.png" style="float:right" width="500" />
# MAGIC
# MAGIC Once our features are ready, we'll save them in Databricks Feature Store. Under the hood, features store are backed by a Delta Lake table.
# MAGIC
# MAGIC This will allow discoverability and reusability of our feature accross our organization, increasing team efficiency.
# MAGIC
# MAGIC Feature store will bring traceability and governance in our deployment, knowing which model is dependent of which set of features. It also simplify realtime serving.
# MAGIC
# MAGIC Make sure you're using the "Machine Learning" menu to have access to your feature store using the UI.

# COMMAND ----------

from databricks.feature_store import FeatureStoreClient

fs = FeatureStoreClient()
tablename = "cdr_hour_features"
try:
    # drop table if exists
    fs.drop_table(f"{catalog}.{schema}.{tablename}")
except:
    pass

# Note: You might need to delete the FS table using the UI
feature_table = fs.create_table(
    name=f"{catalog}.{schema}.{tablename}",
    primary_keys=["towerId", "datetime"],
    timestamp_keys=["datetime"],
    schema=hour_features.schema,
    description=f"These features are derived from the cdr_stream_hour_gold table in the lakehouse. We filtered datetime to be from {start_date} to {end_date}.")

fs.write_table(
    df=hour_features, name=f"{catalog}.{schema}.{tablename}", mode="overwrite")

features = fs.read_table(f"{catalog}.{schema}.{tablename}")
display(features)

# COMMAND ----------

# MAGIC %md-sandbox
# MAGIC
# MAGIC ## Accelerating forecasting model creation using MLFlow and Databricks Auto-ML
# MAGIC
# MAGIC MLflow is an open source project allowing model tracking, packaging and deployment. Everytime your datascientist team work on a model, Databricks will track all the parameter and data used and will save it. This ensure ML traceability and reproductibility, making it easy to know which model was build using which parameters/data.
# MAGIC
# MAGIC ### A glass-box solution that empowers data teams without taking away control
# MAGIC
# MAGIC While Databricks simplify model deployment and governance (MLOps) with MLFlow, bootstraping new ML projects can still be long and inefficient. 
# MAGIC
# MAGIC Instead of creating the same boilerplate for each new project, Databricks Auto-ML can automatically generate state of the art models for Classifications, regression, and forecast.
# MAGIC
# MAGIC
# MAGIC <img width="1000" src="https://github.com/QuentinAmbard/databricks-demo/raw/main/retail/resources/images/auto-ml-full.png"/>
# MAGIC
# MAGIC
# MAGIC Models can be directly deployed, or instead leverage generated notebooks to boostrap projects with best-practices, saving you weeks of efforts.
# MAGIC
# MAGIC <br style="clear: both">
# MAGIC
# MAGIC <img style="float: right" width="600" src="https://github.com/QuentinAmbard/databricks-demo/raw/main/retail/resources/images/churn-auto-ml.png"/>
# MAGIC
# MAGIC ### Using Databricks Auto ML with our aggregated hourly CDR dataset
# MAGIC
# MAGIC Auto ML is available in the "Machine Learning" space. All we have to do is start a new Auto-ML experimentation and select the feature table we just created (`cdr_stream_hour_features`)
# MAGIC
# MAGIC Our prediction target is the `totalRecords_CDR` column.
# MAGIC
# MAGIC Click on Start, and Databricks will do the rest.
# MAGIC
# MAGIC While this is done using the UI, you can also leverage the [python API](https://docs.databricks.com/applications/machine-learning/automl.html#automl-python-api-1)

# COMMAND ----------

# DBTITLE 1,Start a AutoML run 
from databricks import automl
from datetime import datetime

xp_path = f"/Users/{current_user}/databricks_automl/{schema}"
xp_name = f"automl_{schema}_{datetime.now().strftime("%Y-%m-%d_%H:%M:%S")}"
automl_run = automl.forecast(
    experiment_name=xp_name,
    experiment_dir=xp_path,
    dataset=fs.read_table(f"{catalog}.{schema}.{tablename}"),
    target_col="totalRecords_CDR",
    timeout_minutes=10,
    time_col="datetime",
    country_code="US",
    frequency="h",
    horizon=168,
    identity_col="towerId",
    output_database=f"{catalog}.{user_schema}",
    primary_metric="smape") 

