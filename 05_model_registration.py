# Databricks notebook source
# MAGIC %run ./.setup

# COMMAND ----------

# MAGIC %md
# MAGIC ## Get best model from AutoML Experiment

# COMMAND ----------

# AutoML experiment path
xp_path = f"/Users/{current_user}/databricks_automl/{schema}"

print(f"xp_path: {xp_path}")

# COMMAND ----------

# Let's get our auto ml experiment using the path defined in previous notebook. 
# This is specific to the demo, it gets the experiment ID of the last Auto ML run and the best model.
client = MlflowClient()

#Get experiment ID
filter_string = f"name like '{xp_path}/%'"
experiment_id = client.search_experiments(filter_string=filter_string)[0].experiment_id

#Get best model run_id
best_model = client.search_runs(experiment_ids=[experiment_id], order_by=["metrics.val_smape ASC"], max_results=1, filter_string="status = 'FINISHED'")[0]
run_id = best_model.info.run_id

print('Best Model: ', best_model.data.tags['estimator_name'], best_model.data.params)
print('Best run ID: ', run_id)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Register model to UC

# COMMAND ----------

model_name = f"telco_forecast_{current_user_no_at}"
model_uri = f"runs:/{run_id}/model"
tablename = "cdr_stream_hour_features"

# Register model in UC
try:
    # Get the model if it is already registered to avoid re-deploying the endpoint
    latest_model = client.get_model_version_by_alias(f"{catalog}.{schema}.{model_name}", "Champion")
    print(f"Our model is already deployed on UC: {catalog}.{schema}.{model_name}")
except:
    # Enable Unity Catalog with mlflow registry
    # Add model within our catalog
    latest_model = mlflow.register_model(model_uri, f"{catalog}.{schema}.{model_name}")

    # Flag it as Production ready using UC Aliases
    client.set_tag(run_id, key="db_table", value=f"{catalog}.{schema}.{tablename}")
    client.set_registered_model_alias(
        name=f"{catalog}.{schema}.{model_name}",
        alias="Champion",
        version=latest_model.version)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Add model version descriptions using the API
# MAGIC You can use MLflow APIs to find the recently trained model version, then add descriptions to the model version and the registered model:

# COMMAND ----------

def get_latest_model_version(model_name):
    client = MlflowClient()
    model_version_infos = client.search_model_versions("name = '%s'" % model_name)
    return max([int(model_version_info.version) for model_version_info in model_version_infos])

# COMMAND ----------

# The main model description, typically done once.
client.update_registered_model(
    name=f"{catalog}.{schema}.{model_name}",
    description="This model forecasts the total call records for the next 72 hours for each cell tower.")

# Gives more details on this specific model version
client.update_model_version(
    name=f"{catalog}.{schema}.{model_name}",
    version=get_latest_model_version(model_name=f"{catalog}.{schema}.{model_name}"),
    description=f"This model version was built using AutoML. The best model uses {best_model.data.tags['estimator_name']} ")

# COMMAND ----------

# MAGIC %md
# MAGIC ### View the model in the UI
# MAGIC You can view and manage registered models and model versions in Unity Catalog using Catalog Explorer ([AWS](https://docs.databricks.com/data/index.html)|[Azure](https://learn.microsoft.com/azure/databricks/data/)|[GCP](https://docs.gcp.databricks.com/data/index.html)).
# MAGIC Look for the model you just created under your catalog and `telco_reliability` schema.
