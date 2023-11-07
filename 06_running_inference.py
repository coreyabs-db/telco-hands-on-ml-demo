# Databricks notebook source
# MAGIC %md
# MAGIC # Running Inference - Batch or serverless real-time
# MAGIC
# MAGIC
# MAGIC In previous notebook, we have saved our model to Unity Catalog.
# MAGIC
# MAGIC All we need to do now is use this model to run Inferences. A simple solution is to share the model name to our Data Engineering team and they'll be able to call this model within the pipeline they maintained. That's what we did in our Delta Live Table pipeline!
# MAGIC
# MAGIC Alternatively, this can be schedule in a separate job. Here is an example to show you how MLFlow can be directly used to retriver the model and run inferences.
# MAGIC
# MAGIC <!-- Collect usage data (view). Remove it to disable collection. View README for more details.  -->
# MAGIC <img width="1px" src="https://www.google-analytics.com/collect?v=1&gtm=GTM-NKQ8TT7&tid=UA-163989034-1&aip=1&t=event&ec=dbdemos&ea=VIEW&dp=%2F_dbdemos%2Flakehouse%2Flakehouse-retail-c360%2F04-Data-Science-ML%2F04.3-running-inference&cid=1444828305810485&uid=7635338147052150">

# COMMAND ----------

# MAGIC %md
# MAGIC ## TO BE COMPLETED 1)inferencing with spark 2)real time scoring.
# MAGIC notebook 04.5 deploys model to an endpoint but needs to test out scoring

# COMMAND ----------

# Setup
import re
current_user = dbutils.notebook.entry_point.getDbutils().notebook().getContext().tags().apply('user')
if current_user.rfind('@') > 0:
  current_user_no_at = current_user[:current_user.rfind('@')]
else:
  current_user_no_at = current_user
current_user_no_at = re.sub(r'\W+', '_', current_user_no_at)

# COMMAND ----------

catalog = f'{current_user_no_at}_demo_catalog'
catalog = "abs_dev"
schema = 'telco_reliability'
table_name = 'cdr_stream_hour_features'
model_name = f"telco_forecast_{current_user_no_at}"


# COMMAND ----------

import mlflow
from pyspark.sql.functions import pandas_udf, PandasUDFType
from pyspark.sql.types import *

spark.sql(f'USE CATALOG {catalog}')
spark.sql(f'USE SCHEMA {schema}')

#Use Databricks Unity Catalog to save our model
mlflow.set_registry_uri('databricks-uc')   

# COMMAND ----------

# MAGIC %md-sandbox
# MAGIC ##Deploying the model for batch inferences
# MAGIC
# MAGIC <img style="float: right; margin-left: 20px" width="600" src="https://github.com/QuentinAmbard/databricks-demo/raw/main/retail/resources/images/churn_batch_inference.gif" />
# MAGIC
# MAGIC Now that our model is available in the Registry, we can load it to compute our inferences and save them in a table to start building dashboards.
# MAGIC
# MAGIC We will use MLFlow function to load a pyspark UDF and distribute our inference in the entire cluster. If the data is small, we can also load the model with plain python and use a pandas Dataframe.
# MAGIC
# MAGIC If you don't know how to start, Databricks can generate a batch inference notebook in just one click from the model registry: Open MLFlow model registry and click the "User model for inference" button!

# COMMAND ----------

# MAGIC %md ### Scaling inferences using Spark 
# MAGIC We'll first see how it can be loaded as a spark UDF and called directly in a SQL function:

# COMMAND ----------



#            Alias
#                                                                                  Model name       |
# #Get packaged udf for prediction                                                                                        |          |
# make_forecast_udf = mlflow.pyfunc.spark_udf(spark, model_uri=f"models:/{catalog}.{schema}.{model_name}@prod", result_type='double')
# #Register the function to use in SQL
# spark.udf.register("make_forecast", make_forecast_udf)

# COMMAND ----------

# MAGIC %md ### Inference with pandas
# MAGIC

# COMMAND ----------

loaded_model = mlflow.pyfunc.load_model(model_uri=f"models:/{catalog}.{schema}.{model_name}@prod")

# COMMAND ----------

from pyspark.sql.types import *
target_col = "totalRecords_CDR"
time_col = "datetime"
unit = "hour"
id_cols = ["towerId"]
horizon = 168

model = loaded_model._model_impl.python_model
col_types = [StructField(f"{n}", FloatType()) for n in model.get_reserved_cols()]
col_types.append(StructField("ds",TimestampType()))
col_types.append(StructField("ts_id",StringType()))
result_schema = StructType(col_types)

future_df = model.make_future_dataframe(include_history=False)
future_df["ts_id"] = future_df[id_cols].apply(tuple, axis=1)
future_df = future_df.rename(columns={time_col: "ds"})
future_df.head()
forecast_pd = future_df.groupby(id_cols).apply(lambda df: model._predict_impl(df, model._horizon)).reset_index()
forecast_pd [['ds', 'yhat_lower', 'yhat_upper', 'yhat', 'towerId']].head()

# COMMAND ----------

# MAGIC %md-sandbox
# MAGIC
# MAGIC ## Realtime model serving with Databricks serverless serving
# MAGIC
# MAGIC <img style="float: right; margin-left: 20px" width="700" src="https://github.com/databricks-demos/dbdemos-resources/blob/main/images/retail/lakehouse-churn/lakehouse-c360-model-serving.png?raw=true" />
# MAGIC
# MAGIC Databricks also provides serverless serving.
# MAGIC
# MAGIC Click on model Serving, enable realtime serverless and your endpoint will be created, providing serving over REST api within a Click.
# MAGIC
# MAGIC Databricks Serverless offer autoscaling, including downscaling to zero when you don't have any traffic to offer best-in-class TCO while keeping low-latencies model serving.

# COMMAND ----------

# MAGIC %md
# MAGIC To deploy your serverless model, 
# MAGIC open the [Model Serving menu](https://e2-demo-tools.cloud.databricks.com/?o=1660015457675682#mlflow/endpoints), and select the model you registered within Unity Catalog, OR [run notebook to deploy model realtime]($./04.5[optional]_model_realtime_serving)

# COMMAND ----------

dataset = spark.table(table_name).select().limit(3).toPandas()
#Make it a string to send to the inference endpoint
# dataset['last_transaction'] = dataset['last_transaction'].astype(str)
dataset

# COMMAND ----------

# DBTITLE 1,Call the REST API deployed using standard python
import os
import requests
import numpy as np
import pandas as pd
import json

model_endpoint_name = model_name


def score_model(dataset):
  url = f'https://{dbutils.notebook.entry_point.getDbutils().notebook().getContext().browserHostName().get()}/serving-endpoints/{model_endpoint_name}/invocations'
  headers = {'Authorization': f'Bearer {dbutils.notebook.entry_point.getDbutils().notebook().getContext().apiToken().get()}', 'Content-Type': 'application/json'}
  ds_dict = {'dataframe_split': dataset.to_dict(orient='split')}
  data_json = json.dumps(ds_dict, allow_nan=True)
  response = requests.request(method='POST', headers=headers, url=url, data=data_json)
  if response.status_code != 200:
    raise Exception(f'Request failed with status {response.status_code}, {response.text}')
  return response.json()

#Deploy your model and uncomment to run your inferences live!
score_model(dataset)

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC # Next step: Leverage inferences and automate actions to increase revenue
# MAGIC
# MAGIC ## Automate action to reduce churn based on predictions
# MAGIC
# MAGIC We now have an end 2 end data pipeline analizing and predicting churn. We can now easily trigger actions to reduce the churn based on our business:
# MAGIC
# MAGIC - Send targeting email campaign to the customer the most likely to churn
# MAGIC - Phone campaign to discuss with our customers and understand what's going
# MAGIC - Understand what's wrong with our line of product and fixing it
# MAGIC
# MAGIC These actions are out of the scope of this demo and simply leverage the Churn prediction field from our ML model.
# MAGIC
# MAGIC ## Track churn impact over the next month and campaign impact
# MAGIC
# MAGIC Of course, this churn prediction can be re-used in our dashboard to analyse future churn and measure churn reduction. 
# MAGIC
# MAGIC The pipeline created with the Lakehouse will offer a strong ROI: it took us a few hours to setup this pipeline end 2 end and we have potential gain for $129,914 / month!
# MAGIC
# MAGIC <img width="800px" src="https://raw.githubusercontent.com/QuentinAmbard/databricks-demo/main/retail/resources/images/lakehouse-retail/lakehouse-retail-churn-dbsql-prediction-dashboard.png">
# MAGIC
# MAGIC <a href='/sql/dashboards/e9cbce37-da29-482d-962c-5b63dc5fac3b'>Open the Churn prediction DBSQL dashboard</a> | [Go back to the introduction]($../00-churn-introduction-lakehouse)
