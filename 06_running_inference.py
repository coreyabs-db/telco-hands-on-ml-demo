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

# MAGIC %run ./.setup

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
# MAGIC We'll first see how it can be loaded as a spark UDF and called directly in a SQL function. 
# MAGIC
# MAGIC     # Alias - Model name - Get packaged udf for prediction
# MAGIC     model_uri = f"models:/{catalog}.{schema}.{model_name}@Champion"
# MAGIC     make_forecast_udf = mlflow.pyfunc.spark_udf(
# MAGIC       spark, model_uri=model_uri, result_type='double')
# MAGIC
# MAGIC     # Register the function to use in SQL
# MAGIC     spark.udf.register("make_forecast", make_forecast_udf)
# MAGIC
# MAGIC We've disabled it here for now, but this gives you a rough example of how to go about it.

# COMMAND ----------

# MAGIC %md ### Inference with pandas
# MAGIC

# COMMAND ----------

model_name = f"telco_forecast_{current_user_no_at}"
model_uri = f"models:/{catalog}.{schema}.{model_name}@Champion"
loaded_model = mlflow.pyfunc.load_model(model_uri=model_uri)

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

model = loaded_model._model_impl.python_model
col_types = [StructField(f"{n}", FloatType()) for n in model.get_reserved_cols()]
col_types.append(StructField("ds",TimestampType()))
col_types.append(StructField("ts_id",StringType()))
result_schema = StructType(col_types)

future_df = model.make_future_dataframe(include_history=False)
future_df["ts_id"] = future_df[id_cols].apply(tuple, axis=1)
future_df = future_df.rename(columns={time_col: "ds"})
display(future_df)

# COMMAND ----------

# Predict future with the default horizon
forecast_pd = future_df.groupby(id_cols).apply(lambda df: model._predict_impl(df, model._horizon)).reset_index()
display(forecast_pd)
