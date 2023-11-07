# Databricks notebook source
# MAGIC %md
# MAGIC # Prophet training
# MAGIC - This is an auto-generated notebook.
# MAGIC - To reproduce these results, attach this notebook to a cluster with runtime version **13.3.x-gpu-ml-scala2.12**, and rerun it.
# MAGIC - Compare trials in the [MLflow experiment](#mlflow/experiments/3031282363601977).
# MAGIC - Clone this notebook into your project folder by selecting **File > Clone** in the notebook toolbar.

# COMMAND ----------

import mlflow
import databricks.automl_runtime

target_col = "totalRecords_CDR"
time_col = "datetime"
unit = "hour"

id_cols = ["towerId"]

horizon = 168

# COMMAND ----------

# MAGIC %md
# MAGIC ## Load Data

# COMMAND ----------

import mlflow
import os
import uuid
import shutil
import pandas as pd
import pyspark.pandas as ps

# Create temp directory to download input data from MLflow
input_temp_dir = os.path.join(
    os.environ["SPARK_LOCAL_DIRS"], "tmp", str(uuid.uuid4())[:8])
os.makedirs(input_temp_dir)

# Download the artifact and read it into a pandas DataFrame
input_data_path = mlflow.artifacts.download_artifacts(
    run_id="66585519425940cb8db83c43aea6965a",
    artifact_path="data",
    dst_path=input_temp_dir)

input_file_path = os.path.join(input_data_path, "training_data")
input_file_path = "file://" + input_file_path
df_loaded = ps.from_pandas(pd.read_parquet(input_file_path))

# Preview data
df_loaded.head(5)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Aggregate data by `id_col` and `time_col`
# MAGIC Group the data by `id_col` and `time_col`, and take average if there are multiple `target_col` values in the same group.

# COMMAND ----------

group_cols = [time_col] + id_cols

df_aggregated = df_loaded.groupby(group_cols).agg(y=(target_col, "avg")).reset_index()

df_aggregated = df_aggregated.assign(ts_id=lambda x: x["towerId"].astype(str))

df_aggregated.head()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Train Prophet model
# MAGIC - Log relevant metrics to MLflow to track runs
# MAGIC - All the runs are logged under [this MLflow experiment](#mlflow/experiments/3031282363601977)
# MAGIC - Change the model parameters and re-run the training cell to log a different trial to the MLflow experiment

# COMMAND ----------

import logging

# disable informational messages from prophet
logging.getLogger("py4j").setLevel(logging.WARNING)

# COMMAND ----------

from pyspark.sql.types import *

df_schema = df_aggregated.to_spark().schema
result_columns = id_cols + [
    "model_json",
    "prophet_params",
    "start_time",
    "end_time",
    "mse",
    "rmse",
    "mae",
    "mape",
    "mdape",
    "smape",
    "coverage"]

result_schema = StructType(
    [StructField(id_col, df_schema[id_col].dataType) for id_col in id_cols]
    + [
        StructField("model_json", StringType()),
        StructField("prophet_params", StringType()),
        StructField("start_time", TimestampType()),
        StructField("end_time", TimestampType()),
        StructField("mse", FloatType()),
        StructField("rmse", FloatType()),
        StructField("mae", FloatType()),
        StructField("mape", FloatType()),
        StructField("mdape", FloatType()),
        StructField("smape", FloatType()),
        StructField("coverage", FloatType()),
    ]
)


def prophet_training(history_pd):
    from hyperopt import hp
    from databricks.automl_runtime.forecast.prophet.forecast import (
        ProphetHyperoptEstimator)

    seasonality_mode = ["additive", "multiplicative"]
    search_space = {
        "changepoint_prior_scale": hp.loguniform(
            "changepoint_prior_scale", -6.9, -0.69),
        "seasonality_prior_scale": hp.loguniform("seasonality_prior_scale", -6.9, 2.3),
        "holidays_prior_scale": hp.loguniform("holidays_prior_scale", -6.9, 2.3),
        "seasonality_mode": hp.choice("seasonality_mode", seasonality_mode)}
    country_holidays = "US"
    run_parallel = False

    hyperopt_estim = ProphetHyperoptEstimator(
        horizon=horizon,
        frequency_unit=unit,
        metric="smape",
        interval_width=0.8,
        country_holidays=country_holidays,
        search_space=search_space,
        num_folds=5,
        max_eval=10,
        trial_timeout=530,
        random_state=300783239,
        is_parallel=run_parallel)

    results_pd = hyperopt_estim.fit(history_pd)
    results_pd[id_cols] = history_pd[id_cols]
    results_pd["start_time"] = pd.Timestamp(history_pd["ds"].min())
    results_pd["end_time"] = pd.Timestamp(history_pd["ds"].max())

    return results_pd[result_columns]


def train_with_fail_safe(df):
    try:
        return prophet_training(df)
    except Exception as e:
        print(f"Encountered an exception while training timeseries: {repr(e)}")
        return pd.DataFrame(columns=result_columns)

# COMMAND ----------

import mlflow
from databricks.automl_runtime.forecast.prophet.model import (
    mlflow_prophet_log_model,
    MultiSeriesProphetModel,
)

with mlflow.start_run(
    experiment_id="3031282363601977", run_name="Prophet"
) as mlflow_run:
    mlflow.set_tag("estimator_name", "Prophet")
    mlflow.log_param("holiday_country", "US")
    mlflow.log_param("interval_width", 0.8)
    df_aggregated = df_aggregated.rename(columns={time_col: "ds"})

    forecast_results = (
        (
            df_aggregated.to_spark()
            .repartition(sc.defaultParallelism, "ts_id")
            .groupby("ts_id")
            .applyInPandas(train_with_fail_safe, result_schema)
        )
        .cache()
        .pandas_api()
    )
    results_pdf = forecast_results[
        id_cols + ["model_json", "start_time", "end_time"]
    ].to_pandas()
    results_pdf["ts_id"] = results_pdf[id_cols].astype(str).agg("-".join, axis=1)
    results_pdf["ts_id_tuple"] = results_pdf[id_cols].apply(tuple, axis=1)

    # Check whether every time series's model is trained
    ts_models_trained = set(results_pdf["ts_id"].unique().tolist())
    ts_ids = set(df_aggregated["ts_id"].unique().tolist())

    if len(ts_models_trained) == 0:
        raise Exception("Trial unable to train models for any identities. Please check the training cell for error details")

    if ts_ids != ts_models_trained:
        mlflow.log_param("partial_model", True)
        print(f"WARNING: Models not trained for the following identities: {ts_ids.difference(ts_models_trained)}")

    # Log the metrics to mlflow
    avg_metrics = (
        forecast_results[["mse", "rmse", "mae", "mape", "mdape", "smape", "coverage"]]
        .mean()
        .to_frame(name="mean_metrics")
        .reset_index())
    avg_metrics["index"] = "val_" + avg_metrics["index"].astype(str)
    avg_metrics.set_index("index", inplace=True)
    mlflow.log_metrics(avg_metrics.to_dict()["mean_metrics"])

    # Create mlflow prophet model
    results_pdf = results_pdf.set_index("ts_id_tuple")
    model_json = results_pdf["model_json"].to_dict()
    start_time = results_pdf["start_time"].to_dict()
    end_time = results_pdf["end_time"].to_dict()
    end_history_time = max(end_time.values())
    prophet_model = MultiSeriesProphetModel(
        model_json, start_time, end_history_time, horizon, unit, time_col, id_cols)

    # Generate sample input dataframe
    sample_input = df_loaded.head(1).to_pandas()
    sample_input[time_col] = pd.to_datetime(sample_input[time_col])
    sample_input.drop(columns=[target_col], inplace=True)

    mlflow_prophet_log_model(prophet_model, sample_input=sample_input)

# COMMAND ----------

forecast_results.head(5)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Analyze the predicted results

# COMMAND ----------

# Load the model
run_id = mlflow_run.info.run_id
loaded_model = mlflow.pyfunc.load_model(f"runs:/{run_id}/model")

# COMMAND ----------

model = loaded_model._model_impl.python_model
col_types = [StructField(f"{n}", FloatType()) for n in model.get_reserved_cols()]
col_types.append(StructField("ds",TimestampType()))
col_types.append(StructField("ts_id",StringType()))
result_schema = StructType(col_types)

future_df = model.make_future_dataframe(include_history=False)
future_df["ts_id"] = future_df[id_cols].apply(tuple, axis=1)
future_df = future_df.rename(columns={time_col: "ds"})
future_df.head()

# COMMAND ----------

# Predict future with the default horizon
forecast_pd = future_df.groupby(id_cols).apply(lambda df: model._predict_impl(df, model._horizon)).reset_index()

# COMMAND ----------

# Plotly plots is turned off by default because it takes up a lot of storage.
# Set this flag to True and re-run the notebook to see the interactive plots with plotly
use_plotly = False

# COMMAND ----------

# Choose a random id for plot
forecast_pd["ts_id"] = forecast_pd[id_cols].apply(tuple, axis=1)
id = set(forecast_pd.index.to_list()).pop()
ts_id = forecast_pd["ts_id"].loc[id]
# Get the prophet model for plot
model = loaded_model._model_impl.python_model.model(ts_id)
predict_pd = forecast_pd[forecast_pd["ts_id"] == ts_id]

# COMMAND ----------

# MAGIC %md
# MAGIC ### Plot the forecast with change points and trend
# MAGIC Plot the forecast using the `plot` method with your forecast dataframe. You can use `prophet.plot.add_changepoints_to_plot` to overlay significant changepoints. An interactive figure can be created with plotly.

# COMMAND ----------

from prophet.plot import add_changepoints_to_plot, plot_plotly

if use_plotly:
    fig = plot_plotly(model, predict_pd, changepoints=True, trend=True, figsize=(1200, 600))
else:
    fig = model.plot(predict_pd)
    a = add_changepoints_to_plot(fig.gca(), model, predict_pd)
fig

# COMMAND ----------

# MAGIC %md
# MAGIC ### Plot the forecast components
# MAGIC Use the `Prophet.plot_components` method to see the components. By default you'll see the trend, yearly seasonality, and weekly seasonality of the time series. You can also include holidays. An interactive figure can be created with plotly.

# COMMAND ----------

from prophet.plot import plot_components_plotly
if use_plotly:
    fig = plot_components_plotly(model, predict_pd, figsize=(900, 400))
    fig.show()
else:
    fig = model.plot_components(predict_pd)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Show the predicted results

# COMMAND ----------

predict_cols = id_cols + ["ds", "yhat"]
forecast_pd = forecast_pd.reset_index()
display(forecast_pd[predict_cols].tail(horizon))
