# Databricks notebook source
# MAGIC %md # Train and register a Scikit learn model for model serving
# MAGIC
# MAGIC This notebook trains an ElasticNet model using the diabetes dataset from scikit learn. Databricks autologging is also used to both log metrics and to register the trained model to the Databricks Model Registry.
# MAGIC
# MAGIC After running the code in this notebook, you have a registered model ready for model inference with Databricks Model Serving ([AWS](https://docs.databricks.com/machine-learning/model-serving/index.html)|[Azure](https://docs.microsoft.com/azure/databricks/machine-learning/model-serving/index)).

# COMMAND ----------

# MAGIC %run ./.setup

# COMMAND ----------

# MAGIC %md
# MAGIC ## Import libraries

# COMMAND ----------

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import ElasticNet
from sklearn import datasets

# Import mlflow
import mlflow
import mlflow.sklearn

# COMMAND ----------

# MAGIC %md
# MAGIC ## Load data

# COMMAND ----------

# Load Diabetes datasets
diabetes = datasets.load_diabetes()
diabetes_X = diabetes.data
diabetes_y = diabetes.target

# Create pandas DataFrame for sklearn ElasticNet linear_model
diabetes_Y = np.array([diabetes_y]).transpose()
d = np.concatenate((diabetes_X, diabetes_Y), axis=1)
cols = diabetes.feature_names + ['progression']
diabetes_data = pd.DataFrame(d, columns=cols)
# Split the data into training and test sets. (0.75, 0.25) split.
train, test = train_test_split(diabetes_data)

# The predicted column is "progression" which is a quantitative measure of disease progression one year after baseline
train_x = train.drop(["progression"], axis=1)
test_x = test.drop(["progression"], axis=1)
train_y = train[["progression"]]
test_y = test[["progression"]]

# COMMAND ----------

# MAGIC %md
# MAGIC ## Log and register the model
# MAGIC
# MAGIC The following code automattically logs the trained model. By specifying `registered_model_name` in the autologging configuration, the model trained is automatically registered to the Databricks Model Registry. 

# COMMAND ----------

model_name = f"{catalog}.{schema}.diabetes_example"
mlflow.sklearn.autolog(
  log_input_examples=True, 
  registered_model_name=model_name)
alpha = 0.05
l1_ratio = 0.05

# Run ElasticNet
lr = ElasticNet(alpha=alpha, l1_ratio=l1_ratio, random_state=42)
lr.fit(train_x, train_y)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Register the model using the UI
# MAGIC
# MAGIC Now open the experiments tab and find the logged model. When you open the model run, you should see a button to register the model. Just click that and register the model to Unity Catalog.

# COMMAND ----------

# MAGIC %md
# MAGIC ## Create serving endpoint
# MAGIC
# MAGIC Now open the serving endpoints tab and click create endpoint.
# MAGIC
# MAGIC Select the same model from Unity Catalog and select whichever version you want to use. 
# MAGIC
# MAGIC Note: for this session, no need to actually click the final create button. We can simply share the one I've already created for us.

# COMMAND ----------

# MAGIC %md
# MAGIC Note: for both of these steps, there are [API's](https://docs.databricks.com/api/workspace/servingendpoints) you can use to automate this step. We just wanted to give you some exposure to the UI as well.

# COMMAND ----------

# MAGIC %md
# MAGIC ## Query the endpoint

# COMMAND ----------

model_serving_endpoint_name = "diabetes-example-endpoint"

# COMMAND ----------

import requests

token = dbutils.notebook.entry_point.getDbutils().notebook().getContext().apiToken().getOrElse(None)

# With the token, you can create our authorization header for our subsequent REST calls
headers = {
    "Authorization": f"Bearer {token}",
    "Content-Type": "application/json"
  }

# Next you need an endpoint at which to execute your request which you can get from the notebook's tags collection
java_tags = dbutils.notebook.entry_point.getDbutils().notebook().getContext().tags()

# This object comes from the Java CM - Convert the Java Map opject to a Python dictionary
tags = sc._jvm.scala.collection.JavaConversions.mapAsJavaMap(java_tags)

# Lastly, extract the Databricks instance (domain name) from the dictionary
instance = tags["browserHostName"]


def score_model(data_json: dict):
    url =  f"https://{instance}/serving-endpoints/{model_serving_endpoint_name}/invocations"
    response = requests.request(method="POST", headers=headers, url=url, json=data_json)
    if response.status_code != 200:
        raise Exception(f"Request failed with status {response.status_code}, {response.text}")
    return response.json()
  
payload_json = { "dataframe_records": test[:5].to_dict(orient="records") }

score_model(payload_json)
