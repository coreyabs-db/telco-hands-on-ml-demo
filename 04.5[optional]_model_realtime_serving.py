# Databricks notebook source
# MAGIC %md
# MAGIC
# MAGIC # Create a model serving endpoint with Python
# MAGIC
# MAGIC This notebook covers wrapping the REST API queries for model serving endpoint creation, updating endpoint configuration based on model version, and endpoint deletion with Python for your Python model serving workflows.
# MAGIC
# MAGIC Learn more about model serving on Databricks ([AWS](https://docs.databricks.com/machine-learning/model-serving/create-manage-serving-endpoints.html) | [Azure](https://learn.microsoft.com/azure/databricks/machine-learning/model-inference/serverless/create-manage-serverless-endpoints)). 
# MAGIC
# MAGIC ## Requirements
# MAGIC
# MAGIC Databricks Runtime ML 12.0 or above
# MAGIC

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
schema = 'telco_reliability'
table_name = 'cdr_stream_hour_features' #feature table
model_name = f"telco_forecast_{current_user_no_at}"

model_serving_endpoint_name = model_name

# COMMAND ----------

# MAGIC %md
# MAGIC ## Get token and model version 
# MAGIC
# MAGIC  The following section demonstrates how to provide both a token for the API, which can be obtained from the notebook and how to get the latest model version you plan to serve and deploy.

# COMMAND ----------

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
import mlflow
from mlflow.tracking.client import MlflowClient
def get_latest_model_version(model_name: str, catalog: str, schema: str):
  client = MlflowClient()
  model = client.get_model_version_by_alias(f"{catalog}.{schema}.{model_name}", "prod")
  #models = client.get_latest_versions(model_name, stages=["None"])
  new_model_version = model.version
  return new_model_version

# COMMAND ----------

# MAGIC %md
# MAGIC ## Set up configurations

# COMMAND ----------

import requests
mlflow.set_registry_uri('databricks-uc')
my_json = {
  "name": model_serving_endpoint_name,
  "config": {
   "served_models": [{
     "model_name": f"{catalog}.{schema}.{model_name}",
     "model_version": get_latest_model_version(model_name=model_name, catalog=catalog, schema=schema),
     "workload_size": "Small",
     "scale_to_zero_enabled": True
   }]
 }
}

# COMMAND ----------

# MAGIC %md
# MAGIC The following defines Python functions that:
# MAGIC - create a model serving endpoint
# MAGIC - update a model serving endpoint configuration with the latest model version
# MAGIC - delete a model serving endpoint

# COMMAND ----------

def func_create_endpoint(model_serving_endpoint_name):
  #get endpoint status
  endpoint_url = f"https://{instance}/api/2.0/serving-endpoints"
  url = f"{endpoint_url}/{model_serving_endpoint_name}"
  r = requests.get(url, headers=headers)
  if "RESOURCE_DOES_NOT_EXIST" in r.text:  
    print("Creating this new endpoint: ", f"https://{instance}/serving-endpoints/{model_serving_endpoint_name}/invocations")
    re = requests.post(endpoint_url, headers=headers, json=my_json)
  else:
    new_model_version = (my_json['config'])['served_models'][0]['model_version']
    print("This endpoint existed previously! We are updating it to a new config with new model version: ", new_model_version)
    # update config
    url = f"{endpoint_url}/{model_serving_endpoint_name}/config"
    re = requests.put(url, headers=headers, json=my_json['config']) 
    # wait till new config file in place
    import time,json
    #get endpoint status
    url = f"https://{instance}/api/2.0/serving-endpoints/{model_serving_endpoint_name}"
    retry = True
    total_wait = 0
    while retry:
      r = requests.get(url, headers=headers)
      assert r.status_code == 200, f"Expected an HTTP 200 response when accessing endpoint info, received {r.status_code}"
      endpoint = json.loads(r.text)
      if "pending_config" in endpoint.keys():
        seconds = 10
        print("New config still pending")
        if total_wait < 6000:
          #if less the 10 mins waiting, keep waiting
          print(f"Wait for {seconds} seconds")
          print(f"Total waiting time so far: {total_wait} seconds")
          time.sleep(10)
          total_wait += seconds
        else:
          print(f"Stopping,  waited for {total_wait} seconds")
          retry = False  
      else:
        print("New config in place now!")
        retry = False
  assert re.status_code == 200, f"Expected an HTTP 200 response, received {re.status_code}"
  
def func_delete_model_serving_endpoint(model_serving_endpoint_name):
  endpoint_url = f"https://{instance}/api/2.0/serving-endpoints"
  url =  f"{endpoint_url}/{model_serving_endpoint_name}" 
  response = requests.delete(url, headers=headers)
  if response.status_code != 200:
    raise Exception(f"Request failed with status {response.status_code}, {response.text}")
  else:
    print(model_serving_endpoint_name, "endpoint is deleted!")
  #return response.json()

# COMMAND ----------

func_create_endpoint(model_serving_endpoint_name)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Wait for end point to be ready
# MAGIC
# MAGIC The `wait_for_endpoint()` function defined in the following command gets and returns the serving endpoint status.  

# COMMAND ----------

#GET /api/2.0/serving-endpoints/{name}

import time, mlflow

def wait_for_endpoint():
    endpoint_url = f"https://{instance}/api/2.0/serving-endpoints"
    while True:
        url =  f"{endpoint_url}/{model_serving_endpoint_name}"
        response = requests.get(url, headers=headers)
        assert response.status_code == 200, f"Expected an HTTP 200 response, received {response.status_code}\n{response.text}"

        status = response.json().get("state", {}).get("ready", {})
        #print("status",status)
        if status == "READY": print(status); print("-"*80); return
        else: print(f"Endpoint not ready ({status}), waiting 10 seconds"); time.sleep(10) # Wait 10 seconds
        
api_url = mlflow.utils.databricks_utils.get_webapp_url()
#print(api_url)

wait_for_endpoint()

# Give the system just a couple extra seconds to transition
time.sleep(5)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Score the model (TO BE COMPLETED)
# MAGIC
# MAGIC The following command defines the `score_model()` function  and an example scoring request under the `payload_json` variable.

# COMMAND ----------

dataset = spark.table(f'{catalog}.{schema}.{table_name}').limit(3).toPandas()
#Make it a string to send to the inference endpoint
# dataset['last_transaction'] = dataset['last_transaction'].astype(str)
dataset

# COMMAND ----------

def score_model(dataset):
  import json
  url = f'https://{dbutils.notebook.entry_point.getDbutils().notebook().getContext().browserHostName().get()}/serving-endpoints/{model_serving_endpoint_name}/invocations'
  headers = {'Authorization': f'Bearer {dbutils.notebook.entry_point.getDbutils().notebook().getContext().apiToken().get()}', 'Content-Type': 'application/json'}
  ds_dict = {'dataframe_split': dataset.to_dict(orient='split')}
  data_json = json.dumps(ds_dict, allow_nan=True)
  response = requests.request(method='POST', headers=headers, url=url, data=data_json)
  if response.status_code != 200:
    raise Exception(f'Request failed with status {response.status_code}, {response.text}')
  return response.json()

# COMMAND ----------

spark.sql(f'USE CATALOG {catalog}')
spark.sql(f'USE SCHEMA {schema}')

sql_statement = '''
  SELECT
    towerId,
    datetime,
    sum(totalRecords_CDR) as totalRecords_CDR
  FROM cdr_stream_hour_gold_table
  WHERE datetime >= '2023-04-01' and datetime < '2023-04-15'
  GROUP BY towerId, datetime
  ORDER BY towerId, datetime
'''

payload = (spark
                                   .sql(sql_statement)
                                   .repartition(sc.defaultParallelism, ["towerId"])
                                ).cache().toPandas()

# COMMAND ----------

payload

# COMMAND ----------

print(score_model(payload))

# COMMAND ----------

# MAGIC %md
# MAGIC ## Delete the endpoint

# COMMAND ----------

# func_delete_model_serving_endpoint(model_serving_endpoint_name)
