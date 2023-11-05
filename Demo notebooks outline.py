# Databricks notebook source
# MAGIC %md
# MAGIC (code from original dbdemo in folder `archive`)
# MAGIC Notebook 01 
# MAGIC - Create catalog and schema (using `cindy_test_catalog`)   
# MAGIC   `catalog = f'{current_user_no_at}_demo_catalog'`  
# MAGIC   `schema = 'telco_reliability'`
# MAGIC - Reads in data from S3 and save as table in UC (CRD data from S3 location, not using DLT or streaming for demo)
# MAGIC - See catalog https://e2-demo-field-eng.cloud.databricks.com/explore/data/cindy_test_catalog/telco_reliability?o=1444828305810485
# MAGIC
# MAGIC Notebook 02
# MAGIC - Feature engineering on `'cdr_stream_hour_gold'` table. Select only 05-01-2023 to 05-15-2023 and drop a column
# MAGIC - Save filtered table as a feature table in UC `'cdr_stream_hour_gold_features'`
# MAGIC - Use feature table and start a AutoML run to build forecast models
# MAGIC
# MAGIC Notebook 02.5 (optional)
# MAGIC - AutoML generated notebook for best prophet model
# MAGIC
# MAGIC Notebook 03
# MAGIC - Get experiment id and run id for the best model from AutoML 
# MAGIC - Register model to UC and add tags and descriptions
# MAGIC
# MAGIC Notebook 04
# MAGIC - Different deployment and inferening (pandas, spark, realtime)
# MAGIC
# MAGIC Notebook 04.5 (optional)
# MAGIC - Programmatically create real time serving endpoint 
# MAGIC - Test out real time scoring
# MAGIC
# MAGIC Notebook 05 (no job creation yet, more just instructions)
# MAGIC - Retraining workflow
# MAGIC

# COMMAND ----------


