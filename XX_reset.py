# Databricks notebook source
# MAGIC %run ./.setup

# COMMAND ----------

for row in spark.sql("show tables").collect():
    print(f"dropping {row.tableName}")
    spark.sql(f"drop table {row.tableName}")
