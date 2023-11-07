# Databricks notebook source
# MAGIC %md-sandbox
# MAGIC # Deploying and orchestrating the full workflow
# MAGIC
# MAGIC <img style="float: right; margin-left: 10px" width="300px" src="https://raw.githubusercontent.com/QuentinAmbard/databricks-demo/main/retail/resources/images/lakehouse-retail/lakehouse-retail-churn-5.png" />
# MAGIC
# MAGIC All our assets are ready. We now need to define when we want our ETL pipeline to kick in and refresh the tables.
# MAGIC
# MAGIC In our case, we decided that the best tradoff is to ingest new data every hours:
# MAGIC
# MAGIC - Start the ETL job to ingest new data and refresh our tables
# MAGIC - Refresh the DBSQL dashboard if any (and potentially notify downstream applications)
# MAGIC - Retrain our model to include the lastest date and capture potential behavior change
# MAGIC
# MAGIC <!-- Collect usage data (view). Remove it to disable collection. View README for more details.  -->
# MAGIC <img width="1px" src="https://www.google-analytics.com/collect?v=1&gtm=GTM-NKQ8TT7&tid=UA-163989034-1&aip=1&t=event&ec=dbdemos&ea=VIEW&dp=%2F_dbdemos%2Flakehouse%2Flakehouse-retail-c360%2F05-Workflow-orchestration%2F05-Workflow-orchestration-churn&cid=1444828305810485&uid=7635338147052150">

# COMMAND ----------

# MAGIC %md-sandbox
# MAGIC ## Orchestrating our Forecast pipeline with Databricks Workflows
# MAGIC
# MAGIC <img style="float: right; margin-left: 10px" width="600px" src="https://www.databricks.com/wp-content/uploads/2022/05/workflows-orchestrate-img.png" />
# MAGIC
# MAGIC With Databricks Lakehouse, no need for external orchestrator. We can use [Workflows](/#job/list) (available on the left menu) to orchestrate our Churn pipeline within a few click.
# MAGIC
# MAGIC
# MAGIC
# MAGIC ###  Orchestrate anything anywhere
# MAGIC With workflow, you can run diverse workloads for the full data and AI lifecycle on any cloud. Orchestrate Delta Live Tables and Jobs for SQL, Spark, notebooks, dbt, ML models and more.
# MAGIC
# MAGIC ### Simple - Fully managed
# MAGIC Remove operational overhead with a fully managed orchestration service, so you can focus on your workflows not on managing your infrastructure.
# MAGIC
# MAGIC ### Proven reliability
# MAGIC Have full confidence in your workflows leveraging our proven experience running tens of millions of production workloads daily across AWS, Azure and GCP.

# COMMAND ----------

# MAGIC %md-sandbox
# MAGIC
# MAGIC ## Creating your workflow
# MAGIC
# MAGIC <img style="float: right; margin-left: 10px" width="600px" src="https://raw.githubusercontent.com/QuentinAmbard/databricks-demo/main/retail/resources/images/lakehouse-retail/lakehouse-retail-churn-workflow.png" />
# MAGIC
# MAGIC A Databricks Workflow is composed of Tasks.
# MAGIC
# MAGIC Each task can trigger a specific job:
# MAGIC
# MAGIC * Delta Live Tables
# MAGIC * SQL query / dashboard
# MAGIC * Model retraining / inference
# MAGIC * Notebooks
# MAGIC * dbt
# MAGIC * ...
# MAGIC
# MAGIC In this example, can see our 3 tasks:
# MAGIC
# MAGIC * Start the ETL pipeline to ingest new data and refresh our tables
# MAGIC * Refresh the DBSQL dashboard if any(and potentially notify downstream applications)
# MAGIC * Retrain our Forecasting model

# COMMAND ----------

# DBTITLE 1,Template script to create a job from notebook 
from databricks.sdk import WorkspaceClient
from databricks.sdk.service.jobs import Task, Note
bookTask, Source

w = WorkspaceClient()

job_name            = input("Some short name for the job (for example, my-job): ")
description         = input("Some short description for the job (for example, My job): ")
existing_cluster_id = input("ID of the existing cluster in the workspace to run the job on (for example, 1234-567890-ab123cd4): ")
notebook_path       = input("Workspace path of the notebook to run (for example, /Users/someone@example.com/my-notebook): ")
task_key            = input("Some key to apply to the job's tasks (for example, my-key): ")

print("Attempting to create the job. Please wait...\n")

j = w.jobs.create(
  job_name = job_name,
  tasks = [
    Task(
      description = description,
      existing_cluster_id = existing_cluster_id,
      notebook_task = NotebookTask(
        base_parameters = dict(""),
        notebook_path = notebook_path,
        source = Source("WORKSPACE")
      ),
      task_key = task_key
    )
  ]
)

print(f"View the job at {w.config.host}/#job/{j.job_id}\n")

# COMMAND ----------

# MAGIC %md-sandbox
# MAGIC
# MAGIC ## Monitoring your runs
# MAGIC
# MAGIC <img style="float: right; margin-left: 10px" width="600px" src="https://raw.githubusercontent.com/QuentinAmbard/databricks-demo/main/retail/resources/images/lakehouse-retail/lakehouse-retail-churn-workflow-monitoring.png" />
# MAGIC
# MAGIC Once your workflow is created, we can access historical runs and receive alerts if something goes wrong!
# MAGIC
# MAGIC In the screenshot we can see that our workflow had multiple errors, with different runtime, and ultimately got fixed.
# MAGIC
# MAGIC Workflow monitoring includes errors, abnormal job duration and more advanced control!

# COMMAND ----------

# MAGIC %md
# MAGIC ## Conclusion
# MAGIC
# MAGIC Not only Datatabricks Lakehouse let you ingest, analyze and infer churn, it also provides a best-in-class orchestrator to offer your business fresh insight making sure everything works as expected!
# MAGIC
# MAGIC [Go back to introduction]($../00-churn-introduction-lakehouse)
