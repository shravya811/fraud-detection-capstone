# Azure guide (Section 3)

These steps use the **Azure CLI** for setup and the **Python scripts in `mlops/`** for everything
else. I could not run any Azure step while preparing this project, so run each step, check the
result, and send the exact error if something fails.

## Cost warning (read first)
- Creating a workspace, resource group, storage account and Key Vault costs little or nothing by itself.
- A **compute instance bills while running** - stop it when you are not using it.
- The **compute cluster** is set to scale to 0 nodes, so it costs nothing while idle.
- A **managed online deployment bills for its VM the whole time it exists**, even with no traffic.
  Delete the endpoint as soon as you have your screenshots.
- Check current prices and your VM quota on the Azure pricing page and in the portal before creating anything.

## Step 1 - Install tools and sign in
```powershell
pip install -r requirements-azure.txt
winget install -e --id Microsoft.AzureCLI      # or download Azure CLI from Microsoft
az login
az extension add -n ml
az account show --query id -o tsv               # your subscription id
```

## Step 2 - Resource group and workspace
```powershell
az group create --name rg-fraud-capstone --location centralindia
az ml workspace create --name mlw-fraud-capstone --resource-group rg-fraud-capstone --location centralindia
```
The workspace creates its supporting **storage account** (data, models, logs), **Key Vault**
(secrets) and **Application Insights** (monitoring). The **container registry** is created
automatically the first time a model image is built (during deployment).

Portal alternative: portal.azure.com -> **Create a resource** -> search **Azure Machine Learning**
-> **Create** -> pick the subscription, create the resource group, give a workspace name, region
**Central India**, **Review + create**.

**Screenshot 1:** portal -> your resource group -> the list of resources.

## Step 3 - Environment variables (PowerShell, every new terminal)
```powershell
$env:AZURE_SUBSCRIPTION_ID = "<paste your subscription id>"
$env:AZURE_RESOURCE_GROUP  = "rg-fraud-capstone"
$env:AZURE_WORKSPACE_NAME  = "mlw-fraud-capstone"
```
Do not paste these values into chat or commit them to Git.

## Step 4 - Compute
```powershell
python -m mlops.create_compute --instance-name fraud-ci-<yourname>
```
**Screenshot 2:** ml.azure.com -> your workspace -> **Compute** (instance + cluster).

## Step 5 - Log runs to MLflow (after `src.train_models` finished)
```powershell
python -m mlops.log_to_mlflow --local     # optional dry run, writes to ./mlruns
python -m mlops.log_to_mlflow             # real run, logs to Azure ML
```
Open ml.azure.com -> **Jobs** -> experiment **fraud-detection-capstone**. Tick all runs ->
**Compare** (metrics, parameters, charts).
**Screenshot 3:** the runs list. **Screenshot 4:** the compare view. **Screenshot 5:** one run's
**Outputs + logs** tab showing `plots/` and `model`.

## Step 6 - Register the best model
```powershell
python -m mlops.register_best_model
python -m mlops.register_best_model --promote 1     # after you are happy: tag Production
```
**Screenshot 6:** ml.azure.com -> **Models** -> `fraud-detection-model` (versions and tags).

Local `.pkl` = one file on your disk. MLflow model = that model plus its environment and code
stored with a run. Registered model = a named, versioned entry in the workspace that deployments refer to.

## Step 7 - Deploy to a managed online endpoint
```powershell
python -m mlops.deploy_endpoint --endpoint-name fraud-ep-<yourname> --model-version 1 --deployment blue
```
Takes several minutes. Application Insights logging is switched on in the script.
**Screenshot 7:** ml.azure.com -> **Endpoints** -> your endpoint -> **Details** (deployment `blue`,
100% traffic, instance type and count).

## Step 8 - Test the endpoint
```powershell
python -m mlops.test_endpoint --endpoint-name fraud-ep-<yourname>
```
**Screenshot 8:** the terminal output, or Endpoints -> **Test** tab.

FastAPI as a client (new terminal, same project):
```powershell
$env:AZURE_ENDPOINT_URL = "<scoring URI printed by test_endpoint>"
$env:AZURE_ENDPOINT_KEY = "<primary key from Endpoints -> Consume tab>"
uvicorn api.main:app --reload
```
Open http://127.0.0.1:8000/docs and call **POST /predict/azure**.
**Screenshot 9:** that call in Swagger UI.

## Step 9 - Application Insights
Endpoint -> **Monitoring** (or the Application Insights resource -> **Metrics / Failures /
Performance**). Request count, latency and failures appear a few minutes after traffic.
**Screenshot 10:** the metrics chart.

## Step 10 - Monitoring, retraining, blue/green
- Drift and performance: `python -m mlops.monitoring --build-reference`, then
  `python -m mlops.monitoring --check recent.csv`.
- Retraining triggers: any feature PSI >= 0.25, recall drop of more than 0.10 on newly labelled
  data, or a scheduled monthly retrain.
- Blue/green: deploy the new model as `green` (starts at 0% traffic), test it, then
  `python -m mlops.shift_traffic --endpoint-name <name> --blue 90 --green 10`, and later
  `--blue 0 --green 100`. **Rollback** = `--blue 100 --green 0`.
  To deploy green: re-run `deploy_endpoint` with `--deployment green --model-version 2`.

## Step 11 - Clean up (stops all charges)
```powershell
python -m mlops.deploy_endpoint --endpoint-name fraud-ep-<yourname> --delete
az ml compute stop --name fraud-ci-<yourname> --resource-group rg-fraud-capstone --workspace-name mlw-fraud-capstone
az group delete --name rg-fraud-capstone --yes     # removes EVERYTHING, only when fully finished
```

## Most likely things to need a fix
1. The scoring image/Python versions in `mlops/deploy_endpoint.py` (`BASE_IMAGE`, `python=3.11`).
2. MLflow version compatibility with `azureml-mlflow` (send the traceback if tracking fails).
3. VM size availability or quota in your region (change `--vm-size` / `--instance-type`).
