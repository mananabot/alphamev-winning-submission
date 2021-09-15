import time

from azureml.core import (
    Workspace, Datastore, Dataset,
    ComputeTarget, Environment,
    Experiment
)
from azureml.pipeline.steps import PythonScriptStep
from azureml.pipeline.core import Pipeline, PipelineData
from azureml.data import OutputFileDatasetConfig
from azureml.core.runconfig import RunConfiguration
from azureml.core.authentication import InteractiveLoginAuthentication

# import ngrok_utils


# Run params
dataset_name = "test-features"
compute_target_name = "mev-compute"
source_dir = "./src_train"
script_name = "train.py"

# Auth
tenant_id = "8b12d3ed-2283-4fb2-8fca-e894e67e99f0"
interactive_auth = InteractiveLoginAuthentication(tenant_id)

# Fetch workspace
ws = Workspace.from_config(auth=interactive_auth)

# Initialize conda environment from config file
env = Environment.from_conda_specification(
    name='env',
    file_path='./.azureml/env_train.yml'
)

# Fetch compute
compute_target = ComputeTarget(workspace=ws, name=compute_target_name)

# # Start ngrok tunnel
# host, port = ngrok_utils.start_tunnel("8000")
#
# # Pass pycharm debug host and port as environment variables
# env.environment_variables = {
#     'PYCHARM_DEBUG_PORT': port,
#     'PYCHARM_DEBUG_HOST': host
# }
#
# Setup run config
run_config = RunConfiguration()
run_config.target = compute_target
run_config.environment = env
run_config.docker.use_docker = True

# Load dataset from default datastore
datastore = Datastore.get_default(ws)
# try:
#     dataset = Dataset.get_by_name(ws, dataset_name)
# except:
dataset_path = [(datastore, dataset_name)]
dataset = Dataset.File.from_files(dataset_path).register(ws, dataset_name)

# Define names/inputs/outputs
dataset_input_name = 'features'
dataset_input = dataset.as_named_input(dataset_input_name)

output = OutputFileDatasetConfig(
    name='report',
    destination=(datastore, "report")
).as_upload(overwrite=True).register_on_complete('report')

# Define pipeline
pipeline_step = PythonScriptStep(
    script_name=script_name,
    source_directory=source_dir,
    arguments=[
        "--input_name", dataset_input_name,
        "--output", output,
    ],
    inputs=[dataset_input],
    outputs=[output],
    compute_target=compute_target,
    runconfig=run_config,
    allow_reuse=False
)

pipeline = Pipeline(
    workspace=ws,
    steps=[pipeline_step]
)

experiment = Experiment(workspace=ws, name='automl')

pipeline_run = experiment.submit(pipeline)

# # Kill ngrok on run end
# kill_status = ["Finished", "Failed"]
# while pipeline_run.get_detailed_status()['status'] not in kill_status:
#     try:
#         time.sleep(10)
#     except KeyboardInterrupt:
#         pipeline_run.cancel()
#
# ngrok_utils.kill_all()