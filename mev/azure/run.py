import os
import time

from typing import Optional, Any, Dict, List

from azureml.core import (
    Workspace, Datastore, Dataset,
    ComputeTarget, Environment,
    Experiment
)
from azureml.pipeline.steps import PythonScriptStep
from azureml.pipeline.core import Pipeline
from azureml.data import OutputFileDatasetConfig
from azureml.core.runconfig import RunConfiguration
from azureml.core.authentication import InteractiveLoginAuthentication

from mev.azure import ngrok_utils


def get_auth_ws(tenant_id: str):
    return Workspace.from_config(
        auth=InteractiveLoginAuthentication(tenant_id)
    )


def upload_real_to_blob(real_filepath: str, dataset_name: str, ws: Workspace):
    datastore = ws.get_default_datastore()
    datastore.upload_files(
        files=[real_filepath],
        target_path=dataset_name
    )


def upload_chunks_to_blob(
    chunks_dir: str,
    dataset_name: str,
    ws: Workspace
) -> None:

    datastore = ws.get_default_datastore()

    Dataset.File.upload_directory(
        src_dir=chunks_dir,
        target=(datastore, dataset_name),
        overwrite=True
    )


def download_submission_to_local(
    ws: Workspace,
    dataset_path: str,
    output_directory: str
) -> None:
    datastore = Datastore.get_default(ws)
    Dataset.File.from_files(
        (datastore, dataset_path)
    ).download(target_path=output_directory, overwrite=True)


def print_status(pipeline_run, output_name) -> None:

    while True:
        is_completed = \
            pipeline_run.get_detailed_status()['status'] == "Finished"

        is_failed = \
            pipeline_run.get_detailed_status()['status'] == "Failed"

        print("Run status:")
        print(f"{output_name} run with ID == {pipeline_run.get_details()['runId']} and status == {pipeline_run.get_detailed_status()['status']}")
        print("=========")

        if is_failed:
            print(f"Run failed: {pipeline_run.get_detailed_status()['details']}")
            return

        if not is_completed:
            time.sleep(10)
        else:
            return


def print_statuses(pipeline_runs: list, output_name) -> None:
    failed_flag = False
    while True:
        is_completed = all([
            run.get_detailed_status()['status'] == "Finished"
            for run in pipeline_runs
        ])

        is_failed = [
            run.get_detailed_status()['status'] == "Failed"
            for run in pipeline_runs
        ]

        is_failed_any = any(is_failed)
        is_failed_all = all(is_failed)

        print("Run statuses:")
        for run in pipeline_runs:
            print(f"{output_name} run with ID == {run.get_details()['runId']} and status == {run.get_detailed_status()['status']}")
        print("=========")

        if is_failed_any and not failed_flag:
            print("At least one experiment failed, continuing...")
            failed_flag = True

        if is_failed_all:
            print("All experiments failed, returning.")
            return

        if not is_completed:
            time.sleep(10)
        else:
            print("All experiments completed successfully.")
            return


def init(
    dataset_name: str,
    compute_target_name: str,
    ws: Workspace,
    environment_variables: Optional[Dict[str, Any]] = None
) -> tuple:

    # Initialize conda environment from config file
    current_file_dir = os.path.dirname(os.path.realpath(__file__))

    env = Environment.from_conda_specification(
        name='env',
        file_path=os.path.join(current_file_dir, ".azureml", "env.yml")
    )

    # Add private pips
    private_pips = [
        os.path.join(
            current_file_dir,
            ".azureml",
            "mev-competition-0.1.9.tar.gz"
        ),
        os.path.join(
            current_file_dir,
            ".azureml",
            "EthTx-0.0.3.tar.gz"  # Hacky version
        )
    ]

    for private_pip in private_pips:
        whl_url = Environment.add_private_pip_wheel(
            ws,
            file_path=private_pip,
            exist_ok=True
        )
        env.python.conda_dependencies.add_pip_package(whl_url)

    # Fetch compute
    compute_target = ComputeTarget(workspace=ws, name=compute_target_name)

    # Add env variables
    if environment_variables is not None:
        env.environment_variables = environment_variables

    # Start ngrok tunnel
    # host, port = ngrok_utils.start_tunnel("8000")
    # print(f"Started ngrok at {host}:{port}")

    # Pass pycharm debug host and port as environment variables
    # env.environment_variables.update({
    #     'PYCHARM_DEBUG_PORT': port,
    #     'PYCHARM_DEBUG_HOST': host
    # })

    # Setup run config
    run_config = RunConfiguration()
    run_config.target = compute_target
    run_config.environment = env
    run_config.docker.use_docker = True

    # Load dataset from default datastore
    datastore = Datastore.get_default(ws)
    dataset_path = [(datastore, dataset_name)]
    dataset = Dataset.File.from_files(dataset_path).register(ws, dataset_name)

    return ws, compute_target, run_config, datastore, dataset


def run_prepare(
    dataset_name: str,
    compute_target_name: str,
    source_dir: str,
    script_name: str,
    ws: Workspace,
    environment_variables: Dict[str, Any],
    with_labels: bool,
    output_name: Optional[str] = 'prepare',
    max_nodes: Optional[int] = 100,
    id_from: Optional[int] = 0,
    nrows: Optional[int] = None
) -> None:

    ws, compute_target, run_config, datastore, dataset = init(
        dataset_name,
        compute_target_name,
        ws,
        environment_variables=environment_variables
    )

    # Define names/inputs/outputs
    dataset_input = dataset.as_named_input(dataset_name)

    output = OutputFileDatasetConfig(
        name=output_name,
        destination=(datastore, output_name)
    ).as_upload(overwrite=True).register_on_complete(output_name)

    # Submit pipeline runs to cluster
    id = id_from
    pipeline_runs = []

    n_files = len(dataset.to_path())
    for i in range(n_files):

        # Define pipeline
        pipeline_step = PythonScriptStep(
            script_name=script_name,
            source_directory=source_dir,
            arguments=[
                "--input_name", dataset_name,
                "--output", output,
                "--with_labels", with_labels,
                "--nrows", nrows,
                "--id", id
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

        experiment = Experiment(workspace=ws, name=f'{output_name}_{i}')

        pipeline_run = experiment.submit(pipeline)
        pipeline_runs.append(pipeline_run)

        id += 1

        if id == n_files:
            break

        if i == max_nodes-1:
            break

    print("All pipeline jobs submitted, waiting for them to finish...")

    #
    # # Cancel all runs on KeyboardInterrupt
    # while True:
    #     try:
    #         time.sleep(10)
    #     except KeyboardInterrupt:
    #         what_to_do = input("Cancel?").strip()
    #         if what_to_do in ['cancel', 'Cancel', 'yes', 'Yes', 'yeah']:
    #             for run in pipeline_runs:
    #                 run.run_cancel()
    #             sys.exit(1)
    #         else:
    #             print("Leaving...")
    #             sys.exit(1)

    # # Kill ngrok on run end
    # kill_status = ["Finished", "Failed"]
    # while pipeline_run.get_detailed_status()['status'] not in kill_status:
    #     try:
    #         time.sleep(10)
    #     except KeyboardInterrupt:
    #         pipeline_run.cancel()
    #
    # ngrok_utils.kill_all()

    return print_statuses(pipeline_runs, output_name)


def run_featurize(
    dataset_name: str,
    dataset_name_real: str,
    compute_target_name: str,
    source_dir: str,
    script_name: str,
    ws: Workspace,
    environment_variables: Dict[str, Any],
    with_labels: bool,
    dataset_name_model: Optional[str] = None,
    output_name: Optional[str] = 'featurize',
):

    ws, compute_target, run_config, datastore, dataset = init(
        dataset_name,
        compute_target_name,
        ws,
        environment_variables=environment_variables
    )

    # Define names/inputs/outputs
    dataset_input = dataset.as_named_input(dataset_name)

    if dataset_name_model is not None:
        dataset_model = Dataset.File.from_files(
            [(datastore, dataset_name_model)]
        ).register(ws, dataset_name_model)

        dataset_input_model = dataset_model.as_named_input(dataset_name_model)

        inputs = [dataset_input, dataset_input_model]
    else:
        inputs = [dataset_input]

    dataset_input_real = Dataset.File.from_files(
        (datastore, dataset_name_real)
    ).as_named_input(dataset_name_real)

    inputs.append(dataset_input_real)

    output = OutputFileDatasetConfig(
        name=output_name,
        destination=(datastore, output_name)
    ).as_upload(overwrite=True).register_on_complete(output_name)

    # Submit pipeline run
    pipeline_step = PythonScriptStep(
        script_name=script_name,
        source_directory=source_dir,
        arguments=[
            "--input_name", dataset_name,
            "--input_name_real", dataset_name_real,
            "--input_name_model", dataset_name_model,
            "--with_labels", with_labels,
            "--output", output,
        ],
        inputs=inputs,
        outputs=[output],
        compute_target=compute_target,
        runconfig=run_config,
        allow_reuse=False
    )

    pipeline = Pipeline(
        workspace=ws,
        steps=[pipeline_step]
    )

    experiment = Experiment(workspace=ws, name=f'{output_name}')

    pipeline_run = experiment.submit(pipeline)

    return print_status(pipeline_run, output_name)


def run_predict(
    dataset_name: str,
    compute_target_name: str,
    source_dir: str,
    script_name: str,
    ws: Workspace,
    environment_variables: Dict[str, Any],
    model_directory: str,
    binary_model_directory: str,
    regression_model_directory: str,
    output_name: Optional[str] = 'predict',
):
    ws, compute_target, run_config, datastore, dataset = init(
        dataset_name,
        compute_target_name,
        ws,
        environment_variables=environment_variables
    )

    # Define names/inputs/outputs
    dataset_input = dataset.as_named_input(dataset_name)

    model_input = Dataset.File.from_files(
        (datastore, model_directory)
    ).as_named_input(model_directory)

    output = OutputFileDatasetConfig(
        name=output_name,
        destination=(datastore, output_name)
    ).as_upload(overwrite=True).register_on_complete(output_name)

    # Submit pipeline run
    pipeline_step = PythonScriptStep(
        script_name=script_name,
        source_directory=source_dir,
        arguments=[
            "--input_name", dataset_name,
            "--model_directory", model_directory,
            "--binary_model_directory", binary_model_directory,
            "--regression_model_directory", regression_model_directory,
            "--output", output,
        ],
        inputs=[dataset_input, model_input],
        outputs=[output],
        compute_target=compute_target,
        runconfig=run_config,
        allow_reuse=False
    )

    pipeline = Pipeline(
        workspace=ws,
        steps=[pipeline_step]
    )

    experiment = Experiment(workspace=ws, name=f'{output_name}')

    pipeline_run = experiment.submit(pipeline)

    return print_status(pipeline_run, output_name)


def run_train(
    dataset_name: str,
    compute_target_names: List[str],
    source_dir: str,
    script_name: str,
    ws: Workspace,
    environment_variables: Dict[str, Any],
    time_limit: int,
    output_name: Optional[str] = 'train',
    kind: Optional[str] = 'both'
):
    """
    Runs binary clf and regression on two different computes.

    Different compute targets and run configs, the rest is the same.
    """

    ws, compute_target_1, run_config_1, datastore, dataset = init(
        dataset_name,
        compute_target_names[0],
        ws,
        environment_variables=environment_variables
    )

    ws, compute_target_2, run_config_2, datastore, dataset = init(
        dataset_name,
        compute_target_names[1],
        ws,
        environment_variables=environment_variables
    )

    # Define names/inputs/outputs
    dataset_input = dataset.as_named_input(dataset_name)

    output = OutputFileDatasetConfig(
        name=output_name,
        destination=(datastore, output_name)
    ).as_upload(overwrite=True).register_on_complete(output_name)

    assert len(compute_target_names) == 2
    assert kind in ['both', 'binary', 'regression']

    # Submit pipeline runs
    pipeline_runs = []
    for i in range(1, 3):

        if kind != 'both':
            model_kind = kind
            if model_kind == 'binary':
                comp_target = compute_target_1
                run_conf = run_config_1
            else:
                comp_target = compute_target_2
                run_conf = run_config_2
        else:
            model_kind = 'binary' if i == 1 else 'regression'
            comp_target = compute_target_1 if i == 1 else compute_target_2
            run_conf = run_config_1 if i == 1 else run_config_2

        pipeline_step = PythonScriptStep(
            script_name=script_name,
            source_directory=source_dir,
            arguments=[
                "--input_name", dataset_name,
                "--kind", model_kind,
                "--time_limit", time_limit,
                "--output", output,
            ],
            inputs=[dataset_input],
            outputs=[output],
            compute_target=comp_target,
            runconfig=run_conf,
            allow_reuse=False
        )

        pipeline = Pipeline(
            workspace=ws,
            steps=[pipeline_step]
        )

        experiment = Experiment(workspace=ws, name=f'{output_name}')

        pipeline_run = experiment.submit(pipeline)

        pipeline_runs.append(pipeline_run)

        if kind != "both":
            break

    return print_statuses(pipeline_runs, output_name)
