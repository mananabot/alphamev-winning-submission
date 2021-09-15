import os
import argparse

from dotenv import load_dotenv

from mev.prepare import divide_dataset
from mev.azure.run import (
    get_auth_ws, run_prepare, run_featurize,
    run_predict, upload_chunks_to_blob,
    upload_real_to_blob,
    download_submission_to_local)

load_dotenv()

ENVIRONMENT_VARIABLES = dict(
    TENANT_ID=os.getenv("TENANT_ID"),
    MONGO_CONNECTION_STRING=os.getenv("MONGO_CONNECTION_STRING"),
    MONGO_DATABASE_NAME=os.getenv("MONGO_DATABASE_NAME"),
    ETHERSCAN_API_KEY=os.getenv("ETHERSCAN_API_KEY"),
    MORALIS_NODE=os.getenv("MORALIS_NODE"),
    ALCHEMY_NODE=os.getenv("ALCHEMY_NODE"),
    SLEEP_TIME=os.getenv("SLEEP_TIME"),
)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--input_filepath", type=str, required=True)
    parser.add_argument("--output_directory", type=str, required=True)
    parser.add_argument(
        "--nodes", type=int, help="Number of cluster nodes", required=True)
    args = parser.parse_args()

    assert args.nodes <= 25, "Max nodes is 25"

    # All params
    compute_target_name = "mev-cluster"

    source_dir_prepare = "./mev/azure/src"
    script_name_prepare = "prepare.py"
    max_nodes = args.nodes  # Used only for dataset decoding with EthTx (run_prepare)

    source_dir_featurize = "./mev/azure/src"
    script_name_featurize = "featurize.py"

    source_dir_predict = "./mev/azure/src"
    script_name_predict = "predict.py"

    model_directory = "train_all"
    binary_model_directory = "report_binary"
    regression_model_directory = "report_regression"

    output_name_prepare = "prepare_september_submission"
    output_name_featurize = "featurize_september_submission"
    output_name_predict = "predict_september_submission"

    dataset_name_prepare = "september_submission_dataset"
    dataset_name_featurize = output_name_prepare
    dataset_name_real = "real_september"
    dataset_name_featurize_model = "featurize_all"
    dataset_name_predict = output_name_featurize

    with_labels = False
    nrows = None

    # Auth to Azure ML
    ws = get_auth_ws(ENVIRONMENT_VARIABLES["TENANT_ID"])

    # Run locally
    upload_real_to_blob(
        real_filepath=args.input_filepath,
        dataset_name=dataset_name_real,
        ws=ws
    )
    divide_dataset(
        filepath=args.input_filepath,
        n_subsets=args.nodes
    )

    upload_chunks_to_blob(
        chunks_dir=os.path.join(os.path.dirname(args.input_filepath), "chunks"),
        dataset_name=dataset_name_prepare,
        ws=ws
    )

    # Run on Azure ML
    print("Running 'prepare' step...")
    run_prepare(
        dataset_name=dataset_name_prepare,
        compute_target_name=compute_target_name,
        source_dir=source_dir_prepare,
        script_name=script_name_prepare,
        ws=ws,
        environment_variables=ENVIRONMENT_VARIABLES,
        with_labels=with_labels,
        max_nodes=max_nodes,
        output_name=output_name_prepare,
        nrows=nrows
    )
    print("Prepare step finished.")

    print("Running 'featurize' step...")
    run_featurize(
        dataset_name=dataset_name_featurize,
        dataset_name_real=dataset_name_real,
        dataset_name_model=dataset_name_featurize_model,
        compute_target_name=compute_target_name,
        source_dir=source_dir_featurize,
        script_name=script_name_featurize,
        ws=ws,
        environment_variables=ENVIRONMENT_VARIABLES,
        with_labels=with_labels,
        output_name=output_name_featurize
    )
    print("Featurize step finished.")

    print("Running 'predict' step...")
    run_predict(
        dataset_name=dataset_name_predict,
        compute_target_name=compute_target_name,
        source_dir=source_dir_predict,
        script_name=script_name_predict,
        ws=ws,
        environment_variables=ENVIRONMENT_VARIABLES,
        model_directory=model_directory,
        binary_model_directory=binary_model_directory,
        regression_model_directory=regression_model_directory,
        output_name=output_name_predict
    )
    print("Predict step finished.")

    print("Downloading submission csv from Azure to local..")
    download_submission_to_local(
        ws=ws,
        dataset_path=output_name_predict,
        output_directory=args.output_directory
    )
    print("Done.")
