import os
import argparse

from dotenv import load_dotenv

from mev.azure.run import get_auth_ws, run_prepare

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
    parser.add_argument(
        "--nodes", type=int, help="Number of cluster nodes", required=True)
    args = parser.parse_args()

    assert args.nodes <= 100, "Max nodes is 100"

    # All params
    dataset_name_prepare = "mev_train_dataset"
    compute_target_name = "mev-cluster"
    output_name = "prepare_new_second_part"

    source_dir_prepare = "./mev/azure/src"
    script_name_prepare = "prepare.py"
    max_nodes = args.nodes

    with_labels = True
    id_from = 50

    # Auth to Azure ML
    ws = get_auth_ws(ENVIRONMENT_VARIABLES["TENANT_ID"])

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
        output_name=output_name,
        id_from=id_from
    )
    print("Prepare step finished.")