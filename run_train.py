import os

from dotenv import load_dotenv

from mev.azure.run import get_auth_ws, run_train

load_dotenv()

ENVIRONMENT_VARIABLES = dict(
    TENANT_ID=os.getenv("TENANT_ID"),
)


if __name__ == "__main__":

    # Params
    dataset_name_train = "featurize_all"

    compute_target_name_1 = "mev-compute"
    compute_target_name_2 = "mev-compute2"

    source_dir_train = "./mev/azure/src_train"
    script_name_train = "train.py"

    output_name = 'train_all_regression_transformed'
    kind = 'regression'
    time_limit = int(3 * 3600)

    # Auth to Azure ML
    ws = get_auth_ws(ENVIRONMENT_VARIABLES["TENANT_ID"])

    print("Running 'train' step...")
    run_train(
        dataset_name=dataset_name_train,
        compute_target_names=[compute_target_name_1, compute_target_name_2],
        source_dir=source_dir_train,
        script_name=script_name_train,
        ws=ws,
        environment_variables=ENVIRONMENT_VARIABLES,
        time_limit=time_limit,
        output_name=output_name,
        kind=kind
    )
    print("Train step finished.")
    print("Done.")
