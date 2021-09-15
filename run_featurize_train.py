import os
import time

from dotenv import load_dotenv

from mev.azure.run import get_auth_ws, run_train, run_featurize

load_dotenv()

ENVIRONMENT_VARIABLES = dict(
    TENANT_ID=os.getenv("TENANT_ID"),
)


if __name__ == "__main__":

    # Params
    compute_target_name_1 = "mev-compute"
    compute_target_name_2 = "mev-compute2"

    source_dir_featurize = "./mev/azure/src"
    script_name_featurize = "featurize.py"

    source_dir_train = "./mev/azure/src_train"
    script_name_train = "train.py"

    output_name_featurize = 'featurize_all'
    output_name_train = 'train_all'

    dataset_name_featurize = "prepare_all"
    dataset_name_train = output_name_featurize

    kind = 'both'
    time_limit = 10 * 3600

    # Auth to Azure ML
    ws = get_auth_ws(ENVIRONMENT_VARIABLES["TENANT_ID"])

    print("Running 'featurize' step...")
    run_featurize(
        dataset_name=dataset_name_featurize,
        compute_target_name=compute_target_name_1,
        source_dir=source_dir_featurize,
        script_name=script_name_featurize,
        ws=ws,
        environment_variables=ENVIRONMENT_VARIABLES,
        with_labels=True,
        output_name=output_name_featurize
    )
    print("Featurize step finished.")

    print("Sleeping 4 mins to wait for compute being ready for next pipeline.")
    time.sleep(4 * 60)

    print("Running 'train' step...")
    run_train(
        dataset_name=dataset_name_train,
        compute_target_names=[compute_target_name_1, compute_target_name_2],
        source_dir=source_dir_train,
        script_name=script_name_train,
        ws=ws,
        environment_variables=ENVIRONMENT_VARIABLES,
        time_limit=time_limit,
        output_name=output_name_train,
        kind=kind
    )
    print("Train step finished.")
    print("Done.")