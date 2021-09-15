import os
import argparse
import shutil
import tempfile
import pickle
import pydevd_pycharm

import pandas as pd
import numpy as np

from azureml.core import Run

from supervised.automl import AutoML
from sklearn.preprocessing import QuantileTransformer

parser = argparse.ArgumentParser()
parser.add_argument("--input_name", required=True)
parser.add_argument("--kind", required=True)
parser.add_argument("--time_limit", required=True)
parser.add_argument("--output", required=True)
args = parser.parse_args()

assert args.kind in ["binary", "regression"]

# # Attach PyCharm debugger
# print("Starting debugger...")
# pydevd_pycharm.settrace(
#     os.environ.get("PYCHARM_DEBUG_HOST"),
#     port=int(os.environ.get("PYCHARM_DEBUG_PORT")),
#     stdoutToServer=False,
#     stderrToServer=False)
# print('PyCharm debugger attached to prepare.py.')
#
current_directory = os.getcwd()
temporary_prefix = f"{current_directory}/"
temporary_path = tempfile.mkdtemp(prefix=temporary_prefix)

input_dataset_filepaths = Run.get_context() \
    .input_datasets[args.input_name] \
    .download(target_path=temporary_path)

features_filepath = [
    f for f in input_dataset_filepaths
    if f.split(".")[-1] == "csv"
]

assert len(features_filepath) == 1

features_filepath = features_filepath[0]

features_df = pd.read_csv(features_filepath)

os.makedirs(args.output, exist_ok=True)

if args.kind == "binary":
    X = features_df.drop(columns=['label_0', 'label_1'])
    y = features_df.label_0.values
else:
    X = features_df[features_df.label_0 == 1].reset_index(drop=True)
    y = X.label_1.values
    X = X.drop(columns=['label_0', 'label_1'])

    # Log transform is shitty
    # y = np.log(y)

    # Use quantile transformer instead and profit
    transformer = QuantileTransformer(output_distribution='uniform')
    y = transformer.fit_transform(y.reshape(-1, 1)).flatten()

    # Save transformer right away
    transformer_output_filepath = os.path.join(args.output, "transformer.pkl")
    pickle.dump(
        transformer,
        open(transformer_output_filepath, "wb")
    )
    print(f"Transformer saved in {transformer_output_filepath}")

output_filepath = f"{args.output}/report_binary" if args.kind == "binary" \
    else f"{args.output}/report_regression"

if args.kind == "binary":
    automl = AutoML(
        mode="Compete",
        results_path=output_filepath,
        total_time_limit=int(args.time_limit),
        algorithms=['LightGBM', 'Xgboost', 'CatBoost']
    )
else:
    automl = AutoML(
        mode="Compete",
        results_path=output_filepath,
        total_time_limit=int(args.time_limit),
        algorithms=['LightGBM', 'Xgboost'],
        eval_metric='mse',
        features_selection=False,
        random_state=999
    )


automl.fit(X, y)

shutil.rmtree(temporary_path)
