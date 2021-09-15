import os
import argparse
import shutil
import tempfile
import csv
import pickle
import pydevd_pycharm

from typing import List

from supervised.automl import AutoML

import pandas as pd
import numpy as np

from azureml.core import Run


parser = argparse.ArgumentParser()
parser.add_argument("--input_name", required=True)
parser.add_argument("--model_directory", required=True)
parser.add_argument("--binary_model_directory", required=True)
parser.add_argument("--regression_model_directory", required=True)
parser.add_argument("--output", required=True)
args = parser.parse_args()

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

if not isinstance(input_dataset_filepaths, list):
    input_dataset_filepaths = [input_dataset_filepaths]

model_directory_filepaths = Run.get_context() \
    .input_datasets[args.model_directory] \
    .download(target_path=temporary_path)


def get_report_dir(
    model_directory_filepaths: List[str],
    model_specific_directory: str
) -> str:

    model_specific_filepath = [
        f for f in model_directory_filepaths
        if model_specific_directory in f
    ][0]

    model_directory_end_idx = \
        model_specific_filepath.find(model_specific_directory) \
        + len(model_specific_directory)

    return model_specific_filepath[:model_directory_end_idx]


binary_model_directory = get_report_dir(
    model_directory_filepaths, args.binary_model_directory)

regression_model_directory = get_report_dir(
    model_directory_filepaths, args.regression_model_directory)

print(f"binary_model_directory: {binary_model_directory}")
print(f"regression_model_directory: {regression_model_directory}")

assert isinstance(binary_model_directory, str)
assert isinstance(regression_model_directory, str)

os.makedirs(args.output, exist_ok=True)

# Load models
automl_binary = AutoML(results_path=binary_model_directory)
automl_regression = AutoML(results_path=regression_model_directory)

# Load transformer
transformer_filepath_list = [f for f in model_directory_filepaths if f.split(".")[-1] == "pkl"]
if transformer_filepath_list:
    assert len(transformer_filepath_list) == 1
    transformer_filepath = transformer_filepath_list[0]
    transformer = pickle.load(open(transformer_filepath, "rb"))

# Predict
res_binary = []
res_regression = []
res_ground_truth_binary = []
res_ground_truth_regression = []

# Dataset filepath is csv (there can be other stuff like w2v model)
input_dataset_filepaths = [
    f for f in input_dataset_filepaths
    if f.split(".")[-1] == "csv"
]

for filepath in input_dataset_filepaths:

    X = pd.read_csv(filepath)

    if 'label_0' in X.columns and 'label_1' in X.columns:

        # Ground truth
        # res_ground_truth_binary.append(X.label_0.values)
        # res_ground_truth_regression.append(X.label_1.values)

        X.drop(columns=['label_0', 'label_1'], inplace=True)

    predictions_binary = automl_binary.predict_proba(X)[:, 1]

    if transformer_filepath_list:
        predictions_regression = transformer.inverse_transform(
            automl_regression.predict(X).reshape(-1, 1)
        ).flatten()
    else:
        predictions_regression = np.exp(
            automl_regression.predict(X)
        ) * 1.7

    res_binary.append(predictions_binary)
    res_regression.append(predictions_regression)

# Flatten to 1 array per model
submission_array_binary = np.array(res_binary).flatten()
submission_array_regression = np.array(res_regression).flatten()

binary_threshold = 0.485
submission_array_regression[submission_array_binary < binary_threshold] = 0.0

# if 'label_0' in X_all.columns and 'label_1' in X_all.columns:
#     ground_truth_array_binary = np.array(res_ground_truth_binary).flatten()
#     ground_truth_array_regression = np.array(res_ground_truth_regression).flatten()

# Save submission to csv
submission_filepath = os.path.join(args.output, "submission.csv")
submission = csv.writer(open(submission_filepath, 'w', encoding='UTF8'))
for x, y in zip(submission_array_binary, submission_array_regression):
    submission.writerow([x, y])

print(f"Submission saved to '{submission_filepath}'")

# # Save ground truth as well
# if 'label_0' in X_all.columns and 'label_1' in X_all.columns:
#     try:
#         ground_truth_filepath = os.path.join(args.output, "ground_truth.csv")
#         df_ground_truth = pd.DataFrame({
#             'label_0': ground_truth_array_binary,
#             'label_1': ground_truth_array_regression
#         })
# 
#         df_ground_truth.to_csv(ground_truth_filepath, index=False)
#         print(f"Ground truth saved to '{ground_truth_filepath}'")
#     except:
#         pass

if os.path.exists(temporary_path):
    shutil.rmtree(temporary_path)
