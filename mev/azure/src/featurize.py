import os
import argparse
import shutil
import tempfile
import pydevd_pycharm

import pandas as pd
import numpy as np

from azureml.core import Run

from mev.features import (
    featurize_df, read_feature_names,
    get_tx_hash, sort_that_shit)


parser = argparse.ArgumentParser()
parser.add_argument("--input_name", required=True)
parser.add_argument("--input_name_real", required=True)
parser.add_argument("--input_name_model", required=False)
parser.add_argument(
  '--with_labels',
  default=False,
  type=lambda x: (str(x).lower() == 'true'),
  required=False)
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

real_dataset_filepath = Run.get_context() \
    .input_datasets[args.input_name_real] \
    .download(target_path=temporary_path)

# Should be single file
print(f"real_dataset_filepath: {real_dataset_filepath}")
assert len(real_dataset_filepath) == 1
real_dataset_filepath = real_dataset_filepath[0]

os.makedirs(args.output, exist_ok=True)

print(f"args.input_name_model: {args.input_name_model}")

if (args.input_name_model is not None) and (args.input_name_model != 'None'):

    input_model_filepaths = Run.get_context() \
        .input_datasets[args.input_name_model] \
        .download(target_path=temporary_path)

    w2v_model_filepath = [
        f for f in input_model_filepaths if
        f.split(".")[-1] == "model"
    ][0]
    feature_names_filepath = [
        f for f in input_model_filepaths
        if f.split(".")[-1] == "txt"
    ][0]
    feature_names = read_feature_names(feature_names_filepath)
else:
    w2v_model_filepath = None
    feature_names = None

# This is in a wrong order completely, need to sort by real input
df_decoded_list = [pd.read_csv(filepath) for filepath in input_dataset_filepaths]
df_decoded = pd.concat(df_decoded_list)

# Sort based on real data
df_real = pd.read_csv(real_dataset_filepath)

tolerance = 0.01
tolerance_abs = int(df_real.shape[0] * tolerance)

tx_hash_real = df_real.txHash.tolist()
tx_hash_decoded = get_tx_hash(df_decoded)

df_decoded_sorted, n_fails = sort_that_shit(df_decoded, tx_hash_decoded, tx_hash_real)

assert n_fails < tolerance_abs

print(f"w2v_model_filepath: {w2v_model_filepath}")

df_featurized, word_to_vec_model = featurize_df(
    df=df_decoded_sorted,
    with_labels=args.with_labels,
    feature_names=feature_names,
    word_to_vec_model_filepath=w2v_model_filepath,
    return_word_to_vec_model=True,
    include_tx_hash=True
)

assert 'tx_hash' in df_featurized.columns

# Check the order
df_featurized_tx_hash = df_featurized.tx_hash.values
tx_hash_real = np.array(tx_hash_real)

assert tx_hash_real.shape[0] - np.sum(df_featurized_tx_hash == tx_hash_real) < tolerance_abs

# Drop the tx_hash feature
df_featurized.drop(columns=['tx_hash'], inplace=True)

assert 'tx_hash' not in df_featurized.columns

feature_names = df_featurized.columns.tolist()
feature_names_filepath = os.path.join(args.output, "feature_names.txt")
with open(feature_names_filepath, "w") as f:
    for item in feature_names:
        f.write("%s\n" % item)
print(f"Feature names written in: {feature_names_filepath}")

# Test equality
feature_names_saved = read_feature_names(feature_names_filepath)
was_some_unequal = False
for f1, f2 in zip(feature_names, feature_names_saved):
    if f1 != f2:
        print("Inequality:")
        print(f"f1 is: {f1}")
        print(f"f2 is: {f2}")
        was_some_unequal = True

if not was_some_unequal:
    print("All saved feature names are equal to dataframe's!")

try:
    filepath = f"{args.output}/featurized.csv"
    df_featurized.to_csv(filepath, index=False)
    print(f"Saved in: {filepath}")

    filepath_word_to_vec = f"{args.output}/word_to_vec.model"
    word_to_vec_model.save(filepath_word_to_vec)
    print(f"Saved Word2Vec model in: {filepath_word_to_vec}")
except Exception as e:
    print("Exception:")
    print(str(e))
    if os.path.exists(temporary_path):
        print("Exception:")
        print(str(e))
        print("Removing tree")
        shutil.rmtree(temporary_path)

if os.path.exists(temporary_path):
    shutil.rmtree(temporary_path)
