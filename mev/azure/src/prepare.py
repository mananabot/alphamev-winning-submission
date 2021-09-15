import os
import argparse
import shutil
import tempfile
import time
import pydevd_pycharm

import pandas as pd

from azureml.core import Run

from mev.prepare import Decoder


parser = argparse.ArgumentParser()
parser.add_argument("--input_name", required=True)
parser.add_argument("--output", required=True)
parser.add_argument("--id", required=True)
parser.add_argument(
    '--with_labels',
    default=False,
    type=lambda x: (str(x).lower() == 'true'),
    required=False)
parser.add_argument("--nrows", required=True)
args = parser.parse_args()

# Attach PyCharm debugger
# print(f"Starting debugger at host == {os.environ.get('PYCHARM_DEBUG_HOST')} and port == {os.environ.get('PYCHARM_DEBUG_PORT')}")
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

mongo_connection_string = os.getenv("MONGO_CONNECTION_STRING")
mongo_database_name = os.getenv("MONGO_DATABASE_NAME")
etherscan_api_key = os.getenv("ETHERSCAN_API_KEY")
moralis_node = os.getenv("MORALIS_NODE")
alchemy_node = os.getenv("ALCHEMY_NODE")
sleep_time = float(os.getenv("SLEEP_TIME"))


decoder = Decoder(
    mongo_connection_string,
    mongo_database_name,
    etherscan_api_key,
    sleep_time,
    node_url=[moralis_node, alchemy_node]
)

filepath = sorted(input_dataset_filepaths)[int(args.id)]

if (args.nrows is not None) and (args.nrows != 'None'):
    example_df = pd.read_csv(filepath, nrows=int(args.nrows))
else:
    example_df = pd.read_csv(filepath)

transactions = example_df.to_dict('records')

try:
    res = []
    for tx in transactions:
        t1 = time.time()
        try:
            tx_decoded = decoder.decode_tx(tx, with_labels=args.with_labels)
        except Exception as e:
            print("Exception during decoding:")
            print(str(e))
            print("Appending empty transaction and continuing...")
            tx_decoded = {
                'events': [],
                'call': {},
                'transfers': [],
                'balances': [],
                'metadata': {},
            }

            if args.with_labels:
                tx_decoded['label_0'] = 0.0
                tx_decoded['label_1'] = 0.0

        res.append(tx_decoded)

        print(f"Decoding time: {time.time() - t1}")
except Exception as e:
    print("Exception:")
    print(str(e))
    print("Removing tree")
    shutil.rmtree(temporary_path)

try:
    os.makedirs(args.output, exist_ok=True)
    res_df = pd.DataFrame(res)
    filepath = f"{args.output}/decoded_{args.id}.csv"
    res_df.to_csv(filepath, index=False)
    print(f"Saved in: {filepath}")
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
