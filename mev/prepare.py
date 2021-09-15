import sys
import os
import ast
import csv
import shutil

from typing import List, Optional, Dict, Union

import pandas as pd
import numpy as np

from ethtx import EthTx, EthTxConfig
from ethtx.models.objects_model import Call

csv.field_size_limit(sys.maxsize)


def divide_dataset(filepath: str, n_subsets: int) -> None:

    # Determine chunksize
    with open(filepath) as csvfile:
        csv_reader = csv.reader(csvfile)
        n_rows = sum(1 for row in csv_reader)
        chunksize = int(np.ceil(n_rows / n_subsets))

    # Make sure a fresh and empty dir is created
    filepath_dir = os.path.abspath(os.path.dirname(filepath))
    chunks_dir = os.path.join(filepath_dir, "chunks")
    if os.path.exists(chunks_dir):
        shutil.rmtree(chunks_dir)
    os.makedirs(chunks_dir)

    # Save chunks
    i = 0
    for chunk in pd.read_csv(filepath, chunksize=chunksize, iterator=True):
        chunk.to_csv(f"{filepath_dir}/chunks/chunk_{i}.csv", index=False)
        print(f"Chunk number {i} written.")
        i += 1
    print("Dataset dividing finished.")


class Decoder:

    def __init__(
        self,
        mongo_connection_string: str,
        mongo_database_name: str,
        etherscan_api_key: str,
        sleep_time: float,
        node_url: Union[str, List[str]],
        chain: Optional[str] = 'mainnet',
        etherscan_urls:
            Optional[Dict[str, str]] = {
                "mainnet": "https://api.etherscan.io/api"
            },
    ):
        self.mongo_connection_string = mongo_connection_string
        self.mongo_database_name = mongo_database_name
        self.etherscan_api_key = etherscan_api_key

        ethtx_config = EthTxConfig(
            mongo_connection_string=mongo_connection_string,
            mongo_database=mongo_database_name,
            etherscan_api_key=etherscan_api_key,
            web3nodes={
                chain: node_url if isinstance(node_url, list) else [node_url],
            },
            default_chain=chain,
            etherscan_urls=etherscan_urls,
            sleep_time=sleep_time
        )

        self.ethtx = EthTx.initialize(ethtx_config)
        self.web3provider = self.ethtx.providers.web3provider

    def prepare_call_tree(self, tx_trace: str) -> List[Call]:

        def apply(trace_dict: dict) -> Call:
            """
            Apply recursively on trace_dict
            """

            # Some tx have these fields missing
            has_subcall = 'calls' in trace_dict.keys()
            has_output = 'output' in trace_dict.keys()
            has_gasused = 'gasUsed' in trace_dict.keys()

            is_selfdestruct = trace_dict['type'] == "SELFDESTRUCT"

            if not is_selfdestruct:
                call = Call(
                    call_type=trace_dict['type'].lower(),
                    call_gas=int(trace_dict['gas'], 16),
                    from_address=trace_dict['from'],
                    to_address=trace_dict['to'],
                    call_value=int(trace_dict['value'], 16),
                    call_data=trace_dict['input'],
                    return_value=trace_dict['output'] if has_output else "",
                    gas_used=int(trace_dict['gasUsed'], 16) if has_gasused else 0,
                    status=True,
                    error="",
                    subcalls=trace_dict['calls'] if has_subcall else []
                )
            else:
                call = Call(
                    call_type=trace_dict['type'].lower(),
                    call_gas=0,
                    from_address="",
                    to_address=trace_dict['to'],
                    call_value=0,
                    call_data="",
                    return_value="",
                    gas_used=0,
                    status=True,
                    error="",
                    subcalls=[]
                )

            for i, subcall in enumerate(call.subcalls):
                subcall = apply(subcall)
                call.subcalls[i] = subcall
            return call

        trace_dict = ast.literal_eval(tx_trace)

        return apply(trace_dict)

    def decode_tx(self, tx: dict, with_labels=False) -> dict:
        """
        Semantic decode tx
        """
        tx_hash = tx['txHash']

        # W3
        w3transaction = self.web3provider.get_transaction(tx_hash)
        w3block = self.web3provider.get_block(w3transaction.blockNumber)
        w3receipt = self.web3provider.get_receipt(w3transaction.hash.hex())

        # Little hacky
        transaction_metadata = w3transaction.to_object(w3receipt)
        block_metadata = w3block.to_object()
        transaction_events = [w3log.to_object() for w3log in w3receipt.logs]

        tx_trace = tx['txTrace']
        call_tree = self.prepare_call_tree(tx_trace)

        # ABI
        abi_decoded_events = self.ethtx.decoders.abi_decoder.decode_events(
            events=transaction_events,
            block=block_metadata,
            transaction=transaction_metadata
        )
        abi_decoded_call = self.ethtx.decoders.abi_decoder.decode_calls(
            root_call=call_tree,
            block=block_metadata,
            transaction=transaction_metadata
        )
        abi_decoded_transfers = self.ethtx.decoders.abi_decoder.decode_transfers(
            call=abi_decoded_call,
            events=abi_decoded_events
        )
        abi_decoded_balances = self.ethtx.decoders.abi_decoder.decode_balances(
            transfers=abi_decoded_transfers
        )

        proxies = self.ethtx.decoders.get_proxies(call_tree)

        # Semantic
        decoded_metadata = self.ethtx.decoders.semantic_decoder.decode_metadata(
            block_metadata=block_metadata,
            tx_metadata=transaction_metadata
        )
        decoded_events = self.ethtx.decoders.semantic_decoder.decode_events(
            events=abi_decoded_events,
            tx_metadata=decoded_metadata,
            token_proxies=proxies
        )
        decoded_call = self.ethtx.decoders.semantic_decoder.decode_calls(
            call=abi_decoded_call,
            tx_metadata=decoded_metadata,
            token_proxies=proxies
        )
        decoded_transfers = self.ethtx.decoders.semantic_decoder.decode_transfers(
            transfers=abi_decoded_transfers,
            tx_metadata=decoded_metadata
        )
        decoded_balances = self.ethtx.decoders.semantic_decoder.decode_balances(
            balances=abi_decoded_balances,
            tx_metadata=decoded_metadata
        )

        assert not isinstance(decoded_call, list)
        assert not isinstance(decoded_metadata, list)

        ret = {
            'events': [e.json() for e in decoded_events],
            'call': decoded_call.json(),
            'transfers': [t.json() for t in decoded_transfers],
            'balances': [b.json() for b in decoded_balances],
            'metadata': decoded_metadata.json()
        }

        if with_labels:
            ret['label_0'] = tx['Label0'] * 1.0
            ret['label_1'] = tx['Label1']

        return ret
