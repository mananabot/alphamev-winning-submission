import ast

from typing import Any, List, Union, Tuple

import pandas as pd
import numpy as np

from gensim.models import Word2Vec


def featurize_df(
    df: pd.DataFrame,
    with_labels: bool,
    feature_names: List[str] = None,
    word_to_vec_vector_size: int = 50,
    word_to_vec_model_filepath: str = None,
    return_word_to_vec_model: bool = False,
    include_tx_hash: bool = False
) -> Union[
        pd.DataFrame,
        Tuple[pd.DataFrame, Word2Vec]
     ]:

    features = []
    event_chains = []
    numeric_lengths = []

    for row in df.iterrows():

        row = row[1]

        try:
            events = ast.literal_eval(row.events)
        except ValueError:
            events = row.events
        try:
            call = ast.literal_eval(row.call)
        except ValueError:
            call = row.call
        try:
            transfers = ast.literal_eval(row.transfers)
        except ValueError:
            transfers = row.transfers
        try:
            balances = ast.literal_eval(row.balances)
        except ValueError:
            balances = row.balances
        try:
            metadata = ast.literal_eval(row.metadata)
        except ValueError:
            metadata = row.metadata

        # There was an exception during decoding which has returned empty everything
        if (not events) and (not call) and (not transfers) and (not balances) and (not metadata):
            event_chains.append([])
            features.append({})
            continue

        feature = {}
        numeric = {}
        event_chain = []

        # Events
        for e in events:

            # Event chain
            try:
                contract_name = e['contract_name'].lower()
            except AttributeError:
                contract_name = 'unknown'

            try:
                event_name = e['event_name'].lower()
            except AttributeError:
                event_name = 'unknown'

            contract_name = hash_to_word(contract_name) \
                if is_hash(contract_name) \
                else contract_name

            event_name = hash_to_word(event_name) \
                if is_hash(event_name) \
                else event_name

            res = contract_name + ' ' + event_name
            event_chain.append(res)

            event_chain = [' '.join(event_chain)]

            # Event numeric params
            params = e['parameters']

            numeric_param = {
                p['name']: p['value']
                for p in params
                if isinstance(p['value'], (int, float))
            }

            numeric_lengths.append(len(numeric_param))

            numeric_param_adj = numeric_param.copy()

            if any([k in numeric.keys() for k in numeric_param]):

                for k in numeric_param:

                    if k == '':
                        k = 'unknown'

                    if k in numeric.keys():

                        k_ = [kk for kk in numeric.keys() if k in kk]

                        k_increment = [
                            kk for kk in k_
                            if kk.split("_")[-1].isnumeric()
                        ]

                        # First duplicate
                        if not k_increment:
                            k_increment = 1
                        # Consecutive duplicates
                        else:
                            k_increment = max([
                                int(kk[-1]) for kk in k_
                                if kk[-1].isnumeric()
                            ])

                        k_new = f"{k}_{k_increment}"

                        numeric_param_adj[k_new] = numeric_param_adj.pop(k)

            numeric.update(numeric_param_adj)

        # Numeric
        for k in numeric:
            feature[k] = float(numeric[k])

        # Lengths
        feature['event_len'] = len(events)
        # feature['call_depth'] = "TODO"  # recursion needed?

        # Metadata
        feature['gas_price'] = float(metadata['gas_price'])
        feature['gas_used'] = float(metadata['gas_used'])
        feature['gas_limit'] = float(metadata['gas_limit'])
        feature['tx_value'] = float(metadata['tx_value'])

        # Add tx_hash for later check
        if include_tx_hash:
            feature['tx_hash'] = metadata['tx_hash']

        # Transfers
        # TODO

        # Labels
        if with_labels:
            feature['label_0'] = row.label_0
            feature['label_1'] = row.label_1

        if not event_chain:
            event_chain = ['nothing']

        event_chains.append(event_chain)
        features.append(feature)

    if word_to_vec_model_filepath is not None:
        word_to_vec_model = Word2Vec.load(word_to_vec_model_filepath)
    else:
        word_to_vec_model = Word2Vec(
            sentences=event_chains,
            vector_size=word_to_vec_vector_size,
            min_count=0
        )

    event_chain_vectors_list = []
    for e in event_chains:
        try:
            v = word_to_vec_model.wv[e[0]]
        except:
            v = word_to_vec_model.wv['nothing']
        event_chain_vectors_list.append(v)
    event_chain_vectors = np.array(event_chain_vectors_list)

    event_chain_colnames = [f"event_chain_{i}" for i in range(word_to_vec_vector_size)]

    df = pd.DataFrame(features).fillna(0.0)
    df[event_chain_colnames] = event_chain_vectors

    if feature_names is not None:
        # Remove columns with unknown feature names
        for col in df.columns:
            if col not in feature_names:
                if col != 'tx_hash':
                    df.drop(columns=col, inplace=True)
                else:
                    if not include_tx_hash:
                        df.drop(columns=col, inplace=True)

        # Paste known features
        df_all_features = pd.DataFrame(columns=feature_names)
        df_final = pd.concat([df_all_features, df]).fillna(0.0)
    else:
        df_final = df

    # Make sure there are no blank colnames
    if '' in df_final.columns:
        blank_idx = df_final.columns.tolist().index('')
        df_final.rename(
            columns={df_final.columns[blank_idx]: "unknown"},
            inplace=True
        )

    if return_word_to_vec_model:
        return df_final, word_to_vec_model
    return df_final


def is_hash(value: Any) -> bool:
    if isinstance(value, str):
        if len(value) > 2:
            if value[:2] == '0x':
                return True
    return False


def hash_to_word(hash: str) -> str:
    return f"unknown{hash[-4:]}"


def write_feature_names(feature_names: List[str]) -> None:
    with open("feature_names.txt", "w") as f:
        for item in feature_names:
            f.write("%s\n" % item)


def read_feature_names(filepath: str) -> List[str]:
    with open(filepath) as f:
        feature_names = f.readlines()
        feature_names = [line.rstrip() for line in feature_names]
    return feature_names


def get_tx_hash(prepare_df: pd.DataFrame) -> List[int]:
    prepare_df_reset = prepare_df.reset_index(drop=True)
    tx_hash_prepare = []
    unknown_idx = 0
    for i, m in enumerate(prepare_df_reset.metadata):
        try:
            m = ast.literal_eval(m)
        except ValueError:
            pass
        try:
            m_tx_hash = m['tx_hash']
        except:
            m_tx_hash = f'unknown_{unknown_idx}'
            unknown_idx += 1
        tx_hash_prepare.append(m_tx_hash)
    return tx_hash_prepare


def sort_that_shit(prepare_df, tx_hash_prepare, tx_hash_test):
    prepare_df.reset_index(drop=True, inplace=True)
    prepare_dict = prepare_df.to_dict('records')
    empty_row = {
        'events': [],
        'call': {},
        'transfers': [],
        'balances': [],
        'metadata': {}
    }
    sorted_prepare_dicts = []
    n_fails = 0
    for hash_test in tx_hash_test:
        try:
            hash_prepare_idx = tx_hash_prepare.index(hash_test)
            sorted_prepare_dicts.append(prepare_dict[hash_prepare_idx])
        except ValueError:
            sorted_prepare_dicts.append(empty_row)
            n_fails += 1
    return pd.DataFrame(sorted_prepare_dicts), n_fails
