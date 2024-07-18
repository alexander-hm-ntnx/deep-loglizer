"""
Preprocess logpai/loghub structured dataset into processed test and train files
"""

import os
import re
import pickle
import argparse
import pandas as pd
import numpy as np
from utils import decision, json_pretty_dump
from collections import OrderedDict, defaultdict

seed = 42
np.random.seed(seed)

parser = argparse.ArgumentParser()
parser.add_argument("--log_file", type=str)
parser.add_argument("--data_dir", type=str)
parser.add_argument("--data_name", type=str)
parser.add_argument("--train_anomaly_ratio", default=1.0, type=float)
parser.add_argument("--test_ratio", default=0.2, type=float)
# TODO: Add a time field delimeter
params = vars(parser.parse_args())
params["random_sessions"] = True  # shuffle sessions

data_dir = os.path.join(params["data_dir"], params["data_name"])
os.makedirs(data_dir, exist_ok=True)

def preprocess(
    log_file,
    test_ratio=None,
    train_anomaly_ratio=1,
    random_sessions=False,
    **kwargs
):
    """Process structured log file to session_test and session_train files. 
    
    Arguments
        log_file: path to structured log file

        data_dir: path to output directory for processed data

        data_name: name for output file for processed data

        train_anomaly_ratio:

    Returns

    
    """
    print(f"Loading {os.path.basename(log_file)} logs.")
    struct_log = pd.read_csv(log_file, engine="c", na_filter=False, memory_map=True)

    struct_log["time"] = pd.to_datetime(
        struct_log["Time"], format="mixed"
    )
    struct_log["seconds_since"] = (
        (struct_log["time"] - struct_log["time"][0]).dt.total_seconds().astype(int)
    )

    session_dict = OrderedDict()
    column_idx = {col: idx for idx, col in enumerate(struct_log.columns)}
    for idx, row in enumerate(struct_log.values):
        current = row[column_idx["seconds_since"]]
        # if idx == 0:
        #     sessid = current
        # elif current - sessid > time_range:
        #     sessid = current
        sessid = current
        if sessid not in session_dict:
            session_dict[sessid] = defaultdict(list)
        session_dict[sessid]["templates"].append(row[column_idx["EventTemplate"]])
        # session_dict[sessid]["label"].append(
        #     row[column_idx["Label"]]
        # )  # labeling for each log

    # labeling for each session
    # for k, v in session_dict.items():
    #     session_dict[k]["label"] = [int(1 in v["label"])]

    session_idx = list(range(len(session_dict)))
    # split data
    if random_sessions:
        print("Using random partition.")
        np.random.shuffle(session_idx)

    session_ids = np.array(list(session_dict.keys()))

    train_lines = int(( - test_ratio) * len(session_idx))
    test_lines = int(test_ratio * len(session_idx))

    session_idx_train = session_idx[0:train_lines]
    session_idx_test = session_idx[-test_lines:]

    session_id_train = session_ids[session_idx_train]
    session_id_test = session_ids[session_idx_test]

    print("Total # sessions: {}".format(len(session_ids)))

    session_train = {
        k: session_dict[k]
        for k in session_id_train
    }

    session_test = {k: session_dict[k] for k in session_id_test}


    print("# train sessions: {}".format(len(session_train)))
    print("# test sessions: {}".format(len(session_test)))

    print(session_train)

    with open(os.path.join(data_dir, "session_train.pkl"), "wb") as fw:
        pickle.dump(session_train, fw)
    with open(os.path.join(data_dir, "session_test.pkl"), "wb") as fw:
        pickle.dump(session_test, fw)
    json_pretty_dump(params, os.path.join(data_dir, "data_desc.json"))

    print("Saved to {}".format(data_dir))
    return session_train, session_test


if __name__ == "__main__":
    preprocess(**params)
