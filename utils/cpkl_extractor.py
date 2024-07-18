"""Extract .cpkl file to output file"""
import os
import pickle
import argparse
import pprint
from pathlib import Path

parser = argparse.ArgumentParser()

# Input Params
parser.add_argument("--input_path", type=str)
parser.add_argument("--output_path", default=None, type=str)

params = vars(parser.parse_args())
if params["output_path"] == None:
    params["output_path"] = Path(params["input_path"]).with_suffix('')

if __name__ == "__main__":
    with open(params["input_path"], "rb") as cpkl:
        data = pickle.load(cpkl)
        
        with open(params["output_path"], "w") as f:
            pprint.pprint(data, stream=f)