import argparse
import pickle
import sys
from pathlib import Path

import pandas as pd
import scr_path
from correlating import Template
from config import CORRELATIONS

parser = argparse.ArgumentParser()
parser.add_argument("-t", "--threshhold")
parser.add_argument("-d", "--deployments")
args, unknown = parser.parse_known_args()
thresh = float(args.threshhold) if args.threshhold else 0.5
depl = int(args.deployments) if args.deployments else 7


with open("templates.txt", "r") as f:
    f.readline()
    for line in f:
        if line.strip() == "":
            continue

        items = line.split()
        name, low, high, start, end, source = items
        if name in unknown:
            low, high, start, end = [float(i) for i in (low, high, start, end)]
            source = Path(source)

            templ = Template(
                frequency_lims=(low, high),
                time_segment=(start, end),
                source_recording_path=source,
                name=name,
            )

            directory = Path(CORRELATIONS) / name
            directory.mkdir(exist_ok=True)
            with open(directory / "all_7depl.pkl", "wb") as f:
                corrs = templ.template_match(
                    file_cap=None,
                    n_deployments=depl,
                    thresh=thresh,
                )
                pickle.dump(corrs, f)

            print(corrs)
