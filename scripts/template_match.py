import argparse
from pathlib import Path

import pickle
import pandas as pd
from untracked.correlating import Template
from config import CORRELATIONS

DESC = (
"""
correlate a template over the dataset. 
To use a template, add it to ./templates.csv including (name,low,high,start,end,site,deployment,recording)
run with: python template_match [template_name/s]. 
"""
)

parser = argparse.ArgumentParser(description=DESC)
parser.add_argument("-t", "--threshhold", default=7)
parser.add_argument("-d", "--deployments", default=0.5)

args, unknown = parser.parse_known_args()

if not unknown:
    parser.print_help()
    exit()

TEMPLATES = Path('templates.csv')
df = pd.read_csv(TEMPLATES)
thresh = float(args.threshhold) if args.threshhold else 0.5
depl = int(args.deployments) if args.deployments else 7

if not TEMPLATES.exists():
    raise RuntimeError(f'file {TEMPLATES} does not exist. run with -h to see usage')

templates = df["name"].unique()
if not any([u for u in unknown if u in templates]):
    raise RuntimeError(f"none of {unknown} in {templates}")

for i, row in df[df['name'].isin(unknown)].iterrows():
    low, high, start, end = [float(i) for i in (
        row['low'], row['high'], row['start'], row['end']
    )]
    source = Path(row['source'])
    name = row['name']

    print(f"\n-- correlating {name} --")

    templ = Template(
        frequency_lims=(low, high),
        time_segment=(start, end),
        source_recording_path=source,
        name=name,
    )
    
    directory = Path(CORRELATIONS) / name
    directory.mkdir(exist_ok=True)
    f = directory / "all_7depl.pkl"
    
    with open(f, "wb") as f:
        corrs = templ.template_match(
            file_cap=None,
            n_deployments=depl,
            thresh=thresh,
            save_incremental=True
        )
        pickle.dump(corrs, f)

    print(corrs)
    print(f"correlations stored in {f}")
    
