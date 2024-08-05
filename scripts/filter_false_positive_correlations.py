import pickle
import sys
from pathlib import Path

import scr_path
from settings import CORRELATIONS

name = sys.argv[1]

with open(Path("false_positives") / name, "r") as f:
    lines = list(filter(lambda x: x.strip() != "", f))
    indicies = []
    for i, n in enumerate(lines):
        n = n.strip()
        if n.startswith("!"):
            continue
        if n == "-":
            indicies += range(int(lines[i - 1]) + 1, int(lines[i + 1]))
        else:
            indicies.append(int(n))

print(indicies)
indicies = map(lambda x: x - 1, indicies)  # 0 base

with open(Path(CORRELATIONS) / name / "all_7depl_filtered.pkl", "rb") as f:
    df = pickle.load(f)
    df_verified = df.drop(indicies)

with open(Path(CORRELATIONS) / name / "all_7depl_verified.pkl", "wb") as f:
    pickle.dump(df_verified, f)
