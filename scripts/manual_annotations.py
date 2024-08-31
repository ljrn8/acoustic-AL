from config import *
from util import WavDataset
from pathlib import Path
from preprocessing import SpectrogramSequence

import pandas as pd
import pickle
import librosa
import matplotlib.pyplot as plt
import numpy as np
import subprocess


# --- script ---

with open(INTERMEDIATE / 'metadata' / 'initial_training_recordings.pkl', 'rb') as f:
    all_train_recordings = pickle.load(f)
    
incomplete_ann = pd.read_csv(INTERMEDIATE / 'metadata' / 'incomplete_manual_annotations.csv')

# :)
to_annotate = incomplete_ann[
        incomplete_ann['recording'].isin(all_train_recordings)
    ]


annotations_f = ANNOTATIONS / 'annotations.csv'

header = ['label', 'recording', 'min_t', 'max_t', 'min_f', 'max_f', 'datetime', 'recording_length']
with open(annotations_f, 'a') as f:
    pd.DataFrame([header]).to_csv(f, header=False, index=False)

buffer_path = Path("/mnt/e/manual_annotations") / 'buffer'
store_path =  Path("/mnt/e/manual_annotations") / 'raw'

import shutil, os

ds = WavDataset()
for i, (recording, group_df) in enumerate(to_annotate.groupby('recording')):
    # command = f'/mnt/c/Program\ Files/Sonic\ Visualiser/Sonic\ Visualiser.exe {ds[recording]}'
    # command = f'sonic-visualiser {ds[recording]}'
    
    log.info(f"\n\n------ {recording} {i}/{len(to_annotate.groupby('recording'))} -------")
    
    log.info(f"existing correlations -> {group_df.label.value_counts()}")
    
    # skip already annotated
    already_annotated = [r[:-4] for r in os.listdir(store_path)]
    if recording in already_annotated:
        continue
    
    # open in sonic visualiser
    command = f'cd {ds[recording].parent} && \
        /mnt/c/Program\ Files/Sonic\ Visualiser/Sonic\ Visualiser.exe {recording} >> /home/ethan/working/acoustic-AL/scripts/logs/svisual'
    
    # also in audacity to see better sr
    # also = f'cd {ds[recording].parent} && \
    #     /mnt/c/Program\ Files/Audacity/Audacity.exe {recording} 2>&1 >> /dev/null'
    
    process = subprocess.Popen(command, shell=True, stdout=subprocess.PIPE) # svisual not found
    process.wait()
    print(process.returncode)
    
    # collect anntoations file
    files = os.listdir(buffer_path)
    if len(files) > 1:
        log.warning("!!!! len files > 1")
    f = files[0]
    shutil.move(buffer_path / f, store_path / (recording + ".csv"))
    
    
    # label ref:
    # nr = '
    # triangle = ;
    # fast trlil = l
    # upsweep = ]






    
    


