from config import *
from util import WavDataset
from pathlib import Path
from preprocessing import SpectrogramSequence

import pandas as pd
import pickle
import librosa
import matplotlib.pyplot as plt
import numpy as np

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


export_path = ANNOTATIONS / 'svisualizer_annotations'

import subprocess

ds = WavDataset()
for recording, group_df in to_annotate.groupby('recording'):
    command = f'svisual {ds[recording]}'
    process = subprocess.Popen(command, shell=True, stdout=subprocess.PIPE) # svisual not found
    process.wait()
    print(process.returncode)







    
    


