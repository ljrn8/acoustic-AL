import numpy as np
import pandas as pd
import librosa

from util import WavDataset
from config import *

from scipy.signal import resample_poly
import pickle
import h5py
import soundfile as sf
from tqdm import tqdm


SR = 16_000 # required for YAMnet / VGGish
TRAIN_FILE = INTERMEDIATE / 'train.hdf5'
DEFAULT_TOKENS = {  
    "fast_trill_6khz": 0,
    "nr_syllable_3khz": 1,
    "triangle_3khz": 2,
    "upsweep_500hz": 3,
}

annotations = pd.read_csv(ANNOTATIONS / 'manual_annotations' / 'initial_manual_annotations.csv')
annotated_recordings = annotations.recording.unique()

with open(ANNOTATIONS / 'manual_annotations' / 'initial_training_recordings.pkl', 'rb') as f:
    training_recordings = pickle.load(f)

print(f'n# annotated recs: ', len(annotated_recordings))
print(f'n# total training recs: ', len(training_recordings))
print(f'n# of annotations: ', len(annotations))

if input("continue? ").lower() in ['n', 'no']:
    exit()

train_f = h5py.File(TRAIN_FILE, 'w')

ds = WavDataset()
for rec in tqdm(training_recordings):

    # samples
    rec_path = ds[rec]
    s, given_sr = sf.read(rec_path)
    s = resample_poly(s, SR, given_sr) 
    if len(s.shape) > 1:
        s = np.mean(s, axis=1)
    
    # labels
    labelled_timesteps = np.zeros(shape=(4, len(s)), dtype=bool)    
    if rec in annotated_recordings:
        rec_df = annotations[annotations.recording == rec]
        for label, label_index in DEFAULT_TOKENS.items():
            labelwize_annotations = rec_df[rec_df.label == label]
            start_times, end_times = (
                np.array(labelwize_annotations["min_t"].astype(float)), 
                np.array(labelwize_annotations["max_t"].astype(float))
            )
            label_start_samples = librosa.time_to_samples(start_times, sr=SR)
            label_end_samples = librosa.time_to_samples(end_times,sr=SR)

            for start, end in zip(label_start_samples, label_end_samples):
                labelled_timesteps[label_index, start:end] = 1
    
    # store in dataset
    group = train_f.create_group(rec)
    group.create_dataset("X", data=s, dtype=np.float32)
    group.create_dataset("Y", data=labelled_timesteps, dtype=bool)