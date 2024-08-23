from config import *
from preprocessing import DEFAULT_TOKENS
import numpy as np
import pandas as pd
import librosa
from util import *

from tqdm import tqdm

# import scipy.io.wavfile
from scipy.signal import resample_poly
from maad import sound, util
import pickle
import h5py

import soundfile as sf

from datetime import datetime
now = datetime.now().strftime('%Y-%m-%d_%H:%M:%S')
log = logging.getLogger(__name__)


# hdf5 -> 1s to get all S chunks for 1 recording (<40mins for an epoch)
# maad.signal.wavfile.read + others -> 4.5s for all S chunks of 1 rec

# TODO move to configurable file
SR = 22_000
OVERLAP = 512
WIN_LEN = 1024
WINDOW_TYPE = 'hann'
SITE = 1
TRAIN_DEPLOYMENTS = 7
TRAIN_FILE = INTERMEDIATE / 'train.hdf5'
TEST_FILE = INTERMEDIATE / 'test.hdf5'
REC_CAP = -1 # debuggin only
DT = np.float16

from shutil import copy

def to_frames(seconds):
    return librosa.time_to_frames(
            seconds,
            sr=SR,
            hop_length=OVERLAP,
            n_fft=WIN_LEN,
    )
    
def process_S(rec_path: Path):
    rec = rec_path.name
    log.info(f"-- PROCESSING [{rec}] --")
    
    log.info("reading file")
    # given_sr, y = scipy.io.wavfile.read(rec_path)
    y, given_sr = sf.read(rec_path)
    # y, sr = librosa.load(rec_path, sr=SR) NOTE still need to try
    
    log.info("resampling")
    y = resample_poly(y, SR, given_sr) 
    if len(y.shape) > 1:
        y = np.mean(y, axis=1)

    
    log.info("stft")
    S, _, _, _ = sound.spectrogram(
        y, SR, WINDOW_TYPE, WIN_LEN, OVERLAP
    )

    log.info("normalizing")
    min_val = np.min(S)
    max_val = np.max(S)
    S_norm = (S - min_val) / (max_val - min_val)
    
    # log.info("power -> db")
    #S = util.power2dB(Sxx, 80)
    
    return S_norm



# --- script ---

annotation_df = pd.read_csv(ANNOTATIONS / 'correlated_annotations_7depl.csv')
ds = WavDataset()

log.info("retreiving recordings")

# all recording paths
all_recordings = ds.get_wav_paths()

# !! fix
with open('objects/correlated_recordings_7deployments.pkl', 'rb') as f:
    train_recordings = pickle.load(f)
    train_recordings = [ds[rec] for rec in train_recordings] # conv to paths

test_recordings = [r for r in all_recordings if r not in train_recordings]

# recordings containing/not-containing annotations
annotated = [ds[rec] for rec, _ in annotation_df.groupby('recording')]
not_annotated_train = [r for r in train_recordings if r not in annotated]

log.info(f'n# recordings: {len(all_recordings)}')
log.info(f'n# train recordings: {len(train_recordings)}')
log.info(f'n# test recordings: {len(test_recordings)}')
log.info(f'n# annotated training recordings: {len(annotated)}, eg -> {annotated[-1]}')
log.info(f'n# not annotated training recordings: {len(not_annotated_train)}, eg -> {not_annotated_train[-1]}')


for dataset_file in TRAIN_FILE, TEST_FILE:
    destination = dataset_file.with_name(now + '_' + str(dataset_file.name))
    log.info(f"backing up dataset in {destination}")
    copy(dataset_file, destination)
    assert destination.exists()

if not input("overwite existing dataset? ").lower() in ['y', 'yes']:
    exit()

train_f = h5py.File(TRAIN_FILE, 'w')
test_f = h5py.File(TEST_FILE, 'w')
    
train_f.close()
test_f.close()

train_f = h5py.File(TRAIN_FILE, 'w')
test_f = h5py.File(TEST_FILE, 'w')

# process annotated recordings seperately
log.info("== writing annotated train set ==")
for rec, rec_df in tqdm(annotation_df[:REC_CAP].groupby('recording'), desc=' writing annotated train set'):
    rec_path = ds[rec]

    S = process_S(rec_path)

    n_frames = to_frames(get_wav_length(rec_path))
    Y = np.zeros(shape=(4, n_frames), dtype=bool)
    
    assert S.shape[1] == Y.shape[1], f'mismatch shapes: {S.shape} -> {Y.shape}'

    log.info("computing Y")
    for label, label_index in DEFAULT_TOKENS.items():
        
        labelwize_annotations = rec_df[rec_df["label"] == label]
        start_times, end_times = (
            np.array(labelwize_annotations["min_t"].astype(np.int32)), 
            np.array(labelwize_annotations["max_t"].astype(np.int32))
        )
        
        label_start_frames = librosa.time_to_frames(
            start_times,
            sr=SR,
            hop_length=OVERLAP,
            n_fft=WIN_LEN,
        )
        
        label_end_frames = librosa.time_to_frames(
            end_times,
            sr=SR,
            hop_length=OVERLAP,
            n_fft=WIN_LEN,
        )

        for start, end in zip(label_start_frames, label_end_frames):
            Y[label_index, start:end] = 1

    log.info(f"{S.shape[1]} {Y.shape[1]}")

    # add to train dataset
    group = train_f.create_group(rec)
    group.create_dataset("X", data=S, dtype=DT) # NOTE could be byte
    group.create_dataset("Y", data=Y, dtype=bool)


# process un annotated training set
log.info("== writing (non annotated) train set ==")
for rec_path in tqdm(not_annotated_train[:REC_CAP], desc="writing (non annotated) train set"):
    rec = rec_path.name
    S = process_S(rec_path)
    Y = np.zeros(shape=(4, n_frames), dtype=bool)
    
    group = train_f.create_group(rec)
    group.create_dataset("X", data=S, dtype=DT)
    group.create_dataset("Y", data=Y, dtype=bool)
    
    
# process test set
log.info("== writing test set ==")
for rec_path in tqdm(test_recordings[:REC_CAP], desc="writing test set"):
    rec = rec_path.name
    S = process_S(rec_path)
    
    group = test_f.create_group(rec)
    group.create_dataset("X", data=S, dtype=DT)
    

X = np.array(train_f[list(train_f)[0]]['X'])

log.debug(f"list(train_ds): {list(train_f)}")
log.debug(f"X[0] -> {X}")
log.debug(f"X mean -> {X.mean()}")
log.debug(f"X max -> {X.max()}")

train_f.close()
test_f.close()




