from config import *
from preprocessing import DEFAULT_TOKENS
import numpy as np
import pandas as pd
import librosa
from util import *

from tqdm import tqdm

import scipy.io.wavfile
from scipy.signal import resample_poly
from maad import sound, util

log = logging.getLogger(__name__)

# TODO move to configurable file
SR = 22_000
OVERLAP = 512
WIN_LEN = 2048
WINDOW_TYPE = 'hann'
N_DEPLOYMENTS = 11
SITE = 1
DS_FILE = INTERMEDIATE / 'dataset.hdf5'

annotation_df = pd.read_csv(ANNOTATIONS / 'initial_dataset_7depl_metadata.csv')
ds = Dataset()


# --- script ---

log.info("getting recordings")

# all recording paths
all_recordings = []
for d in range(1, N_DEPLOYMENTS):
    all_recordings += ds.get_recordings(d, SITE, path=True)

# recording paths containing annotations
annotated = np.array([
    ds.get_data_path(df['deployment'].iloc[0], df['site'].iloc[0]) / rec
    for rec, df in annotation_df.groupby('recording')
])

annotated = np.array(annotated)
not_annotated = [r for r in all_recordings if r not in annotated]

log.info(f'total n# recordings: {len(all_recordings)}')
log.info(f'n# annotated recordings: {len(annotated)}, eg -> {annotated[-1]}')
log.info(f'n# not annotated recordings: {len(not_annotated)}, eg -> {not_annotated[-1]}')

## !! debuging
DS_FILE = '../untracked/test.hdf5'
not_annotated = not_annotated[:5]

log.info("processing annotated recordings")

# process annotated recordings seperately
for rec, rec_df in tqdm(annotation_df[:3].groupby('recording'), desc='processing annotated recordings'):
    rec_path = ds.get_data_path(rec_df['deployment'].iloc[0], rec_df['site'].iloc[0]) / rec
    
    with timeit("load"):
        given_sr, y = scipy.io.wavfile.read(rec_path)
        
    with timeit("resample"):
        s = resample_poly(y, SR, given_sr)
        if len(s.shape) > 1:
            s = np.mean(s, axis=1)

    # with timeit("stft"):
    #     S = SFT.stft(y)
    
    with timeit("stft " + rec):
        S, _, _, _ = sound.spectrogram(
            s, SR, WINDOW_TYPE, WIN_LEN, OVERLAP
        )

    with timeit("converting to db " + rec):
        S_db = util.power2dB(S, 80)
        
    n_frames = librosa.samples_to_frames(len(y))

    Y_recording = np.zeros(shape=(n_frames, 4))

    with timeit("computing Y"):
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
                Y_recording[start:end, label_index] = 1

    log.debug(f"recordings shapes= {S.shape}, {Y_recording.shape}")
    # ((1024, 8592), (112500, 4))

# TODO paddings
