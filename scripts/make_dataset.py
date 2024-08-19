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
WIN_LEN = 1024
WINDOW_TYPE = 'hann'
SITE = 1
TRAIN_DEPLOYMENTS = 7
TRAIN_FILE = INTERMEDIATE / 'train.hdf5'
TEST_FILE = INTERMEDIATE / 'test.hdf5'


# --- script ---

def to_frames(seconds):
    return librosa.time_to_frames(
            seconds,
            sr=SR,
            hop_length=OVERLAP,
            n_fft=WIN_LEN,
    )

annotation_df = pd.read_csv(ANNOTATIONS / 'initial_dataset_7depl_metadata.csv')
ds = WavDataset()

log.info("retreiving recordings")

# all recording paths
all_recordings = ds.get_wav_files()


train_recordings = [r for r in all_recordings if r.containes
test_recordings = 


# recordings containing/not-containing annotations
annotated = [ds[rec] for rec, _ in annotation_df.groupby('recording')]
not_annotated = [r for r in all_recordings if r not in annotated]

log.info(f'total n# recordings: {len(all_recordings)}')
log.info(f'n# annotated recordings: {len(annotated)}, eg -> {annotated[-1]}')
log.info(f'n# not annotated recordings: {len(not_annotated)}, eg -> {not_annotated[-1]}')

log.info("processing annotated recordings")

# process annotated recordings seperately
for rec, rec_df in tqdm(annotation_df[:3].groupby('recording'), desc='processing annotated recordings'):
    rec_path = ds[rec]
    
    log.info(f"PROCESSING [{rec}]")
    
    log.info("reading file")
    given_sr, y = scipy.io.wavfile.read(rec_path)
    
    log.info("resampling")
    y = resample_poly(y, SR, given_sr) 
    if len(y.shape) > 1:
        y = np.mean(y, axis=1)

    # with timeit("stft"):
    #     S = SFT.stft(y)
    
    log.info("stft")
    Sxx, _, _, _ = sound.spectrogram(
        y, SR, WINDOW_TYPE, WIN_LEN, OVERLAP
    )
    
    print(Sxx)
    print(Sxx.shape)
    
    log.info("power -> db")
    S = util.power2dB(Sxx, 80)

    n_frames = to_frames(get_wav_length(rec_path))
    Y = np.zeros(shape=(4, n_frames))
    
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

    

# TODO paddings
