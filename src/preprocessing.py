""" Preprocessing classes/functions

TODO
* refactor (nperseg ect)
* overlapping chunks
* yield from h5py datasets 

"""


from tensorflow.keras.utils import Sequence
from maad import sound, util
import librosa

from config import *
from util import WavDataset, timeit

from pathlib import Path
import numpy as np
import time
from tqdm import tqdm
import soundfile as sf
import pandas as pd
from scipy.signal import resample_poly
import h5py

import scipy.io.wavfile as wavfile

import logging

log = logging.getLogger(__name__)

DEFAULT_TOKENS = {  # NOTE 1hot?
        "fast_trill_6khz": 0,
        "nr_syllable_3khz": 1,
        "triangle_3khz": 2,
        "upsweep_500hz": 3,
    }

class SpectrogramSequence(Sequence):
    # TODO overlapping chunks

    nperseg = 1024
    noverlap = 512
    window = "hann"
    db_range = 80

    def __init__(
        self,
        ds: WavDataset = WavDataset(DATA_ROOT),
        chunk_length_seconds=10,
        chunk_overlap_seconds=3,
        batch_size=32,
        train_hdf5_file=INTERMEDIATE / 'train.hdf5',
        label_tokens=DEFAULT_TOKENS,
        sr=22_000
    ):

        self.ds = ds
        self.batch_size = batch_size
        self.label_tokens = label_tokens
        self.sr = sr
        
        # data
        train_f = h5py.File(train_hdf5_file, 'r')
        self.recordings = list(train_f)
        
        # chunk len/overlap in frames
        self.chunk_len = self.s_to_frames(chunk_length_seconds)
        self.chunk_overlap = self.s_to_frames(chunk_overlap_seconds)
        
        log.info(f"n# frames in a chunk: {self.chunk_len}")
        log.info(f"n# frames in chunk overlap: {self.chunk_overlap}")
        
        
        self.chunk_info = []
        
        for recording in tqdm(self.recordings, desc='preparing chunks from hdf5 database'):
            
            
            
            dataset = train_f[recording]
            
            X = dataset["X"]
            Y = dataset["Y"]
            
            n_frames = X.shape[1] 
            hop = self.chunk_len - self.chunk_overlap
            
            assert n_frames == X.shape[1], f'mismatch shapes in {recording}: {X.shape}, {Y.shape}'

            # load in all chunk indexes for this recording            
            self.chunk_info += [
                (
                    recording, start_frame, min(start_frame + self.chunk_len, n_frames-1), 
                ) 
                for start_frame in range(0, n_frames, hop)
            ]
            
        log.info("shuffling chunks")
        self.chunk_info = np.array(self.chunk_info)
        np.random.shuffle(self.chunk_info)
        log.info(f'chunk info length: {len(self.chunk_info)}')
        train_f.close()
    
    
    
    def s_to_frames(self, seconds):
        return librosa.time_to_frames(
            seconds,
            sr=self.sr,
            hop_length=self.noverlap,
            n_fft=self.nperseg,
        )
        
    
    def __len__(self):
        return len(self.chunk_info) // self.batch_size
    
    
    def __getitem__(self, idx):
        batch = self.chunk_info[idx * self.batch_size : (idx + 1) * self.batch_size]
        batch_x = []
        batch_y = []
        prev = None
        S_db = None

        running_X_batch_len = np.zeros(2, dtype=np.float)
        running_Y_batch_len = np.zeros(2, dtype=np.float)

        for X_info, Y in batch:
            site, depl, recording, start_frame, end_frame = X_info
            cur = recording
            if cur != prev:
                S_db = self._load_and_process_recording(recording)

            assert S_db is not None  # sanity check
            S_slice = S_db[:, start_frame:end_frame]
            assert S_slice.shape[1] == Y.shape[0], (
                "spectrogram chunk shape abnormal for " + X_info
            )

            batch_x.append(S_slice)
            batch_y.append(Y)
            prev = cur

            # debugging
            running_X_batch_len += S_slice.shape
            running_Y_batch_len += Y.shape

        logging.debug(
            f"batch average spectrogram chunk shape: {(running_X_batch_len / 32)}"
        )
        logging.debug(f"batch average Y shape: {(running_Y_batch_len / 32)}")
        return np.array(batch_x), np.array(batch_y)
    
    
                
                
    def _extract_samplewise_annotations(self, rec_df, n_frames) -> np.array:
        Y_recording = np.zeros(shape=(n_frames, 4))
        for label, label_index in self.label_tokens.items():
            label_annotations = rec_df[rec_df["label"] == label]
            
            label_start_frames = self.s_to_frames(label_annotations['min_t'])
            label_end_frames = self.s_to_frames(label_annotations['max_t'])
            
            for start, end in zip(label_start_frames, label_end_frames):
                Y_recording[start:end, label_index] = 1

        return Y_recording
    

    def _load_and_process_recording(self, recording) -> np.array:
        with timeit("loading new recording: " + recording):
            start_time = time.time()
            recording_path = self.ds[recording]
            samplerate, s = wavfile.read(recording_path)
            # s, samplerate = sf.read(recording_path)

        with timeit("resampling " + recording):
            if samplerate != self.sr:
                s = resample_poly(s, self.sr, samplerate)

            if len(s.shape) > 1:
                s = np.mean(s, axis=1)

        with timeit("stft " + recording):
            S, _, _, _ = sound.spectrogram(
                s, self.sr, self.window, self.nperseg, self.noverlap
            )

        with timeit("converting to db " + recording):
            S_db = util.power2dB(S, self.db_range)

        # S_db = librosa.amplitude_to_db(np.abs(S), ref=np.max)
        elapsed_time = time.time() - start_time
        print(f"\rTotal: {elapsed_time:.2f} seconds")

        return S_db
