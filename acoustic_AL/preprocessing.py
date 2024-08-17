from tensorflow.keras.utils import Sequence
from maad import sound, util
import librosa

from .config import *
from .util import Dataset, timeit

from pathlib import Path
import numpy as np
import time
from tqdm import tqdm
import soundfile as sf
import pandas as pd
from scipy.signal import resample_poly

import scipy.io.wavfile as wavfile

import logging; log = logging.getLogger(__name__)

class SpectrogramSequence(Sequence):
    # TODO overlapping chunks
    
    DEFAULT_TOKENS = { # NOTE 1hot?
        'fast_trill_6khz': 0,
        'nr_syllable_3khz': 1,
        'triangle_3khz': 2,
        'upsweep_500hz': 3,
    }
    
    nperseg = 1024
    noverlap = 512
    window = "hann"
    db_range = 80

    def __init__(
        self,  
        annotations_df,
        ds: Dataset = Dataset(DATA_ROOT), 
        chunk_len_seconds=10, 
        batch_size=32, 
        sr=96_000, 
        label_tokens=DEFAULT_TOKENS
    ):
        
        self.annotations = annotations_df
        self.ds = ds
        self.batch_size = batch_size
        self.sr = sr
        self.label_tokens = label_tokens
        self.chunk_info = []
        
        # chunk length in frames
        self.chunk_len = librosa.time_to_frames(chunk_len_seconds, sr=sr,
                             hop_length=self.noverlap,
                            n_fft=self.nperseg)
        print("n# frames in chunk: ", self.chunk_len )
        
        # iterate over each recording and its annotations dataframe
        for recording, annotations_group_df in tqdm(self.annotations.groupby('recording'), desc='preparing data'): 
            depl, site = int(annotations_group_df['deployment'].iloc[0]), int(annotations_group_df['site'].iloc[0])
            p = Path(ds.get_data_path(depl, site)) / recording

            # max time
            n_frames = librosa.time_to_frames(annotations_group_df["recording_length"].iloc[0], sr=self.sr,
                                              hop_length=self.noverlap,
                                                n_fft=self.nperseg)

            # encode the entire sample (time step) classification matrix
            Y_all = self._extract_samplewise_annotations(annotations_group_df, n_frames)
            
            # cache X, Y for each seperate chunk, grouped by recording
            # NOTE just group these by recording not chunk?
            for start_frame in range(0, n_frames - self.chunk_len, self.chunk_len):
                
                end_frame = start_frame + self.chunk_len
                X_info = (site, depl, recording, start_frame, end_frame)
                
                self.chunk_info.append(
                    (X_info, Y_all[start_frame:end_frame, :])
                )


    def _extract_samplewise_annotations(self, rec_df, n_frames) -> np.array:
        Y_recording = np.zeros(shape=(n_frames, 4))
        for (label, label_index) in self.label_tokens.items():
            label_annotations = rec_df[rec_df["label"] == label]
            label_start_frames = librosa.time_to_frames(label_annotations["min_t"], 
                                    sr=self.sr, hop_length=self.noverlap, n_fft=self.nperseg)
            label_end_frames = librosa.time_to_frames(label_annotations["max_t"], 
                                    sr=self.sr, hop_length=self.noverlap, n_fft=self.nperseg)
            
            for start, end in zip(label_start_frames, label_end_frames):
                Y_recording[start: end, label_index] = 1 
        
        return Y_recording
    
    
    def __len__(self):
        return len(self.chunk_info) // self.batch_size
    
          
    def _load_and_process_recording(self, recording_info) -> np.array:
        site, depl, recording, start_frame, end_frame = recording_info

        with timeit("loading new recording: " +  recording):
            start_time = time.time()
            recording_path = self.ds.get_data_path(depl, site) / recording
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


    def __getitem__(self, idx):
        batch = self.chunk_info[idx * self.batch_size: (idx + 1) * self.batch_size]
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
                S_db = self._load_and_process_recording(X_info)
            
            assert S_db is not None # sanity check
            S_slice = S_db[:, start_frame:end_frame]
            assert S_slice.shape[1] == Y.shape[0], 'spectrogram chunk shape abnormal for ' + X_info

            batch_x.append(S_slice)
            batch_y.append(Y)
            prev = cur

            # debugging
            running_X_batch_len += S_slice.shape
            running_Y_batch_len += Y.shape

        logging.debug(f'batch average spectrogram chunk shape: {(running_X_batch_len / 32)}')
        logging.debug(f'batch average Y shape: {(running_Y_batch_len / 32)}')
        return np.array(batch_x), np.array(batch_y)
    
    
    
