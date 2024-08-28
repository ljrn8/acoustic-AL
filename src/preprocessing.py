""" Preprocessing classes/functions

TODO
* refactor (nperseg ect)
* overlapping chunks
* yield from h5py datasets 

"""

from tensorflow.keras.utils import Sequence
import librosa

from config import *
from util import WavDataset

import numpy as np
from tqdm import tqdm
import h5py

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
        is_validation: bool = False,
        validation_split = 0.8,

        chunk_length_seconds=10,
        chunk_overlap_seconds=3,
        batch_size=32,
        train_hdf5_file=INTERMEDIATE / 'train.hdf5',
        label_tokens=DEFAULT_TOKENS,
        sr=22_000,
        flat_labels=False,
    ):
        self.flat_labels = flat_labels
        self.batch_size = batch_size
        self.label_tokens = label_tokens
        self.sr = sr
        self.train_hdf5_file = train_hdf5_file
        
        # chunk len/overlap in frames
        self.chunk_len = self.s_to_frames(chunk_length_seconds)
        self.chunk_overlap = self.s_to_frames(chunk_overlap_seconds)
        
        # prepare chunk indexes
        self.chunk_info = self._get_chunks(is_validation, validation_split)
        
        
    def _get_chunks(self, is_validation, validation_split):
        train_f = h5py.File(self.train_hdf5_file, 'r')
        cut_index = int(len(train_f) * validation_split)
        
        if is_validation:     
            self.recordings = list(train_f)[cut_index:]            
        else:   
            self.recordings = list(train_f)[:cut_index]
            
        log.info(f"n# frames in a chunk: {self.chunk_len}")
        log.info(f"n# frames in chunk overlap: {self.chunk_overlap}")
        log.info(f"number of recordings: {len(self.recordings)}/{len(train_f)} with [validation={is_validation}, split={validation_split}]")
        
        chunk_info = []
        
        for recording in tqdm(self.recordings, desc='preparing chunks from hdf5 database'):
            dataset = train_f[recording]
            X = dataset["X"]
            Y = dataset["Y"]
            
            freq_bins = X.shape[0]
            n_frames = X.shape[1] 
            hop = self.chunk_len - self.chunk_overlap
            
            assert n_frames == X.shape[1], f'mismatch shapes in {recording}: {X.shape}, {Y.shape}'

            # load in all chunk indexes for this recording            
            for start_frame in range(0, n_frames, hop):
                if start_frame + self.chunk_len >  n_frames-1:
                    continue
                
                chunk_info.append(
                    (recording, start_frame, start_frame + self.chunk_len) 
                )
            
        log.info("shuffling chunks")
        np.random.shuffle(chunk_info)
        
        log.info(f'chunk info length: {len(chunk_info)}')
        train_f.close()
        
        # ensure every chunk is equal length
        rec, start, end = chunk_info[0]
        ref_length = end - start
        for chunk in chunk_info:
            rec, start, end = chunk
            assert ref_length == end - start
            
        log.info(f'chunk length in frames: {ref_length}')
        log.info(f'frequency bins: {freq_bins}')
        
        return chunk_info
    
    
    
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
        log.debug(f"retrieving batch {idx}")
        batch = self.chunk_info[idx * self.batch_size : (idx + 1) * self.batch_size]
        train_f = h5py.File(self.train_hdf5_file, 'r')
        batch_X = []
        batch_Y = []

        for recording, start_frame, end_frame in batch:
            log.debug(f"slicing dataset => {start_frame, end_frame}")
            X_slice = train_f[recording]["X"][:, start_frame:end_frame]
            Y_slice = train_f[recording]["Y"][:, start_frame:end_frame]

            if self.flat_labels:
                Y_slice = np.array([y_col.any() for y_col in np.array(Y_slice).T])

            shapes = X_slice.shape, Y_slice.shape
            log.debug(f'got shapes {str(shapes)}, Y_sum -> {Y_slice.sum()}') 
            batch_X.append(X_slice)
            batch_Y.append(Y_slice.T) # !! .T 
            
        train_f.close()
        return np.array(batch_X), np.array(batch_Y)
    
    
"""
                
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
"""
