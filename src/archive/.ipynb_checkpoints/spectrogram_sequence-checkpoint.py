## !! old bad module, not to be used, will be remove later 

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
def evaluation_lists(Y_pred, Y_true, threshold=0.5):
    from sklearn.metrics import f1_score, precision_score, recall_score, average_precision_score
    f1, prec, rec, AP = [], [], [], []
    for label, i in DEFAULT_TOKENS.items():
        threshold = 0.5
        Y_true_class = Y_true[:, i]
        Y_pred_binary = (Y_pred[:, i] >= threshold).astype(int)
        f1 += f1_score(Y_true_class, Y_pred_binary)
        prec += precision_score(Y_true_class, Y_pred_binary)
        rec += recall_score(Y_true_class, Y_pred_binary)
        AP += average_precision_score(Y_true_class, Y_pred[:, i])

    return f1, prec, rec, AP"""