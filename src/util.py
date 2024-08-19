""" Optional utily fucntions/workflows for interacting with the dataset
"""

import os
from os import path

from config import *

import librosa
import numpy as np
import pandas as pd
import soundfile as sf
from pathlib import Path
import audioread
import time
from contextlib import contextmanager

from pathlib import Path

log = logging.getLogger(__name__)


@contextmanager
def timeit(priori_message=None):
    """ Log the time take for contextual code block
    """
    if priori_message:
        log.debug(priori_message)

    start_time = time.time()
    yield
    end_time = time.time()
    elapsed_time = end_time - start_time
    log.debug(f" * Time Taken: {elapsed_time:.6f} seconds")


def get_wav_length(filepath) -> float:
    """ efficient wav duration computation given in seconds
    """
    with audioread.audio_open(filepath) as f:
        duration = f.duration
    return duration


def extract_segment(file, output_file, time_segment: tuple):
    """Save a time segment of a sound file to the given output
    """
    start, end = time_segment
    y, sr = librosa.load(file)
    start_sample = int(start * sr)
    end_sample = int(end * sr)
    y_segment = y[start_sample:end_sample]
    sf.write(output_file, y_segment, sr)
    

class WavDataset(dict):
    """ General use dataset for wav datasets. 
    
    Summary:
        This class inherits a dictionary with the mapping WavDataset["file_name.wav"] = path_to_wav.
        Infers the root directory contains unqiue, long, wav formatted recordings, ignoring all other files 
        and the folder hierarchy. 

    
    Example Usage:
        ds = WavDataset()
        recording_path = ds["1_20230316_063000.wav"]
        y, sr = librosa.load(recording_path)
        ...

    """
    root: str
    
    def __init__(self, dataset_root=DATA_ROOT, reject_duplicates=False):
        self.root = dataset_root
        self._parse_wav_files(reject_duplicates)
        
    def _parse_wav_files(self, reject_duplicates: bool):
        for root, dirs, files in os.walk(self.root):
            for file in files:
                if file.lower().endswith('.wav'):
                    full_path = Path(root) / file
                    if file in self and not reject_duplicates:
                        raise RuntimeError(
                            f"duplicate wav name file found:  {full_path}. \
                                Only unique file names are accepted, consider setting reject_duplicates=True"
                        )
                    self[file] = full_path
    
    def get_wav_files(self) -> list:
        return list(self.keys())
    
    def get_wav_paths(self) -> list:
        return list(self.values())
    
    
