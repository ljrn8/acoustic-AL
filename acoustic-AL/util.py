""" Optional utily fucntions/workflows for interacting with the dataset

"""

import os
from os import path

import librosa
import numpy as np
import pandas as pd
import soundfile as sf
from config import *
from pathlib import Path
import audioread
import time
from contextlib import contextmanager
import logging; logger = logging.getLogger(__name__)

@contextmanager
def timeit(priori_message=None):
    if priori_message:
        print(priori_message, end=" ", flush=True)
        
    start_time = time.time()
    yield
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"\t\t\t| Time Taken: {elapsed_time:.6f} seconds", flush=True)

class BoundedBox():
    """Optional, Convenient Spectrogam Bounded Box represenation. 
    May be used for for template matching or annotations.
    
    Arguments:
        frequency_lims (tuple): low, high frequency limits in hertz
        time_segment (tuple): start, end time in seconds for the given segment
        source_recording_path (str): path to the original recording path containing the segment
        name (str, Optional): optional identifier/label of the bounded box 
        y (int): sample amplitude list for the audio segment 
        sr (int): sample rate (seconds/sample) for the audio segment
        S (int): stft array for the audio segment
    
    """
    frequency_lims: tuple
    time_segment: tuple
    source_recording_path: str
    name: str = None
    y: int
    sr: int
    S: int

    def __init__(
        self, source_recording_path, time_segment, frequency_lims=None, sr=None, **kwargs
    ):
        self.frequency_lims = frequency_lims
        self.time_segment = time_segment
        self.source_recording_path = source_recording_path
        self.__dict__.update(kwargs)
        
        y, sr_given = librosa.load(source_recording_path, sr=sr)
        self.sr = sr or sr_given
        self.S, self.y = self.get_stft(y, self.sr)

    def write_audio_segment(self, output_file_path, widen=0):
        y, sr = librosa.load(self.source_recording_path)
        start_s, end_s = self.time_segment
        start_s -= widen
        end_s += widen

        start_sample = int(start_s * sr)
        end_sample = int(end_s * sr)
        y_segment = y[start_sample:end_sample]
        sf.write(output_file_path, y_segment, sr)

    def get_stft(self, y, sr):
        start_s, end_s = self.time_segment
        y = y[int(sr * start_s) : int(sr * end_s)]
        S = librosa.stft(y)
        if self.frequency_range:
            start_hz, end_hz = self.frequency_range
            low_i = int(np.floor(start_hz * (S.shape[0] * 2) / sr))
            high_i = int(np.ceil(end_hz * (S.shape[0] * 2) / sr))
            S = S[low_i:high_i, :]

        return S, y




class Dataset():
    """ Optional Dataset representaion for interacting with recordings. 
    Infers the dataset is structured in a specific form and is thus incompatible with general datasets
    
    Attributes:
        root (str): root directory for the dataset (containes sites)
        sites (list[str]): list of site names
    """

    root: str
    sites: list

    def __init__(self, dataset_root=None):
        self.root = dataset_root or DATA_ROOT
        self.sites = self._get_sites()
        
    def _get_sites(self) -> list:
        folders = os.listdir(self.root)
        return [
            site for site in folders if path.isdir(path.join(self.root, site))
        ]
    
    def get_deployment_path(self, deployment, site) -> Path:
        site = "site" + str(site).zfill(2)
        return Path(self.root) / site / ("deployment_" + str(deployment).zfill(3))
    
    def get_data_path(self, deployment, site) -> Path:
        return self.get_deployment_path(deployment, site) / 'Data'

    def get_recordings(self, deployment, site) -> list:
        return os.listdir(self.get_data_path(deployment, site))
    
    def get_duration(self, file, deployment, site, sr=10_000) -> float:
        y, sr = librosa.load(
            Path(self.get_deployment_path(deployment, site)) / "Data" / file, sr=sr
        )
        return librosa.get_duration(y=y, sr=sr)

    def get_deployment_summary(self, deployment, site) -> pd.DataFrame:
        depl = self.get_deployment_path(deployment, site)
        for i in os.listdir(depl):
            if "summary" in i.lower():
                with open(path.join(depl, i), "r") as f:
                    df = pd.read_csv(f)
                return df
            
    @staticmethod
    def get_wav_length(filename) -> float:
        with audioread.audio_open(filename) as f:
            duration = f.duration 
        return duration 
                
    @staticmethod
    def extract_segment(file, output_file, time_segment: tuple):
        """Save a time segment of a sound file to the given output

        Args:
            file (str): file where segment exists
            output_file (_type_): file to write the segment
            time_segment (tuple[int]): start, end time in seconds for the given segment
        
        """
        start, end = time_segment
        y, sr = librosa.load(file)
        start_sample = int(start * sr)
        end_sample = int(end * sr)
        y_segment = y[start_sample:end_sample]
        sf.write(output_file, y_segment, sr)
 

