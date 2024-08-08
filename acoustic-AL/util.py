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

from tensorflow.keras.utils import Sequence
from sklearn.preprocessing import LabelBinarizer

    
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
    sites: list[str]

    def __init__(self, dataset_root=None):
        self.root = dataset_root or DATA_ROOT
        self.sites = self._get_sites()
        
    def _get_sites(self):
        return [site for site in os.listdir(self.root) if path.isdir(path.join(self.root, site))]
    
    def get_deployment_path(self, deployment, site):
        site = "site" + str(site).zfill(2)
        return path.join(self.root, site, "deployment_" + str(deployment).zfill(3))
    
    def get_data_path(self, deployment, site):
        return Path(self.get_deployment_path(deployment, site)) / 'Data'

    def get_recordings(self, deployment, site):
        return os.listdir(self.get_data_path(deployment, site))
    
    def get_duration(self, file, deployment, site, sr=10_000):
        y, sr = librosa.load(
            Path(self.get_deployment_path(deployment, site)) / "Data" / file, sr=sr
        )
        return librosa.get_duration(y=y, sr=sr)

    def get_deployment_summary(self, deployment, site):
        depl = self.get_deployment_path(deployment, site)
        for i in os.listdir(depl):
            if "summary" in i.lower():
                with open(path.join(depl, i), "r") as f:
                    df = pd.read_csv(f)
                return df
            
    @staticmethod
    def extract_segment(file, output_file, time_segment: tuple[int]):
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
 



class SpectrogramSequence(Sequence):
    # TODO overlapping chunks
    """

    """

    def __init__(self, annotations_df, ds: Dataset, chunk_len_seconds=10, batch_size=32, sr=96_000):
        self.annotations = annotations_df
        self.ds = ds
        self.batch_size = batch_size

        self.sr = sr
        self.chunk_len = int(librosa.time_to_frames(chunk_len_seconds, sr=sr))
        
        self.chunk_info = [] # (recording, start_frame, end_frame, y)
        
        self.label_tokens = LabelBinarizer().fit(self.annotations['label']).classes_
        
        # !!
        print("ensure that this is a tokenization: ", self.label_tokens, self.label_tokens["fast_trill_6khz"])
        
        
        for recording, df in self.annotations.groupby('recording'): 
                    
            p = Path(ds.get_data_path(df['deployment'], df['site'])) / recording
            y, sr = librosa.load(p, sr=sr)
            
            S = librosa.stft(y)
            S_db = librosa.amplitude_to_db(np.abs(S), ref=np.max)
            
            # start and end time in seconds
            start_time = librosa.frames_to_time(start_idx, sr=sr)
            end_time = librosa.frames_to_time(start_idx + self.chunk_len, sr=sr)
            
            for start_idx in range(0, S_db.shape[1], self.chunk_len):
                y = np.zeros(shape=(self.chunk_len, 4), dtype=np.int32)
                
                for row, i in df.iterrows():
                    if end_time > row['min_t'] > start_time or end_time > row['max_t'] > start_time:

                        label_start_frame = librosa.frames_to_time(row['min_t'] - start_time, sr=sr)
                        label_end_frame = librosa.frames_to_time(row['max_t'] - end_time, sr=sr)
                        
                        # Set labels in the y map for frames covering the event
                        y[self.label_tokens[row["label"]], label_start_frame:label_end_frame + 1] = 1  # +1 to include the end frame

                self.chunk_info.append(
                    (recording, start_idx, start_idx + self.chunk_len, y)
                )
        

    # def __len__(self):
    #     return self.datalen // self.batch_size
           
