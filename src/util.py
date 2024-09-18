""" Optional utily functions/workflows for interacting with the dataset
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
import subprocess
from contextlib import contextmanager

from pathlib import Path

log = logging.getLogger(__name__)

def open_visualizer(filepath: Path):
    recording = filepath.name
    command = f'cd {filepath.parent} && \
        /mnt/c/Program\ Files/Sonic\ Visualiser/Sonic\ Visualiser.exe {recording} \
            >> /home/ethan/working/acoustic-AL/scripts/logs/svisual'
    process = subprocess.Popen(command, shell=True, stdout=subprocess.PIPE) # svisual not found
    process.wait()
    print(process.returncode)
    

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


def read_audio_section(filename, start_time, stop_time):
    """ read audio section (y, sr) efficiently from a wav file 
    """
    track = sf.SoundFile(filename)
    can_seek = track.seekable() 
    if not can_seek:
        raise ValueError("not compatible with seeking")

    sr = track.samplerate
    start_frame = int(sr * start_time)
    frames_to_read = int(sr * (stop_time - start_time))
    track.seek(start_frame)
    audio_section = track.read(frames_to_read)
    return audio_section, sr

def get_wav_length(filepath) -> float:
    """ efficient wav duration computation given in seconds
    """
    with audioread.audio_open(filepath) as f:
        duration = f.duration
    return duration


def extract_segment(file, output_file, time_segment: tuple):
    """ Save a time segment of a sound file to the given output
    """
    start, end = time_segment
    y, sr = librosa.load(file)
    start_sample = int(start * sr)
    end_sample = int(end * sr)
    y_segment = y[start_sample:end_sample]
    sf.write(output_file, y_segment, sr)
       

## !! dont use this (will be removed)
class MauritiusDataset:
    """ Old dataset class used for interactive with the muarities deployments. 
    Will be remove before publication, use WavDataset above for general use/

    Attributes:
        root (str): root directory for the dataset (containes sites)
        sites (list[str]): list of site names
    """

    root: str
    sites: list
    recordings: dict

    def __init__(self, dataset_root=None):
        self.root = dataset_root or DATA_ROOT
        self.sites = self._get_sites()

    # DO remove this (possibly no sites)
    def _get_sites(self) -> list:
        folders = os.listdir(self.root)
        return [site for site in folders if path.isdir(path.join(self.root, site))]

    def get_deployment_path(self, deployment, site) -> Path:
        site = "site" + str(site).zfill(2)
        return Path(self.root) / site / ("deployment_" + str(deployment).zfill(3))

    def get_data_path(self, deployment, site) -> Path:
        return self.get_deployment_path(deployment, site) / "Data"

    def get_recordings(self, deployment, site, path=False) -> list:
        recs = os.listdir(self.get_data_path(deployment, site))
        if not path:
            return recs
        else:
            return [self.get_data_path(deployment, site) / r for r in recs]

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


