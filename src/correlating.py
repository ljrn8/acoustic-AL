"""Template matching/Correlation utilities for generating annotations of short audio signatures.

TODO
* refactor to use general WavDataset
    
"""

import os
import pickle
from os import path
from pathlib import Path

from plotting import view_spectrogram
from util import BoundedBox, MauritiusDataset
from config import CORRELATIONS

import numpy as np
import pandas as pd
from maad import sound, util
from maad.rois import template_matching
import librosa
import soundfile as sf


class Template(BoundedBox):
    """Sample Template used for cross correlating over audio data

    Args:
        source_recording_path (str):
            path to the audio file containing the segment
        time_segment (tuple[int]):
            (start, end) in seconds for the segment
        frequency_lims (tuple[int]):
            (high, low) frequency limits fro the segment in hz
        sr: 
            custom sample rate for reading the template/dataset 

    Other Attributes:
        raw_correlations (pandas.DataFrame): stored raw output after running Template.template_match()
        incremental_correlations (list): deployment wize output after running Template.template_match()

    """

    raw_correlations: pd.DataFrame = None
    incremental_correlations = {}

    frequency_lims: tuple
    time_segment: tuple
    source_recording_path: str
    name: str = None
    y: int
    sr: int
    S: int

    def __init__(
        self,
        source_recording_path,
        time_segment,
        frequency_lims=None,
        sr=None,
        **kwargs,
    ):
        self.frequency_lims = frequency_lims
        self.time_segment = time_segment
        self.source_recording_path = source_recording_path
        self.__dict__.update(kwargs)

        y, sr_given = librosa.load(source_recording_path, sr=sr)
        self.sr = sr or sr_given
        self.S, self.y = self.get_stft(y, self.sr)

    def view(self, **kwargs):
        view_spectrogram(
            file_path=self.source_recording_path,
            time_segment=self.time_segment,
            frequency_range=self.frequency_range,
            **kwargs,
        )
    
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
    
    
    def template_match(
        self,
        dataset: MauritiusDataset,
        just_one_recording: str = None,
        n_deployments: int = 1,
        site: int = 1,
        thresh: float = 0.5,
        save_incremental: bool = True,
    ):
        """correlate the give template object agains the given dataset, storing threshold matches.

        Args:
            dataset (utils.Dataset): dataset object represnting sites and deplyoments.
            just_one_recording (str, optional): single recording filename for testing purposes.
            n_deployments (int, optional): number of deployments to run the correlations for.
            site (int, optional): site number. Defaults to 1.
            thresh (float, optional):
                NNC threshhold by which a match is recorded. Defaults to 0.5.
            save_incremental (bool, optional):
                incrementally save correlations incase of runtime interruption. Defaults to True.

        Returns:
            pd.Dataframe: dataframe containing template matches and metadata
        """

        if not (just_one_recording or n_deployments):
            raise RuntimeError("recording or n_files or n_deployments required")

        start_s, end_s = self.time_segment
        f_low, f_high = self.frequency_lims

        # parameters
        tlims = (start_s, end_s)
        flims = (f_low, f_high)
        nperseg = 1024
        noverlap = 512
        window = "hann"
        db_range = 80

        # template spectrogram
        s, fs = sound.load(self.source_recording_path)
        Sxx_template, _, _, _ = sound.spectrogram(
            s, fs, window, nperseg, noverlap, flims, tlims
        )
        Sxx_template = util.power2dB(Sxx_template, db_range)

        # view template
        print(start_s, end_s)
        view_spectrogram(
            file_path=self.source_recording_path,
            time_segment=(start_s, end_s),
            frequency_range=(f_low, f_high),
            playback=False,
        )
        print("time segment: ", tlims)
        print("frequency range: ", flims)
        print("source: ", self.source_recording_path)

        def _correlate_recording(recording_path):

            # recording template
            try:
                s, fs = sound.load(recording_path)
                Sxx_audio, tn, fn, ext = sound.spectrogram(
                    s, fs, window, nperseg, noverlap, flims
                )
                Sxx_audio = util.power2dB(Sxx_audio, db_range)
            except Exception as e:
                print(e)
                return None

            # NCC
            xcorrcoef, rois = template_matching(
                Sxx_audio, Sxx_template, tn, ext, thresh
            )
            rois["min_f"] = flims[0]
            rois["max_f"] = flims[1]
            return rois

        if just_one_recording:
            print("correlating to a specific recording: ", just_one_recording)
            large_audio_path = path.join(
                dataset.get_data_path(1, 1), "Data", just_one_recording
            )
            return _correlate_recording(large_audio_path)

        else:

            # correlate template over multiple deployments
            all_corrs = pd.DataFrame()
            if save_incremental:
                incremental_corrs_path = (
                    Path(CORRELATIONS) / self.name / (self.name + "_incremental")
                )
                incremental_corrs_path.mkdir(exist_ok=True)

            for deployment in range(1, n_deployments + 1):
                print(f"\n\n == DEPLOYMENT {deployment} == ")

                # get the file paths for the deployment
                data = dataset.get_data_path(deployment, site)
                wav_files = os.listdir(data)
                paths = [path.join(data, wav) for wav in wav_files]

                for i, recording_path in enumerate(paths):

                    # correlate recording
                    print("correlating: ", recording_path, f" | {i} / {len(paths)}")
                    recording_correlations = _correlate_recording(recording_path)
                    if recording_correlations is None:
                        continue

                    print("n# matches: ", len(recording_correlations))

                    # add context to correlations
                    l = len(recording_correlations)
                    recording_correlations["recording"] = [
                        path.basename(recording_path)
                    ] * l
                    recording_correlations["deployment"] = [deployment] * l
                    recording_correlations["site"] = [1] * l

                    # combine with all correlations
                    all_corrs = pd.concat(
                        [all_corrs, recording_correlations], ignore_index=True
                    )

                    if save_incremental:
                        self.incremental_correlations[deployment] = (
                            recording_correlations
                        )
                        with open(
                            Path(incremental_corrs_path)
                            / f"depl_{deployment}_recording_{i}.pkl",
                            "wb",
                        ) as f:
                            pickle.dump(recording_correlations, f)

                print(f"\n == total total running corrs = {len(all_corrs)} == ")

            self.raw_correlations = all_corrs
            return all_corrs
    
    @staticmethod
    def filter_correlations(df, thresh=0.65, overlap_cutoff=0.5):
        """filter the given correlations dataframe but xcorrcoef threshhold and remove near duplicates

        Args:
            df (Dataframe): dataframe expressing correlations.
            thresh (float, optional): xcorrcoef threshhold. Defaults to 0.65.
            overlap_cutoff (float, optional): seconds by which nearby correlations are removed leaving
                only the first occurance. Defaults to 0.5.

        Returns:
            Dataframe: filtered correlations
        """
        print("previous n# correlations: ", len(df))
        df_filtered = df[df["xcorrcoef"] >= thresh]
        df_filtered = df_filtered[
            (df_filtered["min_t"].isna())
            | (df_filtered["min_t"].diff() ** 2 > overlap_cutoff**2)
        ]
        print("filtered n# correlations: ", len(df_filtered))
        return df_filtered


    
