"""Template matching/Correlation utilities for generating annotations of short audio signatures.

Example usage:

    templ = template(source_recording_path=.., time_segment=.. )
    correlations = templ.template_match(
        dataset=Dataset(root),
        n_deployments=2,
        ...
    )
    correlations.to_csv(..)
    
"""

import os
import pickle
from datetime import datetime
from os import path
from pathlib import Path

import librosa
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pygetwindow as gw
from IPython.display import Audio, display
from maad import sound, util
from maad.rois import template_matching
from plotting import view_spectrogram
from config import CORRELATIONS
from util import BoundedBox, Dataset


class Template(BoundedBox):
    """Sample Template used for cross correlating over audio data 

    Args:
        source_recording_path (str): 
            path to the audio file containing the segment
        time_segment (tuple[int]): 
            (start, end) in seconds for the segment
        frequency_lims (tuple[int]): 
            (high, low) frequency limits fro the segment in hz

    Attributes:
        raw_correlations (pandas.DataFrame): stored raw output after running Template.template_match()
        incremental_correlations (list): deployment wize output after running Template.template_match()

    """
    
    raw_correlations: pd.DataFrame = None
    incremental_correlations = {}

    def __init__(self, source_recording_path, time_segment, frequency_lims, **kwargs):
        super().__init__(time_segment, source_recording_path, frequency_lims, **kwargs)

    def view(self, **kwargs):
        view_spectrogram(
            file_path=self.source_recording_path,
            time_segment=self.time_segment,
            frequency_range=self.frequency_range,
            **kwargs,
        )

    def template_match(
        self,
        dataset: Dataset,
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


    def verify_correlations(self, output_csv, custom_df=None):
        """!! NOTE: unused function, remove before publication
        """
        
        if Path(output_csv).exists():
            print(f"'{output_csv}' already exists, appending")

        if self.raw_correlations is not None:
            correlations = self.raw_correlations
        elif custom_df:
            correlations = custom_df
        else:
            raise RuntimeError(
                "existing correlations not found, run correlations via 'template_match()' or provide correlations file via 'corrs_pkl'"
            )

        Path(output_csv).touch()
        with open(output_csv, "a") as f:
            for c in correlations.columns:
                f.write(str(c) + "\t")
            f.write("\n")

        # trimm correlations to the threshhold
        counts = correlations["recording"].value_counts()
        if len(counts) >= 50:
            print("over 50 files found, only showing first 50 in correlations")
            counts = counts[:50]

        plt.figure(figsize=(8, 5))
        counts.plot(kind="bar")
        plt.xticks(fontsize=5, rotation=45)
        plt.xlabel("Recording")
        plt.ylabel("N# Correlations")
        plt.show(block=False)

        figsize = (4, 3)
        y_lim = 7_000

        # grouped by recording, iterate over all correlations
        for recording, matches_df in correlations.groupby("recording"):
            depl = matches_df["deployment"].iloc[0]
            site = matches_df["site"].iloc[0]
            rec_path = path.join(Dataset().get_data_path(depl, site), recording)

            print(f"{len(matches_df)} from: ", rec_path)
            y, sr = librosa.load(rec_path, sr=44_000)
            widen = 0.6

            for i, row in matches_df.iterrows():

                fig, ax = plt.subplots(figsize=figsize)
                frequency_range = (row["min_f"], row["max_f"])
                time_segment = (
                    row["peak_time"] - 0.949 / 2,
                    row["peak_time"] + 0.949 / 2,
                )
                title = str(row["xcorrcoef"])[:5] + "_" + path.basename(rec_path)

                n_fft = 2048
                hop_length = n_fft // 4

                # cut out spectrogram
                start, end = time_segment
                y_cut = y[max(int(sr * (start - widen)), 0) : int(sr * (end + widen))]
                S = librosa.stft(y_cut, n_fft=n_fft, hop_length=hop_length)
                S_db = librosa.amplitude_to_db(np.abs(S), ref=np.max)

                if frequency_range:
                    ax.axhline(y=frequency_range[0], color="g", linestyle="-")
                    ax.axhline(y=frequency_range[1], color="g", linestyle="-")
                    if y_lim:
                        ax.set_ylim(0, y_lim)
                    else:
                        ax.set_ylim(0, min(frequency_range[1] + 9_000, sr * 2))

                if time_segment:
                    ax.axvline(x=int(widen * sr / hop_length), color="g", linestyle="-")
                    ax.axvline(
                        x=S.shape[1] - int(widen * sr / hop_length),
                        color="g",
                        linestyle="-",
                    )

                img = librosa.display.specshow(S_db, y_axis="linear", ax=ax, sr=sr)
                ax.set(title=title)
                ax.set_xlabel(f"{start}-{end}")

                display(Audio(data=y_cut, rate=sr))
                plt.show(block=False)

                # make sure the terminal is still focused
                terminal_windows = gw.getWindowsWithTitle("Terminal")
                pws_windows = gw.getWindowsWithTitle("Powershell")
                for windows in [terminal_windows, pws_windows]:
                    if windows:
                        window = windows[0]
                        window.activate()

                ans = input("is this correlation correct [y/n]?  [default in yes]: ")
                if ans.lower() not in ["n", "no"]:
                    with open(output_csv, "a") as f:
                        for r in row:
                            f.write(str(r) + "\t")
                        f.write("\n")

                plt.clf()
                plt.close()


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
