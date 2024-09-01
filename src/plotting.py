"""Plotting functions for EDA and general visualization

example usage:
    view_spectrogram(
        recording_id="1_20230322_063000.wav",
        time_segment=(3, 4),
        playback=True,
        ...
    )

"""

import os
from os import path

import librosa
import matplotlib.pyplot as plt
import matplotlib
from matplotlib import patches
import numpy as np
import pandas as pd
from IPython.display import Audio, display
from pathlib import Path


from util import MauritiusDataset, read_audio_section


def view_spectrogram(
    file_path: str = None,
    recording_id: str = None,
    annotations: list = None,
    time_segment: tuple = None,
    frequency_range: tuple = None,
    playback: bool = False,
    ax: matplotlib.axes = None,
    sr: int = 96_000,
    title: str = None,
    y_lim: int = None,
    save_as: str = None,
    figsize: tuple = None,
    show_plot: bool = True,
    **kwargs,
) -> matplotlib.axes:
    """Plot the given spectrogram with some convenient options

    Args:
        file_path (str | Path, optional): file path of the .wav file.
        recording_id (str, optional): recording filename in the dataset.
        annotations (list, optional):
            display annotations on the spectrogram given in the form [(label, x, y, width, height), ..]
        time_segment (tuple, optional): start, end integers in seconds spectrogram within the given file.
        frequency_range (tuple, optional): low, high frequency limits in hertz.
        playback (bool, optional): display an ipython audio playback alongside the figure. Defaults to False.
        ax (matplotlib.axes, optional): custom axis for plotting.
        sr (int, optional): custom sample rate (samples/second). Defaults to 96_000.
        title (str, optional): plot title.
        y_lim (int, optional): upperbound for the spectrogram in hertz.
        save_as (str, optional): save the figure as the given file/path.
        figsize (tuple, optional): figure size (width, height).
        show_plot (bool, optional): call plt.show(). Defaults to True.

    Kwargs:
        any other keyword arguments are accepted by 'librosa.display.specshow'

    Returns:
        matplotlib.axes
    """

    if not (file_path or recording_id) and S is None:
        return ValueError("file not specified")

    show_plot = ax is None

    if recording_id:
        if not recording_id.endswith(".wav"):
            recording_id += ".wav"
        file_path = path.join(MauritiusDataset().get_data_path(1, 1), recording_id)

    if not ax:
        fig, ax = plt.subplots(figsize=figsize)

    if time_segment:
        start, end = time_segment
        widen = 0.6
        y, sr = read_audio_section(file_path, start - widen, end + widen)

    else:
        y, sr = librosa.load(file_path, sr=sr)



    n_fft = 2048
    hop_length = n_fft // 4
    S = librosa.stft(y, n_fft=n_fft, hop_length=hop_length)  # cut out time seg

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
            x=S.shape[1] - int(widen * sr / hop_length), color="g", linestyle="-"
        )

    img = librosa.display.specshow(S_db, y_axis="linear", ax=ax, sr=sr, **kwargs)

    ax.set(title=title or file_path or recording_id)

    if annotations:
        for label, x, y, width, height in annotations:
            rect = patches.Rectangle(
                (x, y), width, height, linewidth=2, edgecolor="red", facecolor="none"
            )
            ax.add_patch(rect)
            ax.text(
                x,
                y,
                label,
                color="white",
                fontsize=12,
                bbox=dict(facecolor="red", alpha=0.5),
            )

    if save_as:
        plt.savefig(save_as)
    if show_plot:
        plt.show()
    if playback:
        display(Audio(data=y, rate=sr))

    return ax


def plot_datetime(deployment=1, site=1, save_as=None):
    """plots date agianst time for all recordings of a deployment
    from the deployment summary
    """

    df["DATE"] = pd.to_datetime(df["DATE"], format="%Y-%b-%d")
    df["TIME"] = pd.to_datetime(df["TIME"], format="%H:%M:%S").dt.time
    df = df.sort_values(by="DATE")
    df["time_as_fraction"] = df["TIME"].apply(
        lambda t: t.hour + t.minute / 60 + t.second / 3600
    )

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(df["DATE"], df["time_as_fraction"], c="red")
    plt.xticks(rotation=45)
    plt.xlabel("Date")
    plt.ylabel("Time of Day")
    plt.title(f"Date vs Time of Recordings - Depl: {deployment} site: {site}")
    plt.grid(True)
    ax.set_yticks(range(1, 25, 2))
    ax.set_yticklabels(f"{str(i).zfill(2)}:00" for i in range(1, 25, 2))
    if save_as:
        plt.savefig(f"figures/{save_as}.png")

    plt.show()



def multi_plot_spectogram(files_dir, save_as=None):
    files = os.listdir(files_dir)
    wav_files = [file for file in files if file.lower().endswith(".wav")]

    n_files = len(wav_files)
    print("showing ", n_files, "spectrograms")
    n_rows = (n_files // 15) + 1
    fig, axes = plt.subplots(n_rows, 15, figsize=(20, 5 * n_rows))
    graph_bleed = 15 - (len(wav_files) % 15)
    for ax, file in zip(np.array(axes).flatten()[:-graph_bleed], wav_files):
        y, sr = librosa.load(files_dir + "/" + file)
        D = librosa.stft(y)

        S_db = librosa.amplitude_to_db(np.abs(D), ref=np.max)
        img = librosa.display.specshow(S_db, y_axis="mel", ax=ax)
        ax.set_xlabel("")
        ax.set_ylabel("")

    if save_as:
        plt.savefig(f"figures/{save_as}.png")

    plt.show()
