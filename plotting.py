from util import *
import librosa
import pandas as pd
from os import path
import numpy as np
import matplotlib.pyplot as plt
from IPython.display import Audio, display


def plot_correlations(
    correlations_file,
    reference_wav=None,
    duration=None,
    deployment=1,
    site=1,
    save_as=None,
    frequency_lines=None,
):
    """plots a multigraph of spectrograms from the given correlations file directly from
    the date set (without first locally saving segments)
    """
    # NOTE remember to add +- 20% to see box in context
    if not (reference_wav or duration):
        return ValueError("require wav or duration in seconds")

    # get the segment length
    if reference_wav:
        print("inferring reference length from given file")
        y, sr = librosa.load(reference_wav)
        duration = librosa.get_duration(y=y, sr=sr)

    # find recordings for the given deployment
    data_folder = path.join(get_depl_dir(deployment, site), "Data")
    files = os.listdir(data_folder)
    wav_files = [file for file in files if file.lower().endswith(".wav")]

    # read correlations
    with open(correlations_file) as f:
        lines = f.readlines()
        n_graphs = len(lines)

        # setup graph
        print("showing ", n_graphs, "spectrograms")
        n_rows = (n_graphs // 15) + 1
        fig, axes = plt.subplots(n_rows, 15, figsize=(20, 5 * n_rows))

        # setup graphs
        graph_bleed = 15 - (len(wav_files) % 15)
        for ax, line in zip(np.array(axes).flatten()[:-graph_bleed], lines):
            C, time, file = line.split(",")
            time = float(time)
            file = file.strip()
            print(f"reading [{file}, {time}]", end="\r", flush=True)

            y, sr = load(path.join(data_folder, file))
            y = y[int(sr * time) : int(sr * (time + duration))]

            D = librosa.stft(y)
            S_db = librosa.amplitude_to_db(np.abs(D), ref=np.max)
            img = librosa.display.specshow(S_db, y_axis="linear", ax=ax)
            ax.set_xlabel("")
            ax.set_ylabel("")

            if frequency_lines:
                ax.axhline(y=frequency_lines[0], color="g", linestyle="-")
                ax.axhline(y=frequency_lines[1], color="g", linestyle="-")

        if save_as:
            plt.savefig(f"figures/{save_as}.png")

        print("\n")
        plt.show()


def plot_datetime(deployment=1, site=1, df=None, save_as=None):
    """plots date agianst time for all recordings of a deployment"""
    if df is None:
        df = get_deployment_summary(deployment, site)

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


def view_spectrogram(
    file_path="./test_data/chirp_test_1.wav",
    file_id=None,
    time_segment=None,
    frequency_range=None,
    playback=True,
    ax=None,
    sr=96_000,
    title=None,
    scale_y=True,
    save_as=None,
):

    dir, file = path.split(file_path)
    show_plot = ax is None
    if file_id:
        file_path = path.join(get_depl_dir(1, 1), "Data", file_id)

    # print("opening ", file_path)
    y, sr = load(file_path, sr=sr)

    # print(librosa.get_duration(y=y, sr=sr))

    if time_segment:
        start, end = time_segment
        # widen = (end - start) * 0.2
        widen = 1  # see 1 second outwards
        y = y[max(int(sr * (start - widen)), 0) : int(sr * (end + widen))]

    n_fft = 2048
    hop_length = n_fft // 4
    S = librosa.stft(y, n_fft=n_fft, hop_length=hop_length)

    # # restrict stft on frequency
    # if frequency_range:
    #     start, end = frequency_range
    #     low_i = int(np.floor(start * (S.shape[0] * 2) / sr))
    #     high_i = int(np.ceil(end * (S.shape[0] * 2) / sr))
    #     S = S[low_i:high_i, :]
    #     graph_height = min(4 * ((end-start)/10_000), 4)

    S_db = librosa.amplitude_to_db(np.abs(S), ref=np.max)

    if not ax:
        fig, ax = plt.subplots()

    if frequency_range:
        ax.axhline(y=frequency_range[0], color="g", linestyle="-")
        ax.axhline(y=frequency_range[1], color="g", linestyle="-")

    if scale_y:
        # scale_up = int(frequency_range[1] * 0.3)
        ax.set_ylim(0, frequency_range[1] + 13_000)

    if time_segment:
        ax.axvline(x=int(widen * sr / hop_length), color="g", linestyle="-")
        ax.axvline(
            x=S.shape[1] - int(widen * sr / hop_length), color="g", linestyle="-"
        )

    img = librosa.display.specshow(S_db, y_axis="linear", ax=ax, sr=sr)
    ax.set(title=title if title is not None else file)

    if save_as:
        plt.savefig(f"figures/{save_as}.png")

    if show_plot:
        plt.show()

    if playback:
        display(Audio(data=y, rate=sr))


def multi_plot_spectogram(files_dir, save_as=None):
    files = os.listdir(files_dir)
    wav_files = [file for file in files if file.lower().endswith(".wav")]

    n_files = len(wav_files)
    print("showing ", n_files, "spectrograms")
    n_rows = (n_files // 15) + 1
    fig, axes = plt.subplots(n_rows, 15, figsize=(20, 5 * n_rows))
    graph_bleed = 15 - (len(wav_files) % 15)
    for ax, file in zip(np.array(axes).flatten()[:-graph_bleed], wav_files):
        y, sr = load(files_dir + "/" + file)
        D = librosa.stft(y)

        S_db = librosa.amplitude_to_db(np.abs(D), ref=np.max)
        img = librosa.display.specshow(S_db, y_axis="mel", ax=ax)
        ax.set_xlabel("")
        ax.set_ylabel("")

    if save_as:
        plt.savefig(f"figures/{save_as}.png")

    plt.show()
