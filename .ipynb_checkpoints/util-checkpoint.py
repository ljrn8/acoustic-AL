import librosa
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import subprocess
from os import path
import soundfile as sf

## TODO
# start saving mass figures 
# proper documentation

# havent been able to copy over the dataset yet, just reading data for now 
# (dont worry there will be no modifications made to the drive ever)

if os.name == 'nt':
    # base_dir = "E:\acoustic-AL-dataset" later
    base_dir = "F:\\Joel"
else:
    base_dir = "/media/joel/One Touch/Joel"
    
sites = site_1, site_7 = base_dir + "/site01", base_dir + "/site02"

def get_depl_dir(deployment, site):
    site = site_1 if 1 else site_7
    return site + "/deployment_" + str(deployment).zfill(3)

def get_deployment_summary(deployment, site=1):
    depl = get_depl_dir(deployment, site)
    for i in os.listdir(depl):
        if "summary" in i.lower():
            with open(depl + "/" + i, "r") as f:
                df = pd.read_csv(f)
            return df
        
def list_data(deployment, site=1):
    data = get_depl_dir(deployment, site) + "/Data"
    return os.listdir(data)

def list_data(deployment, site=1):
    data = get_depl_dir(deployment, site) + "/Data"
    return os.listdir(data)

def plot_datetime(deployment=1, site=1, df=None, save_as=None):
    """ plots date agianst time for all recordings of a deployment 
    """
    if df is None:
        df = get_deployment_summary(deployment, site)

    df['DATE'] = pd.to_datetime(df['DATE'], format='%Y-%b-%d')
    df['TIME'] = pd.to_datetime(df['TIME'], format='%H:%M:%S').dt.time
    df = df.sort_values(by='DATE')
    df['time_as_fraction'] = df['TIME'].apply(lambda t: t.hour + t.minute/60 + t.second/3600)

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(df['DATE'], df['time_as_fraction'], c='red')
    plt.xticks(rotation=45)
    plt.xlabel('Date')
    plt.ylabel('Time of Day')
    plt.title(f'Date vs Time of Recordings - Depl: {deployment} site: {site}')
    plt.grid(True)
    ax.set_yticks(range(1, 25, 2))
    ax.set_yticklabels(f'{str(i).zfill(2)}:00' for i in range(1, 25, 2))
    if save_as:
        plt.savefig(f'figures/{save_as}.png')
        
    plt.show()

def open_a(deployment=1, site=1, file_id=None, index=None, file=None):
    """ opens the specified audio segment (filename) or file index (index) in audacity
    """
    if not (file_id or index or file):
        return ValueError("filename or index required")

    if file:
        subprocess.run(f"audacity {file}", shell=True)
        return

    data_dir = path.join(get_depl_dir(deployment, site), "Data")
    for i, file in enumerate(os.listdir(data_dir)):
        p = path.join(data_dir, file)
        if file == file_id or i == index:
            print("opening ", p)
            subprocess.run(f"audacity {p}", shell=True)

def load(file_path, sr=None):
    return librosa.load(file_path, sr=sr) if sr else librosa.load(file_path)

def view_spectogram(filename="./test_data/chirp_test_1.wav", width_factor=2):
    """ view inline spectrogram
    """
    y, sr = load(filename)
    D = librosa.stft(y) 
    S_db = librosa.amplitude_to_db(np.abs(D), ref=np.max)

    duration = librosa.get_duration(y=y, sr=sr)
    width = width_factor * duration
    width = min(width, 20)
    fig, ax = plt.subplots(
        figsize=(width, 5)
    )
    img = librosa.display.specshow(S_db, y_axis='mel', ax=ax)
    ax.set(title=filename)
    fig.colorbar(img, ax=ax, format="%+2.f dB")


def extract_segment(file, output_file, start_seconds, end_seconds):
    """ save part of a sound file 
    """
    y, sr = librosa.load(file)
    start_sample = int(start_seconds * sr)
    end_sample = int(end_seconds * sr)
    y_segment = y[start_sample:end_sample]
    sf.write(output_file, y_segment, sr)


def splice_correlations(correlations_file, reference_wav=None, 
                       duration=None, output_dir="./correlations/segments/",
                      deployment=1, site=1):
    """ produces the correleations (to be annotated) as wav files for verification 
    """
    
    if not (reference_wav or duration):
        return ValueError("require wav or duration in seconds")
    
    # get the segment length 
    if reference_wav:
        print("inferring reference length from given file")
        y, sr = librosa.load(reference_wav)
        duration = librosa.get_duration(y=y, sr=sr)
    
    # read correlations
    with open(correlations_file) as f:
        for line in f:
            time, file = line.split(", ")
            time = float(time)
            file = file.strip()
            
            # find the correlation segment in the original file
            recording_path = get_depl_dir(deployment, site) + "/Data/" + file
            extract_segment(recording_path, output_dir + "corr_" + file[:3] + "_" + str(time) + ".wav",  # TODO 3dp
                            time, time + duration)
            

def multi_plot_spectogram(files_dir):
    files = os.listdir(files_dir)
    wav_files = [file for file in files if file.lower().endswith('.wav')]
    
    n_files = len(wav_files)
    print("showing ", n_files, "spectrograms")
    n_rows = (n_files // 15) + 1
    
    fig, axes = plt.subplots(
        n_rows, 15,
        figsize=(20, 5 * n_rows)
    )

    graph_bleed = 15 - (len(wav_files) % 15)
    for ax, file in zip(np.array(axes).flatten()[:-graph_bleed], wav_files):
        y, sr = load(files_dir + "/" + file)
        D = librosa.stft(y) 
        
        S_db = librosa.amplitude_to_db(np.abs(D), ref=np.max)
        img = librosa.display.specshow(S_db, y_axis='mel', ax=ax)
        ax.set_xlabel('')
        ax.set_ylabel('')
        
    plt.show()
    