import librosa
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import subprocess
from os import path
import soundfile as sf
import sqlite3 

## TODO
# https://docs.python.org/3/tutorial/modules.html
# https://ravensoundsoftware.com/wp-content/uploads/2019/04/Raven14UsersManual_ch9-Correlation.pdf
# replace some of 7 for 1

## TODO eventual
# start saving mass figures 
# proper documentation

def freq_range(S, sr, range):
    """ frequency constricted stft S 
    """
    start_hz, end_hz = range
    low_i = int(np.floor(start_hz * (S.shape[0] * 2) / sr))
    high_i = int(np.ceil(end_hz * (S.shape[0] * 2) / sr))
    return S[low_i:high_i, :]

def bounded_box(y, sr, time_segment, frequency_range=None):
    """ retrive stft of the given given waveform restricted by time and frequency
    """
    start_s, end_s = time_segment
    y = y[int(sr * start_s):int(sr * end_s)]
    S = librosa.stft(y) 
    if frequency_range:
        S = freq_range(frequency_range)
    return S
    
def extract_labels(audacity_project):
    """ TODO
    """
    conn = sqlite3.connect(audacity_project)
    cursor = conn.cursor()

    # cursor.execute("SELECT * FROM labels")
    labels = cursor.fetchall()
    conn.close()
    
    extracted_labels = []
    print(labels)
    for label in labels:
        label_id, tmin, tmax, text, tlabel, fmin, fmax = label
        extracted_labels.append({
            'id': label_id,
            'tmin': tmin,
            'tmax': tmax,
            'text': text,
            'tlabel': tlabel,
            'fmin': fmin,
            'fmax': fmax
        })
    
    return extracted_labels

def get_depl_dir(deployment, site):
    if os.name == 'nt':
        base_dir = "E:\\acoustic-AL-dataset"
    else:
        base_dir = "/media/joel/One Touch/Joel"
        
    site_1, site_7 = path.join(base_dir, "site01"), path.join(base_dir, "site02")
    site = site_1 if site == 1 else site_7
    return path.join(site, "deployment_" + str(deployment).zfill(3))

def get_deployment_summary(deployment, site=1):
    depl = get_depl_dir(deployment, site)
    for i in os.listdir(depl):
        if "summary" in i.lower():
            with open(path.join(depl, i), "r") as f:
                df = pd.read_csv(f)
            return df
        
def list_data(deployment, site=1):
    data = path.join(get_depl_dir(deployment, site) + "/Data")
    return os.listdir(data)

def open_a(deployment=1, site=1, file_id=None, index=None, file=None):
    """ opens the specified audio segment (filename) or file index (index) in audacity
    """
    bn = "\"C://Program Files//Audacity//Audacity.exe\"" if os.name == "nt" else "audacity"
    
    if not (file_id or index or file):
        return ValueError("filename or index required")

    if file:
        subprocess.run(f"{bn} {file}", shell=True)
        return

    data_dir = path.join(get_depl_dir(deployment, site), "Data")
    for i, file in enumerate(os.listdir(data_dir)):
        p = path.join(data_dir, file)
        if file == file_id or i == index:
            print("opening ", p)
            subprocess.run(f"{bn} {p}", shell=True)

def load(file_path, sr=None):
    return librosa.load(file_path, sr=sr) if sr else librosa.load(file_path)

def extract_segment(file, output_file, start_s, end_s):
    """ save part of a sound file 
    """
    y, sr = librosa.load(file)
    start_sample = int(start_s * sr)
    end_sample = int(end_s * sr)
    y_segment = y[start_sample:end_sample]
    sf.write(output_file, y_segment, sr)