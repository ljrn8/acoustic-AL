import matplotlib.pyplot as plt
import numpy as np
import librosa
import librosa.display
from IPython.display import Audio


def plot_spectrogram(y, sr, title, chunk_duration=10):
    num_chunks = int(np.ceil(len(y) / (chunk_duration * sr)))
    for i in range(num_chunks):
        start = i * chunk_duration * sr
        end = min((i + 1) * chunk_duration * sr, len(y))
        y_chunk = y[start:end]

        plt.figure(figsize=(14, 5))
        S = librosa.feature.melspectrogram(y=y_chunk, sr=sr, n_mels=128, fmax=sr/2)
        S_dB = librosa.power_to_db(S, ref=np.max)
        librosa.display.specshow(S_dB, sr=sr, x_axis='time', y_axis='mel', fmax=sr/2)
        plt.colorbar(format='%+2.0f dB')
        plt.title(f"{title} (Chunk {i+1}/{num_chunks})")
        plt.show()
        # plt.savefig("figures/all.png") NOTE as large as the files themselves

def load_and_display_audio(file_path, chunk_duration=10):
    y, sr = librosa.load(file_path, sr=None)
    print(f"Loaded {file_path} with sample rate {sr}")
    plot_spectrogram(y, sr, f"Spectrogram of {file_path}", chunk_duration)
    return Audio(data=y, rate=sr)


###

from util import *
from os import path

data = get_depl_dir(1, 1) + "\\Data"
files = os.listdir(data)
audio_files = [ path.join(data, file) for file in files[:2] ]


for file in audio_files:
    audio_player = load_and_display_audio(file)
    display(audio_player)
    
    


