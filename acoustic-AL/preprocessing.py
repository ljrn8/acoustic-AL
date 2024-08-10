
from tensorflow.keras.utils import Sequence
from sklearn.preprocessing import LabelBinarizer
from maad import sound, util
from util import Dataset
import librosa
from pathlib import Path
from config import *
import numpy as np
import time
from tqdm import tqdm
import soundfile as sf
import pandas as pd

class SpectrogramSequence(Sequence):
    # TODO overlapping chunks
    # TODO random order (?)
    """

    """
    
    DEFAULT_TOKENS = { # NOTE 1hot?
        'fast_trill_6khz': 0,
        'nr_syllable_3khz': 1,
        'triangle_3khz': 2,
        'upsweep_500hz': 3,
    }

    def __init__(self, annotations_df, ds: Dataset, chunk_len_seconds=10, 
                 batch_size=32, sr=96_000, label_tokens=DEFAULT_TOKENS, save_sequence=True):
        
        self.annotations = annotations_df
        self.ds = ds
        self.batch_size = batch_size
        self.sr = sr
        self.chunk_len = int(librosa.time_to_frames(chunk_len_seconds, sr=sr))
        
        self.chunk_info = [] # (site, depl, recording, start_frame, end_frame, y)
        
        # self.label_tokens = LabelBinarizer().fit(self.annotations['label']).classes_
        self.label_tokens = label_tokens
        
        # annotations per recording
        
        # TODO check if duration column
        for recording, df in tqdm(self.annotations.groupby('recording'), desc='preparing data'): 
            depl, site = int(df['deployment'].iloc[0]), int(df['site'].iloc[0])
            p = Path(ds.get_data_path(depl, site)) / recording

            # max time
            frames = librosa.time_to_frames(df["recording_length"].iloc[0])

            for start_frame in range(0, frames-1, self.chunk_len):
                
                start_time = librosa.frames_to_time(start_frame, sr=sr)
                end_time = librosa.frames_to_time(start_frame + self.chunk_len, sr=sr)
            
                y = np.zeros(shape=(self.chunk_len, 4), dtype=np.int32)
                
                for i, row in df.iterrows():
                    if end_time > row['min_t'] > start_time or end_time > row['max_t'] > start_time:

                        label_start_frame = librosa.time_to_frames(row['min_t'] - start_time, sr=sr)
                        label_end_frame = librosa.time_to_frames(row['max_t'] - end_time, sr=sr)
                        
                        # Set labels in the y map for frames covering the event
                        y[self.label_tokens[row["label"]], label_start_frame:label_end_frame + 1] = 1  # +1 to include the end frame

                self.chunk_info.append(
                    ((site, depl, recording, start_frame, start_frame + self.chunk_len), y)
                )

        # convert instance data to df
        if save_sequence:
            data = [(*chunk[0], chunk[1]) for chunk in self.chunk_info]  # Extract and flatten the tuples
            df = pd.DataFrame(data, columns=["site", "depl", "recording", "start_frame", "end_frame", "y"])
            p = OUTPUT_DIR / 'intermediate' / 'instance_data.csv'
            p.parent.mkdir(parents=True, exist_ok=True)
            p.touch()
            df.to_csv(p)


    # def __len__(self):
    #     return self.datalen // self.batch_size
           
    # TODO parallelize? read in segment only in sf? 
    def __getitem__(self, idx):
        batch = self.chunk_info[idx * self.batch_size: (idx + 1) * self.batch_size]
        batch_x, batch_y = [], []

        nperseg = 1024
        noverlap = 512
        window = "hann"
        db_range = 80

        prev = None
        S_db = None
        for (site, depl, recording, start_frame, end_frame), y in tqdm(batch, desc='loading in batch'):
            cur = recording
            
            if cur != prev:
                tqdm.write("loading new recording: ",  recording,  flush=True)
                start_time = time.time()

                # load the file in
                recording_path = Path(self.ds.get_data_path(depl, site)) / recording
                y, samplerate = sf.read(recording_path)      
                
                if samplerate != self.sr:
                    print("!!! foreign sample rate read by soundfile: ", samplerate)
                
                if len(y.shape) > 1:
                    y = np.mean(y, axis=1)
        
                tqdm.write("\rstft ", recording, flush=True)
                Sxx_template, _, _, _ = sound.spectrogram(
                    y, self.sr, window, nperseg, noverlap
                )
                
                tqdm.write("\rconverting to db ", recording,  flush=True)
                S_db = util.power2dB(Sxx_template, db_range)
        
                # S_db = librosa.amplitude_to_db(np.abs(S), ref=np.max)
                elapsed_time = time.time() - start_time
                tqdm.write(f"\rLoaded {recording} in {elapsed_time:.2f} seconds",  flush=True)

            batch_x.append(S_db[:, start_frame:end_frame])
            batch_y.append(y)
            prev = cur
    
        return np.array(batch_x), np.array(batch_y)
