
from tensorflow.keras.utils import Sequence
from sklearn.preprocessing import LabelBinarizer
from util import Dataset
import librosa
from pathlib import Path
from config import *
import numpy as np
import time
from tqdm import tqdm

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
                 batch_size=32, sr=96_000, label_tokens=DEFAULT_TOKENS):
        
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
        

    # def __len__(self):
    #     return self.datalen // self.batch_size
           
    def __getitem__(self, idx):
        batch = self.chunk_info[idx * self.batch_size: (idx + 1) * self.batch_size]
        seen_recordings = set()
        batch_x, batch_y = [], []

        for (site, depl, recording, start_frame, end_frame), y in batch:
            if recording not in seen_recordings:

                print("loading new recording: ",  recording, end='')
                start_time = time.time()

                y, _ = librosa.load(Path(self.ds.get_data_path(depl, site)) / recording, sr=self.sr)
                
                print("\rstft ", recording, end='')
                S = librosa.stft(y)
                
                print("\rconverting to db ", recording, end='')
                S_db = librosa.amplitude_to_db(np.abs(S), ref=np.max)

                elapsed_time = time.time() - start_time
                print(f"\rLoaded {recording} in {elapsed_time:.2f} seconds")
                seen_recordings.add(recording)

            batch_x.append(S_db[:, start_frame:end_frame])
            batch_y.append(y)
    
        return np.array(batch_x), np.array(batch_y)
