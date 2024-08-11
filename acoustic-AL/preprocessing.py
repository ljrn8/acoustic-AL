
from tensorflow.keras.utils import Sequence
from sklearn.preprocessing import LabelBinarizer
from maad import sound, util
from util import Dataset, timeit
import librosa
from pathlib import Path
from config import *
import numpy as np
import time
from tqdm import tqdm
import soundfile as sf
import pandas as pd
from scipy.signal import resample_poly


from concurrent.futures import ProcessPoolExecutor, as_completed


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
    
    nperseg = 1024
    noverlap = 512
    window = "hann"
    db_range = 80

    def __init__(self, annotations_df, ds: Dataset, chunk_len_seconds=10, 
                 batch_size=32, sr=96_000, label_tokens=DEFAULT_TOKENS, save_sequence=False):
        
        self.annotations = annotations_df
        self.ds = ds
        self.batch_size = batch_size
        self.sr = sr
        self.chunk_len = int(librosa.time_to_frames(chunk_len_seconds, sr=sr))
        
        self.chunk_info = [] # (site, depl, recording, start_frame, end_frame, y)
        
        # self.label_tokens = LabelBinarizer().fit(self.annotations['label']).classes_
        self.label_tokens = label_tokens
        
        # annotations per recording
        
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
                        label_end_frame = librosa.time_to_frames(row['max_t'] - start_time, sr=sr)
                        
                        # Set labels in the y map for frames covering the event
                        row_index = self.label_tokens[row["label"]]
                        y[label_start_frame:label_end_frame + 1, row_index] = 1   # !!

                self.chunk_info.append(
                    ((site, depl, recording, start_frame, start_frame + self.chunk_len), y)
                )

        # convert instance data to df
        if save_sequence:
            data = [(*chunk[0], chunk[1]) for chunk in self.chunk_info]
            df = pd.DataFrame(data, columns=["site", "depl", "recording", "start_frame", "end_frame", "y"])
            p = INTERMEDIATE / 'instance_data.csv'
            p.parent.mkdir(parents=True, exist_ok=True)
            p.touch()
            df.to_csv(p)


    def __len__(self):
        return len(self.chunk_info) // self.batch_size
    
          
    def _load_and_process_recording(self, recording_info):
        site, depl, recording, start_frame, end_frame = recording_info

        # load the file in
        print("loading new recording: " +  recording, flush=True)
        with timeit():
            start_time = time.time()
            recording_path = self.ds.get_data_path(depl, site) / recording
            s, samplerate = sf.read(recording_path)   
            # s, _ = librosa.load(recording_path, sr=self.sr)
        
        # resample
        print("resampling", flush=True)
        with timeit():
            if samplerate != self.sr:
                # num_samples = int(len(s) * self.sr / samplerate)
                s = resample_poly(s, self.sr, samplerate)
            if len(s.shape) > 1:
                s = np.mean(s, axis=1)

        # stft
        print("\rstft " + recording, flush=True)
        with timeit():
            Sxx_template, _, _, _ = sound.spectrogram(
                s, self.sr, self.window, self.nperseg, self.noverlap
            )
        
        print("\rconverting to db " + recording, flush=True)
        with timeit():
            S_db = util.power2dB(Sxx_template, self.db_range)

        # S_db = librosa.amplitude_to_db(np.abs(S), ref=np.max)
        elapsed_time = time.time() - start_time
        print(f"\rTotal: {elapsed_time:.2f} seconds")
                    
        return S_db


    def __getitem__(self, idx):
        batch = self.chunk_info[idx * self.batch_size: (idx + 1) * self.batch_size]
        batch_x = []
        batch_y = []

        prev= None
        S_db = None
        for x_info, y in batch:
            site, depl, recording, start_frame, end_frame = x_info
            cur = recording
            if cur != prev:
                S_db = self._load_and_process_recording(x_info)
            
            prev = cur
            batch_x.append(S_db[:, start_frame:end_frame])
            batch_y.append(y)

        return np.array(batch_x), np.array(batch_y)
    
    
    
    # TODO ?
    def __getitem_threaded__(self, idx):
        batch = self.chunk_info[idx * self.batch_size: (idx + 1) * self.batch_size]
        batch_x = []
        batch_y = []

        nperseg = 1024
        noverlap = 512
        window = "hann"
        db_range = 80

        with ProcessPoolExecutor() as executor:
            # Submit tasks for each recording in the batch
            future_to_recording = {
                executor.submit(self._load_and_process_recording, (site, depl, recording, start_frame, end_frame), window, nperseg, noverlap, db_range): (site, depl, recording)
                    for (site, depl, recording, start_frame, end_frame), _ in batch
            }

            # Process results as they complete
            for future in tqdm(as_completed(future_to_recording), total=len(future_to_recording), desc='Processing batch'):
                site, depl, recording = future_to_recording[future]
                try:
                    S_batch, y = future.result()
                    batch_x.append(S_batch)
                    batch_y.append(y)
                    
                except Exception as e:
                    tqdm.write(f"Error loading {recording}: {e}", flush=True)

        return np.array(batch_x), np.array(batch_y)