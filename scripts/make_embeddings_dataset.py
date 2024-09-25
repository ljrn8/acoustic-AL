import tensorflow as tf
from tensorflow  import keras
import tensorflow_hub as hub
import numpy as np

import librosa 
import matplotlib.pyplot as plt
from tqdm import tqdm
from config import *
import h5py
import soundfile as sf
from scipy.signal import resample_poly
from util import DEFAULT_TOKENS
import librosa

SR = 16_000
OUTPUT = INTERMEDIATE / 'embeddings_20p.hdf5'
INPUT = INTERMEDIATE / '22sr_samples.hdf5'
OVERLAP_THRESH =  0.20 # fit in every possible label into a frame (<20.17s)


def compute_frame_labels(label_tensor, threshold):
    frame_length, step_size = int(SR * 0.96), int(SR * 0.48)
    n_labels, total_samples = label_tensor.shape
    n_frames = total_samples // step_size 
    frame_labels = np.zeros((n_labels, n_frames), dtype=int)
    for i in range(n_frames):
        start = i * step_size
        end = start + frame_length
        frame = label_tensor[:, start:end]
        
        frame_label = (np.mean(frame, axis=1) >= threshold).astype(int)
        frame_labels[:, i] = frame_label

    return frame_labels

def get_label_timestep(rec, annotated_recordings, annotations_df, n_samples, sr=SR):
    labelled_timesteps = np.zeros(shape=(4, n_samples), dtype=bool)    
    if rec in annotated_recordings:
        rec_df = annotations_df[annotations_df.recording == rec]
        
        for label, label_index in DEFAULT_TOKENS.items():
            labelwize_annotations = rec_df[rec_df.label == label]
            start_times, end_times = (
                np.array(labelwize_annotations["min_t"].astype(float)), 
                np.array(labelwize_annotations["max_t"].astype(float))
            )
            label_start_samples = librosa.time_to_samples(start_times, sr=sr)
            label_end_samples = librosa.time_to_samples(end_times, sr=sr)
            
            for start, end in zip(label_start_samples, label_end_samples):
                labelled_timesteps[label_index, start:end] = 1
    
    return labelled_timesteps
     
     
     
def main():
    
    print(f'outputing to {OUTPUT}')
    
    print('loading Yamnet ..')
    yamnet_url = 'https://tfhub.dev/google/yamnet/1'
    yamnet_layer = hub.KerasLayer(yamnet_url, input_shape=(None,), dtype=tf.float32, trainable=False)
    if input("continue? ").lower() in ['n', 'no']:
        exit()

    new_ds = h5py.File(OUTPUT, 'w')
    old_ds = h5py.File(INPUT, 'r')

    all_recs = np.load(ANNOTATIONS / 'manual_annotations' / 'all_annotated_recordings_filtered.npy', allow_pickle=True)
    annotations = pd.read_csv(ANNOTATIONS / 'manual_annotations' / 'initial_manual_annotations.csv')
    annotated_recordings = annotations.recording.unique()
    
    print(f'n# annotated recs: ', len(annotated_recordings))
    print(f'n# total \'annotated\' recs: ', len(all_recs))
    print(f'n# of annotations: ', len(annotations))
    
    
    for rec in tqdm(old_ds):
        
        # get recording samples and labeled from old
        # Y = old_ds[rec]['Y']  
        samples = old_ds[rec]['X']

        # resample to 16
        samples_16 = librosa.resample(np.array(samples), orig_sr=22_000, target_sr=SR)
        
        # get labels in 16
        Y_16 = get_label_timestep(rec, annotated_recordings, annotations, n_samples=len(samples_16))
        
        group = new_ds.create_group(rec)  

        # apply yamnet embeddings to samples
        _, embedds, _ = yamnet_layer(samples_16)
        group.create_dataset("X", data=embedds, dtype=np.float32) 

        # convert labels to frames        
        label_frames = compute_frame_labels(Y_16, OVERLAP_THRESH)
        group.create_dataset("Y", data=label_frames, dtype=bool) 

        group.attrs['shapes'] = (embedds.shape, label_frames.shape)
        
        

     
if __name__ == "__main__":
    main()

