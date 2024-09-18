import tensorflow as tf
from tensorflow  import keras
import tensorflow_hub as hub
import numpy as np

import librosa 
from dataset import WavDataset
import matplotlib.pyplot as plt
from tqdm import tqdm
from config import *
import h5py
import soundfile as sf
from scipy.signal import resample_poly


SR = 16_000
OUTPUT = INTERMEDIATE / 'embeddings.hdf5'
OVERLAP_THRESH =  0.47 # < yamnet hop 
DEFAULT_TOKENS = {  
    "fast_trill_6khz": 0,
    "nr_syllable_3khz": 1,
    "triangle_3khz": 2,
    "upsweep_500hz": 3,
}

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


def get_label_timestep(rec, annotated_recordings, annotations_df, n_samples):
    labelled_timesteps = np.zeros(shape=(4, n_samples), dtype=bool)    
    if rec in annotated_recordings:
        rec_df = annotations_df[annotations_df.recording == rec]
        
        for label, label_index in DEFAULT_TOKENS.items():
            labelwize_annotations = rec_df[rec_df.label == label]
            start_times, end_times = (
                np.array(labelwize_annotations["min_t"].astype(float)), 
                np.array(labelwize_annotations["max_t"].astype(float))
            )
            label_start_samples = librosa.time_to_samples(start_times, sr=SR)
            label_end_samples = librosa.time_to_samples(end_times,sr=SR)
            
            for start, end in zip(label_start_samples, label_end_samples):
                labelled_timesteps[label_index, start:end] = 1
    
    return labelled_timesteps


def main():
    
    print(f'outputing to {OUTPUT}')
    
    all_recs = np.load(ANNOTATIONS / 'manual_annotations' / 'all_annotated_recordings_filtered.npy', allow_pickle=True)
    annotations = pd.read_csv(ANNOTATIONS / 'manual_annotations' / 'initial_manual_annotations.csv')
    annotated_recordings = annotations.recording.unique()
    
    print(f'n# annotated recs: ', len(annotated_recordings))
    print(f'n# total \'annotated\' recs: ', len(all_recs))
    print(f'n# of annotations: ', len(annotations))
    
    
    print('loading Yamnet ..')
    yamnet_url = 'https://tfhub.dev/google/yamnet/1'
    yamnet_layer = hub.KerasLayer(yamnet_url, input_shape=(None,), dtype=tf.float32, trainable=False)
    if input("continue? ").lower() in ['n', 'no']:
        exit()

    ds_f = h5py.File(OUTPUT, 'w')

    ds = WavDataset()
    for rec in tqdm(all_recs):

        # samples
        rec_path = ds[rec]
        s, given_sr = sf.read(rec_path)
        s = resample_poly(s, SR, given_sr) 
        if len(s.shape) > 1:
            s = np.mean(s, axis=1)
        
        # labels
        labelled_timesteps = get_label_timestep(rec, annotated_recordings, annotations, len(s))

        # embeddings and embedding labels
        _, embedds, _ = yamnet_layer(s)
        label_frames = compute_frame_labels(labelled_timesteps, OVERLAP_THRESH)
        label_frames = label_frames[:, :-1]

        # store in dataset
        group = ds_f.create_group(rec)
        group.create_dataset("X", data=np.array(embedds), dtype=np.float32)
        group.create_dataset("Y", data=np.array(label_frames).T, dtype=bool)
        group.attrs['classwize_labeled_frame_counts'] = [
            sum(label_frames[index, :]) for label, index in DEFAULT_TOKENS.items()
        ]
        group.attrs['n_frames'] = embedds.shape[1]
        
     
if __name__ == "__main__":
    main()


    
