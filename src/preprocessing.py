"""
Preprocessing utilities for active learning trails
"""

from tensorflow  import keras
import numpy as np

import librosa 
import matplotlib.pyplot as plt
from tqdm import tqdm
import librosa

import numpy as np
from keras import layers, metrics
from sklearn.metrics import f1_score, precision_score, recall_score, average_precision_score
from models import build_resnet16
from keras import metrics


def get_resnet16(input_shape) -> keras.Model:
    """
    Retreive compiled resnet16 model with the given input
    """
    model = build_resnet16(input_shape=input_shape)
    model.compile(optimizer='adam',
                  loss='binary_crossentropy',
                  metrics=[
                    metrics.Recall(thresholds=0.5),
                    metrics.Precision(thresholds=0.5),
                    metrics.AUC(curve='pr', name='auc_pr')
                  ])
    return model
    

def evaluation_dict(Y_pred, Y_true, threshold=0.5, view=True) -> dict:
    """
    Evaluate the the given model outputs against precision, recall, AP, F1 (classwize)
    """
    metrics = {}
    for label, i in DEFAULT_TOKENS.items():
        Y_pred_binary = (Y_pred[:, i] >= threshold).astype(int)
        Y_true_class = Y_true[:, i]
        metrics[label] = {
            'f1': f1_score(Y_true_class, Y_pred_binary),
            'precision': precision_score(Y_true_class, Y_pred_binary),
            'recall': recall_score(Y_true_class, Y_pred_binary),
            'auc_pr': average_precision_score(Y_true_class, Y_pred[:, i])
        }
        
    if view:
        import pprint
        pp = pprint.PrettyPrinter(indent=4)
        pp.pprint(metrics)
        
    return metrics


def create_resnet50(input_shape = (40, 107, 1)) -> keras.Model:
    """
    Create the resnet model with a multilabel 4-class head
    """
    from tensorflow.keras.applications import ResNet50 
    base_model = ResNet50(weights=None, include_top=False, input_shape=input_shape)
    
    model = keras.models.Sequential()
    model.add(base_model)
    model.add(layers.GlobalAveragePooling2D()) 
    model.add(layers.Dense(256, activation='relu'))  
    model.add(layers.Dense(4, activation='sigmoid'))
    model.compile(optimizer='adam',
                  loss='binary_crossentropy',
                  metrics=[
                    metrics.Recall(thresholds=0.5),
                    metrics.Precision(thresholds=0.5),
                    metrics.AUC(curve='pr', name='auc_pr'),
                    # tfr.keras.metrics.get(key="map", name="metric/map"),
                ])

    return model


def undersample(X, Y, reduce_to=0.5) -> tuple: 
    """
    Utilizes random undersampling for non-annotated, classless noise samples in X, Y.

    Args:
        reduce_to (float): the proportion of the noise samples to maintain.
    """
    total_instances = len(X)
    assert len(Y) == total_instances

    # annotated chunks
    annotated_mask = np.any(Y, axis=1)
    annotated_X = X[annotated_mask]
    annotated_Y = Y[annotated_mask]

    # unnanotated chunks
    unannotated_mask = ~annotated_mask
    unannotated_X = X[unannotated_mask]
    unannotated_Y = Y[unannotated_mask]

    # restart and sample from the noise the class given the proportion
    n_unannotated_to_sample = int(reduce_to * len(unannotated_X))
    if len(unannotated_X) > 0: 
        sampled_indexes = np.random.choice(len(unannotated_X), size=n_unannotated_to_sample, replace=False)
        sampled_X = unannotated_X[sampled_indexes]
        sampled_Y = unannotated_Y[sampled_indexes]
        new_X = np.vstack((annotated_X, sampled_X))
        new_Y = np.vstack((annotated_Y, sampled_Y))
    
    # no noise samples exist
    else:
        new_X = annotated_X
        new_Y = annotated_Y
    
    return new_X, new_Y

    
def oversample_minority_classes(X, Y) -> tuple:
    """
    Complete Oversampling of the minority classes using the duplication method.
    """
    num_classes = Y.shape[1]
    class_counts = np.sum(Y, axis=0)
    max_count = np.max(class_counts)
    new_X = X
    new_Y = Y
    for class_index in range(num_classes): # 0 1 2 3
        class_indices = np.where(Y[:, class_index] == 1)[0] # locations of nr
        num_samples_needed = max_count - len(class_indices)
        if num_samples_needed > 0:
            sampled_indices = np.random.choice(class_indices, num_samples_needed, replace=True)
            sampled_X = X[sampled_indices]
            sampled_Y = Y[sampled_indices]
            
            new_X = np.vstack((new_X, sampled_X))
            new_Y = np.vstack((new_Y, sampled_Y))
            
    return new_X, new_Y


def AL_split(X, Y, test_amount=0.2, initial_train_amount=0.2) -> tuple[tuple]:
    """
    Split the data into (initial_training, pool, test) sets for 
    active learning trials.
    """
    c = int(len(X) * (1 - test_amount))
    X_other, X_test = X[:c], X[c:]
    Y_other, Y_test = Y[:c], Y[c:]

    cc = int(len(X_other) * (1 - initial_train_amount))
    X_pool, X_train = X_other[:cc], X_other[cc:]
    Y_pool, Y_train = Y_other[:cc], Y_other[cc:]
    return (X_train, Y_train), (X_pool, Y_pool), (X_test, Y_test)

    
def train(model, X, Y, model_dir, reduce_empty_class=0.8, stopping_patience=5, 
          stopping_moniter='val_loss', epochs=10, **kwargs):
    """
    Resuable utility for training keras models with checkpoints and preprocessing

    Args:
        model_dir (string or Path): directory to store logs/history/checkpoints
        reduce_empty_class (float): undersampling proportion for the noise class
    """
    
    from tensorflow.keras.callbacks import TensorBoard, ModelCheckpoint, EarlyStopping
    model_dir.mkdir(exist_ok=True)

    # tensorboard
    log_dir = model_dir / "logs" / "fit"
    log_dir.mkdir(parents=True, exist_ok=True)
    tensorboard_callback = TensorBoard(log_dir=log_dir)

    # checkpoints
    checkpoint_path = model_dir / "training"
    Path(checkpoint_path).mkdir(exist_ok=True)  
    cp_callback = keras.callbacks.ModelCheckpoint(
        checkpoint_path / 'checkpoint.weights.h5', 
        save_weights_only=True,
        verbose=1, 
    )

    # early stopping
    earlystopping_cp = EarlyStopping(
        monitor=stopping_moniter,
        patience=stopping_patience,
        restore_best_weights=True,
    )
    
    histories = []
    for epoch in range(epochs): 
        print(f"--- Epoch {epoch + 1} ---")
        resampled_X, resampled_Y = undersample(X, Y, reduce_to=reduce_empty_class)
        resampled_X, resampled_Y = oversample_minority_classes(resampled_X, resampled_Y)
        history = model.fit(x=resampled_X, y=resampled_Y, 
                  **kwargs, 
                  epochs=1, 
                  verbose=2,
                    callbacks=[tensorboard_callback, cp_callback, earlystopping_cp])
        
        histories.append(history.history)
        
    df = pd.DataFrame(history.history)
    df.to_csv(model_dir / "history.csv")
    model.save(model_dir / 'model.keras')
    return histories


def _get_label(rec, start_frame, end_frame, samples_f) -> np.ndarray:
    seconds_required = {
        'fast_trill_6khz': 0.6,
        'nr_syllable_3khz': 0.2, 
        'triangle_3khz': 0.5, # !!
        'upsweep_500hz': 0.25,
    }
    start_sample, end_sample = librosa.frames_to_samples([
        start_frame, end_frame])
    Y_samples = np.array(samples_f[rec]['Y'][:, start_sample:end_sample])
    label_vec = np.zeros(4, dtype=bool)
    for label, idx in DEFAULT_TOKENS.items():
        samples_required = librosa.time_to_samples(
            seconds_required[label], sr=22_000)
        if Y_samples[idx, :].sum() >= samples_required:
            label_vec[idx] = 1

    return label_vec


def _generate_chunks(chunk_indexes, logmel_f) -> tuple:
    X = np.array([logmel_f[rec]['40mels'][:, int(s):int(e)] 
                  for rec, s, e in tqdm(chunk_indexes)])
    Y = np.array([_get_label(rec, int(s), int(e)) 
                  for rec, s, e in tqdm(chunk_indexes, desc='labels')])
    return X, Y
