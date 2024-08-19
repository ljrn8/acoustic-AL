

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

from util import WavDataset
from preprocessing import SpectrogramSequence
from config import *

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import datetime
from sklearn.model_selection import train_test_split
import argparse


parser = argparse.ArgumentParser()
parser.add_argument("--debug", "-d", action="store_true")
args = parser.parse_args()


# NOTE ALL time is spend in IO
# TODO cmdline args

batch = 32
epochs = 30

debug_df_cap = 100

metrics = [
    "binary_accuracy",
    tf.keras.metrics.Precision(),
    tf.keras.metrics.Recall(),
    #    tf.keras.metrics.AUC(name='auc'),
]

ds = WavDataset(DATA_ROOT)
annotations_df = pd.read_csv(ANNOTATIONS / "initial_dataset_7depl_metadata.csv")

# has_annotations = "1_20230316_063000.wav"
# has_annotations_path = ds.get_data_path(1, 1) / has_annotations


def CRNN(input_shape, num_classes, n_filters):
    freq_len, time_len, _ = input_shape

    model = keras.Sequential()
    model.add(layers.Input(shape=input_shape))

    # CNN layers
    for filters in n_filters:
        model.add(
            layers.Conv2D(
                filters, (3, 3), activation="relu", padding="same", strides=(2, 1)
            )
        )
        model.add(layers.MaxPooling2D((2, 1)))  # frequency pooling only

    model.add(layers.Flatten())
    model.add(
        layers.Reshape((time_len, -1))
    )  # input matrix for RNN shape=(time_frames, new CNN features)

    # RNN layers
    model.add(layers.LSTM(128, return_sequences=True))
    model.add(layers.Dense(num_classes, activation="sigmoid"))

    return model


## ---- script ----

model = CRNN(input_shape=(512, 428, 1), num_classes=4, n_filters=[32, 64, 128, 256])
model.compile(optimizer="adam", loss="binary_crossentropy", metrics=metrics)
log.info(model.summary())
keras.utils.plot_model(model, to_file=FIGURES_DIR / "model.png")

# group and shuffle ds
grouped = annotations_df.groupby("recording")
shuffled_groups = list(grouped)
np.random.shuffle(shuffled_groups)
shuffled_df = pd.concat([df for _, df in shuffled_groups], ignore_index=True)
shuffled_df.reset_index(drop=True, inplace=True)

if args.debug:
    shuffled_df = shuffled_df.iloc[:debug_df_cap]

# test train split
train_df, test_df = train_test_split(shuffled_df, test_size=0.2, shuffle=False)
train_df, validation_df = train_test_split(train_df, test_size=0.2, shuffle=False)

log.info("\n val, test, train recording counts: ")
for i in (validation_df, test_df, train_df):
    log.info(i.shape)


log.info("\npreparing dataset")
sr = 22_000
train_sequence = SpectrogramSequence(annotations_df=train_df, sr=sr, batch_size=batch)
test_sequence = SpectrogramSequence(annotations_df=test_df, sr=sr, batch_size=batch)
validation_sequence = SpectrogramSequence(
    annotations_df=validation_df, sr=sr, batch_size=batch
)

log.info("\nval, train, test chunk counts:")
for s in (train_sequence, test_sequence, validation_sequence):
    log.info(len(s.chunk_info))

log.info("\n-- test batch --")
batch_X, batch_Y = train_sequence.__getitem__(0)
log.info("batch X Y shapes:", batch_X[0].shape, batch_Y[0].shape)


# early_stopping_cb = tf.keras.callbacks.EarlyStopping(
#     monitor='val_prc',
#     verbose=1,
#     patience=30,
#     mode='max',
#     restore_best_weights=True
# )

checkpoint_path = MODEL_DIR / "training_1" / "crnn.ckpt"
Path(checkpoint_path).parent.mkdir(exist_ok=True)
cp_callback = keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_path, save_weights_only=True, verbose=1
)

now_str = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
log_dir = "logs/fit/" + now_str
tensorboard_callback = keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

history = model.fit(
    train_sequence,
    epochs=epochs,
    validation_data=validation_sequence,
    callbacks=[tensorboard_callback, cp_callback],
)

df = pd.DataFrame(history.history)
df.to_csv(log_dir + "hist.csv")
# df[['prc', 'val_prc', 'recall', 'val_recall']].plot()
plt.savefig(FIGURES_DIR / "train.png")


# 1 batch =~ 60MB in memory uncompressed
