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
import argparse

log = logging.getLogger(__name__)



parser = argparse.ArgumentParser()
parser.add_argument("--debug", "-d", action="store_true") # TODO set log level
parser.add_argument("--epochs", "-e", default=10)
args = parser.parse_args()

if args.debug:
    log.setLevel(logging.DEBUG)

EPOCHS = int(args.epochs) 
BATCH = 32

now_str = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
log_dir = "logs/fit/" + now_str

metrics = [
    "binary_accuracy",
    "accuracy", #!
    tf.keras.metrics.Precision(),
    tf.keras.metrics.Recall(),
    #    tf.keras.metrics.AUC(name='auc'),
]

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
model.compile(
    optimizer="adam", 
    loss="binary_crossentropy", 
    metrics=metrics
)
log.info(model.summary())
keras.utils.plot_model(model, to_file=FIGURES_DIR / "model.png")


log.info("\npreparing dataset")
seq_train = SpectrogramSequence(is_validation=False, validation_split=0.8, batch_size=BATCH)
seq_validation = SpectrogramSequence(is_validation=True, validation_split=0.8, batch_size=BATCH)

log.info("\n-- test batch --")
batch_X, batch_Y = seq_validation.__getitem__(len(seq_validation))
log.info(f"batch X Y shapes: {batch_X[0].shape, batch_Y[0].shape}")


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


tensorboard_callback = keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

history = model.fit(
    seq_train,
    epochs=EPOCHS,
    batch_size=BATCH,
    validation_data=seq_validation,
    callbacks=[tensorboard_callback, cp_callback],
)

df = pd.DataFrame(history.history)
df.to_csv(log_dir + "hist.csv")
plt.savefig(FIGURES_DIR / "train.png")

model.save(MODEL_DIR / (now_str + '_crnn.keras'))
