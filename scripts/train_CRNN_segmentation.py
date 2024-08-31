import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

from util import WavDataset
from preprocessing import SpectrogramSequence
from config import *
from models import CRNN, CRNN_flat

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import datetime
import argparse

log = logging.getLogger(__name__)

# !!! make sure to set a high window (upsweeps)

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



## ---- script ----

model = CRNN_flat(input_shape=(512, 428, 1), n_filters=[32, 64, 128, 256])
# model = CRNN_flat(input_shape=(512, 127, 1), n_filters=[32, 64, 128, 256])

model.compile(
    optimizer="adam", 
    loss="binary_crossentropy", 
    metrics=metrics
)
log.info(model.summary())
keras.utils.plot_model(model, to_file=FIGURES_DIR / "model.png")


log.info("\npreparing dataset")
seq_train = SpectrogramSequence(is_validation=False, 
                                validation_split=0.8, batch_size=BATCH,
                                # chunk_length_seconds=3, chunk_overlap_seconds=2,
                                     flat_labels=True
                                )

seq_validation = SpectrogramSequence(is_validation=True, 
                                     validation_split=0.8, batch_size=BATCH,
                                    #  chunk_length_seconds=3, chunk_overlap_seconds=2,
                                     flat_labels=True)

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


# !! NOTE delete later 
seq_train.chunk_info = seq_train.chunk_info[:7000]
seq_validation.chunk_info = seq_validation.chunk_info[:1000]

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
