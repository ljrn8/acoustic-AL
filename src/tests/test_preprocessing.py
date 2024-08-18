"""Unittest to ease developement

TODO
* proper test folder/suite
* PATH

"""

import unittest
from util import *
from config import ANNOTATIONS
from pathlib import Path
import pandas as pd
from preprocessing import SpectrogramSequence
import pickle
import librosa
import matplotlib.pyplot as plt
import soundfile as sf


def dump(o, filename):
    with open(filename, "wb") as f:
        pickle.dump(o, f)


def get(filename):
    with open(filename, "rb") as f:
        return pickle.load(f)


def _view_store_batch(batch):
    batch_X, batch_Y = batch
    for i, (S, y) in enumerate(zip(batch_X, batch_Y)):
        librosa.display.specshow(S)

        # NOTE breaks with overlaps
        # diff = np.diff(y_flat)
        # start_samples = np.where(diff == 1)[0] + 1
        # end_samples = np.where(diff == -1)[0] + 1

        # put dots where there is a class
        y_flat = [any(row) for row in y]
        # assert any(y_flat)
        x_positions = [i for i, value in enumerate(y_flat) if value == 1]
        plt.scatter(x_positions, [400] * len(x_positions), color="red", s=10)

        Path("./objects/seq").mkdir(exist_ok=True)
        np.savetxt(f"./objects/seq/y_flat_{i}.txt", np.array(y_flat))
        np.savetxt(f"./objects/seq/Y_{i}.txt", y)
        plt.savefig(f"./objects/seq/seq_{i}.png")
        # plt.show()


class Tests(unittest.TestCase):

    annotations_df: pd.DataFrame = pd.read_csv(
        ANNOTATIONS / "initial_dataset_7depl_metadata.csv"
    )
    ds: Dataset = Dataset(DATA_ROOT)
    has_annotations = "1_20230316_063000.wav"
    has_annotations_path = ds.get_data_path(1, 1) / has_annotations

    sr = 24_000
    spectrogram_sequence = SpectrogramSequence(annotations_df, ds, sr=sr)

    def test_sequence_init(self):
        pass  # in global

    def test_extract_samplewise_annotations(self):
        rec = self.has_annotations
        group_df = self.annotations_df.query("recording == @rec")

        Y_all = self.spectrogram_sequence._extract_samplewise_annotations(
            group_df, n_frames=self.sr * 60 * 10
        )

        print("for an annotated rec: ")
        for label, index in self.spectrogram_sequence.label_tokens.items():
            print(f"total samples with {label}: {sum(Y_all[:, index])}")

    @unittest.skip
    def test_batch(self):
        ys = np.array([Y for (X, Y) in self.spectrogram_sequence.chunk_info])
        index = [i for i, y in enumerate(ys) if y.flatten().sum() > 0][0]
        print("index with annotation -> ", index)

        batch_index = index // self.spectrogram_sequence.batch_size
        print("batch_index -> ", batch_index)

        ### uncomment if from file ##

        # if Path("./objects/batch.pkl").exists():
        #     batch = get("./objects/batch.pkl")
        # else:
        #     batch = self.spectrogram_sequence.__getitem__(batch_index)
        #     dump(batch, "./objects/batch.pkl")

        batch = self.spectrogram_sequence.__getitem__(batch_index)
        _view_store_batch(batch)

    # dims
    # (512, 467)
    # (467, 4)


if __name__ == "__main__":
    unittest.main()
