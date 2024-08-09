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


class TestSequence(unittest.TestCase):
    
    annotations_df: pd.DataFrame = pd.read_csv(ANNOTATIONS / 'initial_dataset_7depl_metadata.csv')
    ds: Dataset = Dataset(DATA_ROOT)
    
    def test_spectrogram_seq(self):
        self.seq = seq = SpectrogramSequence(self.annotations_df, self.ds)
        batch = seq.__getitem__(0)
        
    

if __name__ == '__main__':
    unittest.main()