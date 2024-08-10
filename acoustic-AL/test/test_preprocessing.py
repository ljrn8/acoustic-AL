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

class TestSequence(unittest.TestCase):
    
    annotations_df: pd.DataFrame = pd.read_csv(ANNOTATIONS / 'initial_dataset_7depl_metadata.csv')
    ds: Dataset = Dataset(DATA_ROOT)
    
    def test_spectrogram_seq(self):
        
        self.seq = seq = SpectrogramSequence(self.annotations_df, self.ds)
        return
        
        p = "./objects/seq.pkl"
        if not Path(p).exists():
            self.seq = seq = SpectrogramSequence(self.annotations_df, self.ds)

            Path(p).touch()
            with open(p, 'wb') as f:
                pickle.dump(seq, f)
                
        else:
            with open(p, 'rb') as f:
                self.seq = seq = pickle.load(f)
            
        
        # batch = seq.__getitem__(0)
        
    

if __name__ == '__main__':
    unittest.main()