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

def dump(o, filename):
    with open(filename, 'wb') as f:
        pickle.dump(o, f)        

def get(o, filename):
    with open(filename, 'rb') as f:
        return pickle.load(f)      

class TestSequence(unittest.TestCase):
    
    annotations_df: pd.DataFrame = pd.read_csv(ANNOTATIONS / 'initial_dataset_7depl_metadata.csv')
    ds: Dataset = Dataset(DATA_ROOT)
    
    def test_spectrogram_seq(self):

        # take_pickle = False

        # if take_pickle:    
        #     p = "./objects/seq.pkl"
        #     if not Path(p).exists():
        #         Path(p).touch()
        #         seq = SpectrogramSequence(self.annotations_df, self.ds, sr=24_000)
        #         with open(p, 'wb') as f:
        #             pickle.dump(seq, f)
        #     else:
        #         with open(p, 'rb') as f:
        #             self.seq = seq = pickle.load(f)

        # else:
            
        # seq = SpectrogramSequence(self.annotations_df, self.ds, sr=24_000)
        
        
        with open("./objects/seq.pkl", 'rb') as f:
            self.seq = seq = pickle.load(f)

        batch = seq.__getitem__(0)
        dump(batch, "./objects/batch.pkl")

        batch = get("./objects/batch.pkl")

        # look at it        
        import librosa
        import  matplotlib.pyplot as plt
        
        S, y = batch[0]
        
        librosa.specshow(S)
        
        y_flat = [any(row) for row in y]
        
        # NOTE breaks with overlaps
        diff = np.diff(y_flat)
        starts = np.where(diff == 1)[0] + 1  
        ends = np.where(diff == -1)[0] + 1 
        
        plt.vlines(starts, label='start')
        plt.vlines(ends, label='end')
        plt.savefig("./objects/seq.png")
        plt.show()
    

if __name__ == '__main__':
    unittest.main()