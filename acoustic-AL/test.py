"""Unittest to ease developement

TODO
* proper test folder/suite

"""



import unittest
from util import *
from config import ANNOTATIONS
from pathlib import Path
import pandas as pd



class TestStringMethods(unittest.TestCase):
    
    annotations_df: pd.DataFrame = pd.read_csv(ANNOTATIONS / 'initial_dataset_7depl.csv')
    ds: Dataset = Dataset(DATA_ROOT)
    
    def test_spectrogram_seq(self):
        seq = SpectrogramSequence(self.annotations_df, 
                                  Dataset())

        
        
        
    
    
    

if __name__ == '__main__':
    unittest.main()
