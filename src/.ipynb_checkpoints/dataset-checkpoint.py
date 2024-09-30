"""
Audio dataset interaction.
"""

from config import DATA_ROOT, logging
from pathlib import Path
import os

log = logging.getLogger(__name__)


class WavDataset(dict):                                                                                           
    """ General use dictionary for wav datasets.                                                                  
                                                                                                                  
    Summary:                                                                                                      
        A dictionary with the mapping WavDataset["file_name.wav"] = pathlib.Path("path/to/wav").                  
        Infers the root directory contains unique, long, wav formatted recordings, ignoring all other files       
        and the folder hierarchy.                                                                                 
                                                                                                                  
    """                                                                                                           
    root: str                                                                                                     
    EG = "1_20230316_063000.wav"                                                                                  
                                                                                                                  
    def __init__(self, dataset_root=DATA_ROOT, reject_duplicates=True):                                           
        self.root = dataset_root                                                                                  
        self._parse_wav_files(reject_duplicates)                                                                  
        if len(self) == 0:                                                                                        
            log.warning(f'wav dataset found 0 wav files under {self.root}')                                       
                                                                                                                  
    def _parse_wav_files(self, reject_duplicates: bool):                                                          
        for root, dirs, files in os.walk(self.root):                                                              
            for file in files:                                                                                    
                if file.lower().endswith('.wav'):                                                                 
                    full_path = Path(root) / file                                                                 
                    if file in self and reject_duplicates:                                                        
                        raise RuntimeError(f"duplicate wav name file found:  {full_path}. \
                                Only unique file names are accepted, consider setting reject_duplicates=True")                                                                                         
                    self[file] = full_path                                                                        
                                                                                                                  
                                                                                                                  
    def get_wav_files(self) -> list:                                                                              
        return list(self.keys())                                                                                  
                                                                                                                  
    def get_wav_paths(self) -> list:                                                                              
        return list(self.values())                                                                                
