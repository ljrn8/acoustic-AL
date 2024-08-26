# Active learning: human-in-the-loop strategies to efficiently analyse big acoustic datasets

_Supervised by – Dr Paul Nguyen Hong Duc, Dr Evgeny Sidenko, Prof. Christine Erbe_

[insert abstract]


## Usage
! currently not in usable state

set the dataset root under `.env`
```.env
DATA_ROOT="/path/to/wavfiles/" 
```

setup a local environment and install the package code under `scr` 
in editable mode
```
pip install -e .
```

run scripts, import modules ect
```
cd scripts
python train_CRNN_segmentation.py [-d]
```
> proper script arguments and configuration will be added once functional


## Project Structure
```
. 
├── src (all resuable utility functions/classes as a package) 
├── figures  
├── models (model files / checkpoints) 
├── notebooks (all jupyter notebooks) 
├── output (intermediate data and annotations) 
│   ├── annotations 
│   └── correlations 
│   └── intermediate
├── scripts
```
