# Active learning: human-in-the-loop strategies to efficiently analyse big acoustic datasets

_Supervised by – Dr Paul Nguyen Hong Duc, Dr Evgeny Sidenko, Prof. Christine Erbe_

[insert abstract]


## Build Dependacies
```
pip install setuptools
```

## Usage
! currently not in usable state

set the dataset root in /.env
```.env
DATA_ROOT="/path/to/wavfiles/" 
```

setup a local environment
```
python -m venv env

# (mac and Linux)
source env/bin/activate 

# (Windows)
env\Scripts\activate 

# or any prefered environment manager
```

install the package code under /scr
```
pip install -e .
```

run scripts ect
```
cd scripts
python train_CNN_segmentation.py [-d]
```
> proper script arguments and configuration will be added once functional


## Project Structure
```
. 
├── acoustic-AL (all resuable utility functions/classes) 
├── figures  
├── models (model files / checkpoints) 
├── notebooks (all jupyter notebooks) 
├── output (intermediate data and annotations) 
│   ├── annotations 
│   └── correlations 
├── scripts
```
