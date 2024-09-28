# Active learning: human-in-the-loop strategies to efficiently analyse big acoustic datasets

_Supervised by – Dr Paul Nguyen Hong Duc, Dr Evgeny Sidenko, Prof. Christine Erbe_

[insert abstract]


## Project Structure
```
. 
├── src (all resuable utility functions/classes as a package) 
├── figures  
├── models (model files/run-logs/checkpoints) 
├── notebooks (all jupyter notebooks, this is where most of the project was done) 
├── output (intermediate data and annotations, most of which are untracked on GitHub) 
│   ├── annotations 
│   └── correlations 
│   └── intermediate
├── scripts (simple reusable scripts)
```


## Development

set the dataset root under `.env`
```.env
DATA_ROOT="/path/to/wavfiles/" 
```

setup a local environment and install the package code under `scr` 
in editable mode
```
pip install -e .
```

