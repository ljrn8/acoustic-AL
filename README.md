# Active learning: human-in-the-loop strategies to efficiently analyse big acoustic datasets

_Supervised by – Dr Paul Nguyen Hong Duc, Dr Evgeny Sidenko, Prof. Christine Erbe_

_Abstract:_ This paper explores methods of sound event detection and classification for terrestrial
life in long-duration audio, applying and comparing state-of-the-art active learning
methods to minimize labelling efforts for passive acoustic monitoring tasks. We take
the most robust case of discrete signature classification by segmentation against a
vast, diverse dataset from Mauritius and find that most traditional clustering and
diversity-based sampling methods are intractable. Existing deep active learning research does not prioritize efficiency at this scale interrupting the annotation workflow significantly. In order for classification of precise, rare bird calls, we design a
lightweight classification-by-segmentation pipeline and propose a novel method for
information diversity sampling on model embeddings with uncertainty. This allowed
the model to correctly adjust to subtle features that discriminate the target classes
in a large diverse dataset. In this case, the proposed method outperformed baselines
achieving 0.88% of the potential performance accessing just 10% labelled data.
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

