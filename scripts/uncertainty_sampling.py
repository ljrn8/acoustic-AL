from preprocessing import oversample_minority_classes
import numpy as np; np.random.seed(0)
from config import OUTPUT_DIR


import pickle
import numpy as np

import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from scikeras.wrappers import KerasClassifier
from modAL import ActiveLearner
import pickle
from tensorflow import keras
import keras_cv
from keras import metrics
from sklearn.metrics import f1_score, precision_score, recall_score, average_precision_score

from util import MULTICLASS_LABELS
from sklearn.metrics import average_precision_score

from preprocessing import oversample_minority_classes, build_resnet16
from AL import *
from util import *
from config import * 


X = np.load(INTERMEDIATE / 'logmel_multiclass_noise.npy')
Y = np.load(INTERMEDIATE / 'logmel_labels_multiclass_noise.npy')


working_dir=OUTPUT_DIR / 'AL' / 'Entropy_300_0.3_' 
identity='Entropy_300_0.3_'
budget_cap=0.3
n_queries=20
oversample=True
batch=32
model_file=MODEL_DIR / 'init_trained.keras'


working_dir.mkdir(exist_ok=True)

# split data
init, pool, test = AL_split(X, Y)
initial_X, initial_Y = init
pool_X, pool_Y = pool
test_X, test_Y = test
currently_labelled = len(initial_X)
initial_ds_size = currently_labelled + len(pool_X)

# duplication oversampling
if oversample:
    initial_X, initial_Y = oversample_minority_classes(initial_X, initial_Y)
    pool_X, pool_Y = oversample_minority_classes(pool_X, pool_Y)
    import gc; gc.collect()

# assume init data has already been fitted
import keras
model = keras.saving.load_model(model_file)
classifier = KerasClassifier(model, batch_size=batch, verbose=2, random_state=0, warm_start=True)

# modAL
learner = ActiveLearner(
    estimator=classifier,
    verbose=2,
    # query_strategy=entropy_sampling 
)

classifier.fit(initial_X[0:1], initial_Y[0:1], epochs=1) # complains if i dont fit something

query_size = int(((pool_X.shape[0] + initial_X.shape[0]) * budget_cap) / n_queries)
print(f'query size: {query_size}')
LB_metrics = []

# keep track of labbelled instances
labeled_X, labeled_Y = initial_X, initial_Y

# active learning loop
for idx in tqdm(range(n_queries)):
    print(f'Query no. {idx + 1}/{n_queries}')

    # query for instances
    query_indicies, query_instances = learner.query(
        pool_X, n_instances=query_size) # labeled_X=labeled_X
    
    # train on instances
    learner.teach(
        X=pool_X[query_indicies], y=pool_Y[query_indicies], 
        only_new=True, verbose=2)

    # get evaluation metrics
    print("evaluating ..")
    currently_labelled += query_size
    labelling_budget = currently_labelled / initial_ds_size 
    pred_Y = classifier.predict_proba(test_X, batch_size=batch, verbose=2)
    LB_metrics.append(
        (labelling_budget, evaluation_dict(pred_Y, test_Y)))


     # keep track of labelled instances
    # labeled_X = np.vstack((labeled_X, pool_X[query_indicies]))
    # labeled_Y = np.vstack((labeled_Y, pool_Y[query_indicies]))
    
    # remove queried instance from pool using index method
    mask = np.ones(len(pool_X), dtype=bool)
    mask[query_indicies] = False
    pool_X = pool_X[mask]
    pool_Y = pool_Y[mask]
        
    # store metrics
    with open(working_dir / 'metrics_overwrite_pkl', 'wb') as f:
        pickle.dump(LB_metrics, f)

# save all metrics and diagrams
pred_Y = classifier.predict_proba(test_X, batch_size=batch, verbose=2)
save_metrics(LB_metrics, working_dir, identity, pred_Y, test_Y)

print("reamining instances: ", pool_X.shape, pool_Y.shape)
