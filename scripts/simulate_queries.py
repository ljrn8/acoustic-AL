from preprocessing import oversample_minority_classes
import numpy as np; np.random.seed(0)
from config import OUTPUT_DIR
from util import *
from AL import AL_split, mAP_and_APs, evaluation_dict, AL_resnet16_simulation
import pickle

from config import * 
from util import *
from query_methods import *
from preprocessing import oversample_minority_classes, build_resnet16
import numpy as np

import numpy as np
import pickle
from tensorflow import keras
import keras_cv

from keras.callbacks import EarlyStopping

X = np.load(INTERMEDIATE / 'logmel_multiclass_noise.npy')
Y = np.load(INTERMEDIATE / 'logmel_labels_multiclass_noise.npy')


def try_train(x, y, oversample=False):
    model = build_resnet16((40, 107, 1)) 
    model.compile(
        optimizer='adam',
        loss=keras_cv.losses.FocalLoss(alpha=0.25, gamma=2)
    )
    earlystopping_cp = EarlyStopping(
        monitor='val_loss',
        patience=3,
        restore_best_weights=True,
    )
    if oversample:
        x, y = oversample_minority_classes(x, y)
    model.fit(
        x=x, y=y, 
        epochs=10, verbose=2, 
        validation_data=(test_X, test_Y),
        batch_size=32,
        callbacks=[earlystopping_cp]
    )    
    return model


import argparse
parser = argparse.ArgumentParser()
parser.add_argument('-n', '--name')  
parser.add_argument('-q', '--query_method')
parser.add_argument('-nq', '--queries', default=20, type=float) 
parser.add_argument('-c', '--cap', default=0.35, type=float) 
parser.add_argument('-os', '--oversample', action='store_true') 



## --- script ---

# parse args
args = parser.parse_args()
working_dir = OUTPUT_DIR / 'AL' / args.name
budget_cap = float(args.cap)
n_queries = int(args.queries)
query_method = {
    'entropy': entropy_sampling,
    'random': random_sampling,
    'IDen': information_density,
    'Adaptive-IDiv': adaptive_information_diversity,
    'Selective-IDiv': selective_information_diversity,
    'Embedding-IDiv': embedding_information_diversity,
    'LC': LC,
    'S-Coreset': selective_coreset
}[args.query_method]


working_dir.mkdir(exist_ok=True)
keras.utils.set_random_seed(0)

# split data
init, pool, test = AL_split(X, Y, initial_train_amount=0.03, test_amount=0.25)

initial_X, initial_Y = init
pool_X, pool_Y = pool
test_X, test_Y = test

currently_labelled = len(initial_X)
initial_ds_size = currently_labelled + len(pool_X)
query_size = int(((pool_X.shape[0] + initial_X.shape[0]) * budget_cap) / n_queries)
print(f'query size: {query_size}')

trained_X, trained_Y = initial_X, initial_Y
LB_metrics = []

# import keras
# model = keras.saving.load_model(MODEL_DIR / 'init_trained.keras')

# train on initail
model = try_train(x=trained_X, y=trained_Y)

# eval first 
pred_Y = model.predict(test_X, verbose=2)
labelling_budget = currently_labelled / initial_ds_size 
LB_metrics.append(
    (labelling_budget, evaluation_dict(pred_Y, test_Y)))

# debuggin
# model = build_resnet16((40, 107, 1)) 
# model.compile(
#     optimizer='adam',
#     loss=keras_cv.losses.FocalLoss(alpha=0.25, gamma=2)
# )

for query in range(1, n_queries+1):
    print(f'Query no. {query}/{n_queries}')
    
    # query_indices, _ =  query_method(model, pool_X, trained_X, n_instances=query_size)
    query_indices = query_method(model, pool_X, trained_X=trained_X, n_instances=query_size)
    
    # track everything being trained on
    trained_X = np.vstack((trained_X, pool_X[query_indices]))
    trained_Y = np.vstack((trained_Y, pool_Y[query_indices])) 
    
    # train 10 epochs on all data
    model = try_train(x=trained_X, y=trained_Y, oversample=args.oversample)

    # evaluate    
    currently_labelled += query_size
    labelling_budget = currently_labelled / initial_ds_size 
    pred_Y = model.predict(test_X, verbose=2)
    LB_metrics.append(
        (labelling_budget, evaluation_dict(pred_Y, test_Y)))
    
    # remove queried instance from pool using index method
    mask = np.ones(len(pool_X), dtype=bool)
    mask[query_indices] = False
    pool_X = pool_X[mask]
    pool_Y = pool_Y[mask]

    with open(working_dir / 'metrics_overwrite.pkl', 'wb') as f:
        pickle.dump(LB_metrics, f)
        

    