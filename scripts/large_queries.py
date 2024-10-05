from preprocessing import oversample_minority_classes
import numpy as np; np.random.seed(0)
from config import OUTPUT_DIR
from util import *
from AL import AL_split, mAP_and_APs, evaluation_dict, AL_resnet16_simulation
import pickle

from config import * 
from util import *
from preprocessing import oversample_minority_classes, build_resnet16
import numpy as np

import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import pickle
from tensorflow import keras
import keras_cv
from keras import Model

from util import MULTICLASS_LABELS
from sklearn.metrics import average_precision_score
from keras.callbacks import EarlyStopping
from sklearn.metrics.pairwise import cosine_similarity, pairwise_distances
from scipy.stats import entropy

X = np.load(INTERMEDIATE / 'logmel_multiclass_noise.npy')
Y = np.load(INTERMEDIATE / 'logmel_labels_multiclass_noise.npy')


def try_train(x, y):
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
    model.fit(
        x=x, y=y, 
        epochs=10, verbose=2, 
        validation_data=(test_X, test_Y),
        batch_size=32,
        callbacks=[earlystopping_cp]
    )    
    return model


def random_sampling(model, pool_X, n_instances):
    i = np.random.choice(range(pool_X.shape[0]), size=n_instances, replace=False)
    return i, X[i] 


def entropy_sampling(model, pool_X, n_instances, X_labeled=None):
    probabilities = model.predict(pool_X, batch_size=32, verbose=2)
    entropy_values = -np.sum(probabilities * np.log(probabilities + 1e-10), axis=1)
    uncertain_indices = np.argsort(entropy_values)[-n_instances:]
    return uncertain_indices, pool_X[uncertain_indices]


def _partial_av_similiarities(flat_X, partitions=30):
    step = len(flat_X) // partitions
    avs = []
    for i in tqdm(range(0, len(flat_X), step)):
        start = i
        stop = min(i + step, len(flat_X)) 
        p = pairwise_distances(flat_X[start:stop, :], flat_X[start:stop, :]) 
        avg_similarities = np.mean(p, axis=1)
        avs.extend(avg_similarities)
    return np.array(avs)
            

def information_diversity(model, pool_X, n_instances=5, alpha=1):
    def compute_entropy(probabilities):
        return -np.sum(probabilities * np.log(probabilities + 1e-10), axis=1)
    
    probs = model.predict(pool_X, verbose=2)
    entropies = compute_entropy(probs)

    flat_X = np.array([x.flatten() for x in pool_X])
    avg_similarities =_partial_av_similiarities(flat_X)
    normalized_similarities =  avg_similarities / avg_similarities.max()
    
    combined_scores = entropies - alpha*normalized_similarities  
    selected_indices = np.argsort(-combined_scores)[:n_instances]
    return selected_indices, pool_X[selected_indices]


def adaptive_information_diversity(model, pool_X, trained_X, n_instances=5, selection_factor=10):
    
    # take a large uncertain selection from unlabeled_X
    indices, selected_X = entropy_sampling(model, pool_X, n_instances=n_instances*selection_factor)
    
    # compute sim with trained and selected
    S = np.vstack((selected_X, trained_X))
    
    # flatten Logmels for similiarities
    S = np.array([s.flatten() for s in S])
    avg_similarities =_partial_av_similiarities(S) # !! TODO try scipy.spatial.distance.pdist
    
    # only look at the similiarities for the selected instances
    n_selected = selected_X.shape[0]
    selected_similarities = avg_similarities[:n_selected]
    
    # normalized_similarities =  selected_similarities / selected_similarities.max() 
    normalized_similarities =  1 / (1 + selected_similarities)
    
    selected_indices = np.argsort(normalized_similarities)[:n_instances] 
    
    # indicies with respect to the original pool
    i = indices[selected_indices]
    return i, pool_X[i]
    
    
def _embeddings_prediction(model, X):
    embs_layer = model.layers[-2]
    embs_model = Model(inputs=model.input, outputs=[embs_layer.output, model.output])
    embs, preds =  embs_model.predict(X)
    return embs, preds      


def _selective_IDiv_embedding(model, pool_X, n_instances=5, selection_factor=10):
    # entropy sampling    
    embeddings, probabilities = _embeddings_prediction(model, pool_X)
    entropy_values = -np.sum(probabilities * np.log(probabilities + 1e-10), axis=1)
    uncertain_indices = np.argsort(entropy_values)[-n_instances:]
    uncertain_embeddings = embeddings[uncertain_indices]
    
    # pairwise dist on embeddings
    p = pairwise_distances(uncertain_embeddings, uncertain_embeddings)
    av_similarity = p.mean(axis=1)
    
    # convert to diversity
    diversities = 1 / (1 + av_similarity)
    selected_indices = np.argsort(-diversities)[:n_instances]
    i = uncertain_indices[selected_indices]
    return i, pool_X[i] 




## --- script ---

working_dir = OUTPUT_DIR / 'AL' / 'EbeddingIDiv_colderstart_fulltrain_noresampling_10Q_0.25'
query_method = _selective_IDiv_embedding
budget_cap = 0.25
n_queries = 10


working_dir.mkdir(exist_ok=True)
keras.utils.set_random_seed(0) # reproducable

# !!! dont have an initial train amount
# split data
init, pool, test = AL_split(X, Y, initial_train_amount=0.03, test_amount=0.25)

initial_X, initial_Y = init
pool_X, pool_Y = pool
test_X, test_Y = test

currently_labelled = len(initial_X)
initial_ds_size = currently_labelled + len(pool_X)
query_size = int(((pool_X.shape[0] + initial_X.shape[0]) * budget_cap) / n_queries)
print(f'query size: {query_size}')

# oversample (should do this here?)
# initial_X, initial_Y = oversample_minority_classes(initial_X, initial_Y)
# pool_X, pool_Y = oversample_minority_classes(pool_X, pool_Y)
# import gc; gc.collect()

trained_X, trained_Y = initial_X, initial_Y
LB_metrics = []

# import keras
# model = keras.saving.load_model(MODEL_DIR / 'init_trained.keras')

# train and evaluate on init dataset
# model = try_train(x=trained_X, y=trained_Y)
# currently_labelled += query_size
# labelling_budget = currently_labelled / initial_ds_size 
# pred_Y = model.predict(test_X, verbose=2)
# LB_metrics.append(
    # (labelling_budget, evaluation_dict(pred_Y, test_Y)))

model = build_resnet16((40, 107, 1)) 
model.compile(
    optimizer='adam',
    loss=keras_cv.losses.FocalLoss(alpha=0.25, gamma=2))

for query in range(1, n_queries+1):
    print(f'Query no. {query}/{n_queries}')
    
    # query_indices, _ =  query_method(model, pool_X, trained_X, n_instances=query_size)
    query_indices, _ =  query_method(model, pool_X, n_instances=query_size)
    
    # track everything being trained on
    trained_X = np.vstack((trained_X, pool_X[query_indices]))
    trained_Y = np.vstack((trained_Y, pool_Y[query_indices])) 
    
    # train 10 epochs on all data
    model = try_train(x=trained_X, y=trained_Y)

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
        

    