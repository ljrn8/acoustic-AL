"""
AL Query methods
"""

import numpy as np
from sklearn.metrics.pairwise import *
from tqdm import tqdm
from keras import Model


def _compute_entropy(probabilities):
    return -np.sum(probabilities * np.log(probabilities + 1e-10), axis=1)


def require(kwargs, arg):
    if arg not in kwargs:
        raise ValueError(f'{arg not in kwargs}')
    return kwargs[arg]


def _partial_av_similiarities(flat_X, partitions=30):
    step = len(flat_X) // partitions
    avs = []
    for i in tqdm(range(0, len(flat_X), step)):
        start = i
        stop = min(i + step, len(flat_X)) 
        distances = pairwise_distances(flat_X[start:stop, :], flat_X[start:stop, :]) 
        similarities = 1 / (1 + distances)
        avg_similarities = np.mean(similarities, axis=1)
        avs.extend(avg_similarities)
    return np.array(avs)


def random_sampling(model, pool_X, n_instances, trained_X=None):
    i = np.random.choice(range(pool_X.shape[0]), size=n_instances, replace=False)
    return i


def entropy_sampling(model, pool_X, n_instances, trained_X=None):
    probabilities = model.predict(pool_X, batch_size=32, verbose=2)
    entropy_values = _compute_entropy(probabilities)
    uncertain_indices = np.argsort(entropy_values)[-n_instances:]
    return uncertain_indices


# def information_diversity(model, pool_X, trained_X, n_instances=5):
#     probs = model.predict(pool_X, verbose=2)
#     entropies = _compute_entropy(probs)
    
#     flat_labeled_X = np.array([x.flatten() for x in trained_X])
#     avg_similarities =_partial_av_similiarities(..., flat_X)
#     pass
    
    
# NOTE: needs to be ran (or skipped?)
def information_density(model, pool_X, n_instances=5, trained_X=None):
    probs = model.predict(pool_X, verbose=2)
    entropies = _compute_entropy(probs)
    flat_X = np.array([x.flatten() for x in pool_X])
    representativeness = avg_similarities =_partial_av_similiarities(flat_X)
    
    combined_scores = entropies*representativeness  
    selected_indices = np.argsort(-combined_scores)[:n_instances]
    return selected_indices

# NOTE: needs to be ran
def adaptive_information_diversity(model, pool_X, trained_X, n_instances=5, selection_factor=10):
    # take a large uncertain selection from unlabeled_X
    indices = entropy_sampling(model, pool_X, n_instances=n_instances*selection_factor)
    selected_X = pool_X[indices] 
    
    # compute sim with trained and selected
    S = np.vstack((selected_X, trained_X))
    
    # flatten Logmels for similiarities
    S = np.array([s.flatten() for s in S])
    avg_similarities =_partial_av_similiarities(S) 
    
    # only look at the similiarities for the selected instances
    n_selected = selected_X.shape[0]
    av_selected_similarities = avg_similarities[:n_selected]
    
    # normalized_similarities =  selected_similarities / selected_similarities.max() 
    diversities =  1 - av_selected_similarities
    selected_indices = np.argsort(-diversities)[:n_instances] 
    
    # indicies with respect to the original pool
    i = indices[selected_indices]
    return i


# NOTE: needs to be ran
def selective_information_diversity(model, pool_X, trained_X, n_instances=5, selection_factor=10):
    probs = model.predict(pool_X, verbose=2)
    entropies = _compute_entropy(probs)

    # initial uncertain selection
    selection_indices = np.argsort(-entropies)[:n_instances*selection_factor]
    selected = pool_X[selection_indices]
    
    # internal diversity
    S_flat = [x.flatten() for x in selected]
    distances = pairwise_distances(S_flat, S_flat)
    similarities = 1 / (1 + distances)
    diversities =  1 - np.mean(similarities, axis=1)
    assert len(S_flat) == len(diversities)
    
    refined_indicies = np.argsort(-diversities)[:n_instances]
    i = selection_indices[refined_indicies]
    return i
    
    

# NOTE: rerun
def embedding_information_diversity(model, pool_X, n_instances=5, selection_factor=10, trained_X=None):
    def _embeddings_prediction(model, X):
        embs_layer = model.layers[-2]
        embs_model = Model(inputs=model.input, outputs=[embs_layer.output, model.output])
        embs, preds =  embs_model.predict(X)
        return embs, preds  
    
    # entropy sampling    
    embeddings, probabilities = _embeddings_prediction(model, pool_X)
    entropy_values = -np.sum(probabilities * np.log(probabilities + 1e-10), axis=1)
    uncertain_indices = np.argsort(-entropy_values)[:n_instances*selection_factor] 
    uncertain_embeddings = embeddings[uncertain_indices]
    
    # pairwise dist on embeddings
    distances = pairwise_distances(uncertain_embeddings, uncertain_embeddings)
    similarities = 1 / (1 + distances)
    
    # convert to diversity
    diversities = 1 - np.mean(similarities, axis=1)
    selected_indices = np.argsort(-diversities)[:n_instances]
    i = uncertain_indices[selected_indices]
    
    print(f"DEBUG: i={i}")
    return i


def LC(model, pool_X, n_instances, trained_X=None):
    predictions = model.predict(pool_X, verbose=2)
    confidence = np.max(predictions, axis=1)
    least_confident_indices = np.argsort(confidence)[:n_instances]
    return least_confident_indices


def selective_coreset(model, pool_X, trained_X, n_instances):
    def euclid_dist(X, Y):
        return pairwise_distances(
            np.array([x.flatten() for x in X]),
            np.array([y.flatten() for y in Y]),
        )

    def k_center_greedy(all_X, selected_X, budget):
        distances = euclid_dist(selected_X, all_X)
        min_distances = distances.min(axis=1)
        i = np.argsort(-min_distances)[:budget]
        return i

    # entropy   
    pred_Y = model.predict(pool_X, verbose=2)
    entropy_values = -np.sum(pred_Y * np.log(pred_Y + 1e-10), axis=1)
    uncertain_indices = np.argsort(-entropy_values)[:n_instances*6]
    uncertain_X = pool_X[uncertain_indices]
    
    # random batch from X
    all_X = np.vstack((pool_X, trained_X))
    size = all_X.shape[0]
    i = np.random.choice(range(size), n_instances*6)
    random_X_batch = all_X[i] 

    refined_i = k_center_greedy(random_X_batch, uncertain_X, n_instances)
    return uncertain_indices[refined_i]


# tests
if __name__ == "__main__":
    from config import *
    from models import build_compile_resnet16
    import keras

    X = np.load(INTERMEDIATE / 'logmel_multiclass_noise.npy')
    Y = np.load(INTERMEDIATE / 'logmel_labels_multiclass_noise.npy')
    # model = build_compile_resnet16()
    model = keras.saving.load_model(MODEL_DIR / 'init_trained.keras')
    
    split = len(X) // 3
    X_train, X_pool = X[:split], X[split:]
    query_size = len(X_train) // 20
    print(f'q size:', query_size)
    
    # test queries
    i = selective_coreset(model, X_pool, trained_X=X_train, n_instances=query_size)
    assert len(i) == query_size
    assert len(i) == len(set(i))
    print(i)
    
    