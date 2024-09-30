from config import *
from util import *
from preprocessing import *
import numpy as np

import tensorflow as tf
from tensorflow  import keras
import numpy as np

import librosa 
import matplotlib.pyplot as plt
from tqdm import tqdm
import h5py
import librosa

from sklearn.metrics import f1_score, precision_score, recall_score, average_precision_score
import pickle
from models import build_resnet16
from keras import metrics


def save_metrics(LB_metrics, working_dir, identity, pred_Y=None, true_Y=None):
    def plot_pr(name, labels, predictions, **kwargs):
        prec, rec, _ = precision_recall_curve(labels, predictions)
        plt.plot(prec, rec, label=name, linewidth=2, **kwargs)
        plt.xlabel('Precision')
        plt.ylabel('Recall')
        plt.xlim([0,1])
        plt.ylim([0,1])
        plt.grid(True)
    
    # save the metrics here aswell
    with open(working_dir / 'metrics.pkl', 'rb') as f:
        pickle.dump(LB_metrics, f)

    # classwize PR curve
    if pred_Y and true_Y:
        label_order = DEFAULT_TOKENS.keys()
        for i, name in enumerate(label_order):
            plot_pr(name, test_Y[:, i], pred_Y[:, i])
            plt.savefig(working_dir / f'{name} PR Curve.png')
            plt.show()

    # AP curves
    x = [lb for lb, _ in LB_metrics]
    for i, l in enumerate(LABELS):
        y = [m[l]['auc_pr'] for _, m in LB_metrics]
        title = f'AP of {l} for {identity}'
        plt.plot(x, y)
        plt.ylim(0, 1)
        plt.title(title)
        plt.savefig(working_dir / f'{title}.png'))
        plt.show()

    # mAP curve
    AP = np.zeros(shape=(4, len(x)))
    for i, l in enumerate(LABELS):
        AP[i] = np.array([m[l]['auc_pr'] for _, m in LB_metrics])
    
    title = f'mAP for {identity}'
    plt.plot(x, AP.mean(axis=0))
    plt.ylim(0, 1)
    plt.title(title)
    plt.savefig(working_dir / f'{title}.png')
    plt.show()


def AL_resnet16_simulation(query_method, X, Y, metrics_pkl, 
                           batch=32, n_queries=200, oversample=True) -> dict:
    """
    Simulation active learning cycle on 40-bin logmel spectrograms with the Resnet-16 architecture.  
    
    Args:
        metrics_pkl (str or Path): pickle file to store the incremental metrics dictionary
        n_queries (int): number of queries to perform. The query size is infered from this.
        query_method (function): custom query method in that takes (classifier, X_unlabelled, n_instances)
            and returns (query indecies, query items)
    """
    # build resnet 16 and  wrap in  scikit learn classifer
    model = get_resnet16((40, 107, 1))
    classifier = KerasClassifier(model, batch_size=batch, verbose=2)

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
    
    # initial data
    learner = ActiveLearner(
        estimator=classifier,
        X_training=initial_X, 
        y_training=initial_Y,
        verbose=2,
        query_strategy=query_method 
    )

    query_size = int(pool_X.shape[0] / n_queries)
    LB_metrics = []
    trained_X = initial_X
    trained_Y = initial_Y
    
    # active learning loop
    for idx in tqdm(range(n_queries)):
        print(f'Query no. {idx + 1}/{n_queries}')
    
        # query for instances
        query_indicies, query_instances = learner.query(pool_X, n_instances=query_size)
        
        # train on instances
        learner.teach(
            X=pool_X[query_indicies], y=pool_Y[query_indicies], 
            only_new=True, verbose=2
        )
    
        # get evaluation metrics
        print("evaluating ..")
        currently_labelled += query_size
        labelling_budget = currently_labelled / initial_ds_size 
        pred_Y = classifier.predict_proba(test_X, batch_size=batch, verbose=2)
        LB_metrics.append(
            (labelling_budget, evaluation_dict(pred_Y, test_Y)))
    
        # store trained on samples 
        trained_X = np.vstack((trained_X, pool_X[query_indicies]))
        trained_Y = np.vstack((trained_Y, pool_Y[query_indicies]))
        
        # remove queried instance from pool
        pool_X = np.delete(pool_X, query_indicies, axis=0)
        pool_Y = np.delete(pool_Y, query_indicies, axis=0)

        # store metrics
        with open(metrics_pkl, 'wb') as f:
            pickle.dump(LB_metrics, f)
    
    return LB_metrics
