"""
Active Learning utilities
"""

from config import * # TODO repalce with direct calls
from util import *
from preprocessing import oversample_minority_classes, build_resnet16
import numpy as np

import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.metrics import precision_recall_curve
from scikeras.wrappers import KerasClassifier
from modAL import ActiveLearner
import pickle

import keras_cv
from keras import metrics


from util import MULTICLASS_LABELS
from sklearn.metrics import average_precision_score


def AL_split(X, Y, test_amount=0.2, initial_train_amount=0.2) -> tuple[tuple]:
    """
    Split the data into (initial_training, pool, test) sets for 
    active learning trials.
    """
    c = int(len(X) * (1 - test_amount))
    X_other, X_test = X[:c], X[c:]
    Y_other, Y_test = Y[:c], Y[c:]

    cc = int(len(X_other) * (1 - initial_train_amount))
    X_pool, X_train = X_other[:cc], X_other[cc:]
    Y_pool, Y_train = Y_other[:cc], Y_other[cc:]
    return (X_train, Y_train), (X_pool, Y_pool), (X_test, Y_test)


def mAP_and_APs(Y_test, Y_pred):
    APs = []
    for i, l in enumerate(MULTICLASS_LABELS):
        AP = average_precision_score(Y_test[:, i], Y_pred[:, i])
        APs.append(AP)

    return np.array(APs).mean(), APs


def save_metrics(LB_metrics: dict, working_dir: Path, identity: str, pred_Y=None, true_Y=None):
    def plot_pr(name, labels, predictions, **kwargs):
        prec, rec, _ = precision_recall_curve(labels, predictions)
        plt.plot(prec, rec, label=name, linewidth=2, **kwargs)
        plt.xlabel('Precision')
        plt.ylabel('Recall')
        plt.xlim([0,1])
        plt.ylim([0,1])
        plt.grid(True)
    
    working_dir.mkdir(exist_ok=True)
    
    # save the metrics here aswell
    with open(working_dir / 'metrics.pkl', 'wb') as f:
        pickle.dump(LB_metrics, f)

    # classwize PR curve
    if pred_Y is not None and true_Y is not None:
        for i, name in enumerate(MULTICLASS_LABELS):
            plot_pr(name, true_Y[:, i], pred_Y[:, i])
        plt.savefig(working_dir / 'PR Curve.png')
        plt.show()

    # AP curves
    x = [lb for lb, _ in LB_metrics]
    for i, l in enumerate(LABELS):
        y = [m[l]['auc_pr'] for _, m in LB_metrics]
        title = f'AP of {l} for {identity}'
        plt.plot(x, y)
        plt.ylim(0, 1)
        plt.title(title)
        plt.savefig(working_dir / f'{title}.png')
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


def AL_resnet16_simulation(query_method, X, Y, working_dir, identity,
                           batch=32, n_queries=200, oversample=True):
    """
    Simulation active learning cycle on 40-bin logmel spectrograms with the Resnet-16 architecture.  
    
    Args:
        n_queries (int): number of queries to perform. The query size is infered from this.
        query_method (function): custom query method in that takes (classifier, X_unlabelled, n_instances)
            and returns (query indecies, query items)
    """
    working_dir.mkdir(exist_ok=True)
    
    # build resnet 16 and  wrap in  scikit learn classifer
    model = build_resnet16((40, 107, 1))
    classifier = KerasClassifier(model, batch_size=batch, verbose=2)
    model.compile(optimizer='adam',
              loss=keras_cv.losses.FocalLoss(alpha=0.25, gamma=2), 
              metrics=[
                metrics.Recall(thresholds=0.5), # NOTE: useless: macro
                metrics.Precision(thresholds=0.5), # NOTE: useless: macro
                metrics.AUC(curve='pr', name='auc_pr') # NOTE: useless: macro
              ])
    
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
        # pred_Y = classifier.predict_proba(test_X, batch_size=batch, verbose=2)
        # LB_metrics.append(
            # (labelling_budget, evaluation_dict(pred_Y, test_Y)))
    
        # store trained on samples 
        trained_X = np.vstack((trained_X, pool_X[query_indicies]))
        trained_Y = np.vstack((trained_Y, pool_Y[query_indicies]))
        
        # remove queried instance from pool
        # pool_X = np.delete(pool_X, query_indicies, axis=0)
        # pool_Y = np.delete(pool_Y, query_indicies, axis=0)

        # remove using index method
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

    print("remaining instances: ", trained_X.shape, trained_Y.shape)
    return classifier, LB_metrics
