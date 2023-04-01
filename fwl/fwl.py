import sys
import numpy as np
from fwl.dataset import Dataset
from fwl.fitness import fitness
from fwl.knn import KNN
from fwl.helpers import euclidean_dist


def are_equal(e1: np.ndarray, e2: np.ndarray) -> np.bool_:
    return (e1 == e2).all()


def relief(x_train: np.ndarray, y_train: np.ndarray) -> np.ndarray:
    '''
    @brief learn weights from a training set using a Greedy approach
    @param x_train training examples
    @param y_train training classes
    @return w weights learned
    '''

    # Initialize W to 0
    w = np.zeros(x_train.shape[1], dtype=np.float32)

    # For each sample in the training set
    for idx, (e, c) in enumerate(zip(x_train, y_train)):
        # Look for the nearest enemy and friend
        enemy = None
        friend = None

        min_dist_enemy = min_dist_friend = sys.maxsize
        for e_p, c_p in zip(x_train, y_train):
            dist = euclidean_dist(e, e_p, np.ones(x_train.shape[1]))

            # Enemy
            if (
                # Not the same example
                not are_equal(e, e_p)
                # Leave-one-out
                and c_p != c
                and dist < min_dist_enemy
            ):
                min_dist_enemy = dist
                enemy = e_p
            # Friend
            elif (
                # Not the same example
                not are_equal(e, e_p)
                # Leave-one-out
                and c_p == c
                and dist < min_dist_friend
            ):
                min_dist_friend = dist
                friend = e_p

        # Update the weights with the component-wise distances from enemy and friend
        w = w + np.abs(e - enemy) - np.abs(e - friend)

    # Normalize weights
    w_max = np.max(w)

    for idx, w_i in enumerate(w):
        if w_i < 0:
            w[idx] = 0
        else:
            w[idx] = w_i / w_max

    return w


def validate(ds: Dataset):
    '''
    5-fold cross validation
    '''
    return
