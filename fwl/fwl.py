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


def relief_fast(x_train: np.ndarray, y_train: np.ndarray) -> np.ndarray:
    '''
    @brief learn weights from a training set using a Greedy approach
    @param x_train training examples
    @param y_train training classes
    @return w weights learned
    '''

    # Initialize W to 0
    w = np.zeros(x_train.shape[1], dtype=np.float32)

    # Precalculate all pair-wise distances:
    # Res: https://sparrow.dev/pairwise-distance-in-numpy/
    dists = np.linalg.norm(
        x_train[:, np.newaxis, :] - x_train[np.newaxis, :, :], axis=-1
    )

    ####### TODO: Maybe do this a little bit faster, maybe with new axes or something like that
    # Get all nearest friends and enemies
    friends = []
    enemies = []
    for i in range(x_train.shape[0]):
        friend = None
        enemy = None
        for j in range(dists[i].shape[0]):
            # Look for nearest friend for ith example
            min_dist_f = sys.maxsize
            min_dist_e = sys.maxsize

            if (
                dists[i, j] > 0
                and dists[i, j] < min_dist_f
                and y_train[i] == y_train[j]
            ):
                min_dist_f = dists[i, j]
                friend = x_train[j]
            if (
                dists[i, j] > 0
                and dists[i, j] < min_dist_e
                and y_train[i] != y_train[j]
            ):
                min_dist_e = dists[i, j]
                enemy = x_train[j]
        friends.append(friend)
        enemies.append(enemy)
    friends = np.array(friends)
    enemies = np.array(enemies)

    # For each sample in the training set
    for i in range(x_train.shape[0]):
        # Look for the nearest enemy and friend
        enemy = enemies[i]
        friend = friends[i]

        # Update the weights with the component-wise distances from enemy and friend
        w = w + np.abs(x_train[i] - enemy) - np.abs(x_train[i] - friend)

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
