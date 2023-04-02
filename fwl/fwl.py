import sys
import numpy as np
import pandas as pd
import time

from fwl.dataset import Dataset
from fwl.knn import KNN
from fwl.helpers import euclidean_dist
from typing import Callable

ALPHA = 0.3


def hit_rate(num_failed: int, total: int) -> float:
    return 100 * ((total - num_failed) / total)


def red_rate(num_ignored: int, num_features: int) -> float:
    return 100 * (num_ignored / num_features)


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


def validate(ds: Dataset, fwl_algo: Callable) -> pd.DataFrame:
    '''
    5-fold cross validation for a dataset
    @param ds: Dataset used to train and validate
    @param fwl_algo: Feature-Weight-Learning algorithm used to fit feature weights
    '''
    clf = KNN(1)

    '''
    Measures table
    - 4 measures: succ_rate, miss_rate, fitness, elapsed_time
    - 5 partitions (folds) of dataset
    '''
    # measures = np.array(
    #     [np.zeros(4), np.zeros(4), np.zeros(4), np.zeros(4), np.zeros(4)],
    #     dtype=np.float32,
    # )
    measures = np.array(np.repeat(0, 6 * 4), dtype=np.float32).reshape((6, 4))

    for test_part_key in range(1, 6):
        # Training stage
        x_train = np.concatenate(
            [
                ds.partitions[i]
                for i in filter(lambda x: x != test_part_key, ds.partitions)
            ]
        )
        y_train = np.concatenate(
            [ds.classes[i] for i in filter(lambda x: x != test_part_key, ds.partitions)]
        )

        start = time.time()
        w = fwl_algo(x_train=x_train, y_train=y_train)
        end = time.time()

        # Testing stage
        test_part = ds.partitions[test_part_key]
        test_y = ds.classes[test_part_key]

        # Fit the model
        clf.fit(X=x_train, y=y_train, w=w)

        # Get predictions with  weights
        predictions = clf.predict(test_part)

        # Test predictions
        num_failed = 0
        for _inp, prediction, label in zip(test_part, predictions, test_y):
            if prediction != label:
                # print(
                #     inp, 'has been classified as ', prediction, 'and should be ', label
                # )
                num_failed += 1
        # num_success = len(test_part) - num_failed

        num_feats_ignored = len(w[w < 0.1])

        # Calc measurements
        hit_r = hit_rate(num_failed=num_failed, total=len(test_part))
        reduction_rate = red_rate(
            num_ignored=num_feats_ignored, num_features=test_part.shape[1]
        )
        fitness = ALPHA * hit_r + (1 - ALPHA) * reduction_rate
        fwl_elapsed_time = end - start

        # print(f'Fitness({[num for num in w]}) = {fitness}')

        measures[test_part_key - 1] = np.array(
            [hit_r, reduction_rate, fitness, fwl_elapsed_time]
        )

    # Calc mean of each statistic
    measures[-1] = np.sum(measures[:-1], axis=0) / (measures.shape[0] - 1)

    # return the pandas DataFrame
    rows = np.array(
        [
            'Partición 1',
            'Partición 2',
            'Partición 3',
            'Partición 4',
            'Partición 5',
            'Media',
        ]
    )
    cols = np.array(['%_clas', '%_red', 'Fit.', 'T(s)'])
    df = pd.DataFrame(measures, index=rows, columns=cols)
    return df
