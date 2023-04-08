import sys
import numpy as np
import pandas as pd
import time

from fwl.dataset import Dataset
from fwl.knn import KNN
from fwl.helpers import euclidean_dist, str_solution, get_seed
from typing import Callable

from sklearn.metrics import accuracy_score

ALPHA = 0.8
MEAN = 0.0
VAR = 0.3

######################################################
###################### UTILS #########################
######################################################


def hit_rate(num_hits: int, total: int) -> float:
    return 100 * (num_hits / total)


def red_rate(num_ignored: int, num_features: int) -> float:
    return 100 * (num_ignored / num_features)


def F(x_train, y_train, x_test, y_test, w, clf: KNN) -> tuple[float, float, float]:
    '''
    Calc target function F(w).
    Returns fitness along with hit rate and reduction rate for convenience
    '''
    clf.fit(X=x_train, y=y_train, w=w)

    # Make predictions - Classify
    predictions = clf.predict(examples=x_test)

    # Test predictions - Get the number of correctly classified examples
    num_hits = int(accuracy_score(y_test, predictions, normalize=False))

    num_feats_ignored = len(w[w < 0.1])

    # Calc measurements
    hit_r = hit_rate(num_hits=num_hits, total=y_test.shape[0])
    red_r = red_rate(num_ignored=num_feats_ignored, num_features=x_test.shape[1])
    return ALPHA * hit_r + (1 - ALPHA) * red_r, hit_r, red_r


######################################################
####################### 1-NN #########################
######################################################


def one_kk(x_train: np.ndarray, y_train: np.ndarray) -> np.ndarray:
    # 1-KK original, unweighted classifier
    return np.ones(x_train.shape[1])


######################################################
###################### RELIEF ########################
######################################################


def greedy(x_train: np.ndarray, y_train: np.ndarray) -> np.ndarray:
    '''
    @brief learn weights from a training set using a Greedy approach
    @param x_train training examples
    @param y_train training classes
    @return w weights learned
    '''

    # Initialize W to 0
    w = np.zeros(x_train.shape[1], dtype=np.float32)

    for i in range(x_train.shape[0]):
        dist = euclidean_dist(x_train, x_train[i], np.ones_like(w, dtype=np.float32))

        # get all nearest examples except itself with dist 0 - Leave-one-out
        nearest_examples = np.argsort(dist)
        nearest_examples = nearest_examples[dist[nearest_examples] > 0.0]

        # identify the nearest friend
        friend = enemy = None
        for nn in nearest_examples:
            if y_train[nn] == y_train[i]:
                friend = nn
                break
        # identify the nearest enemy
        for nn in nearest_examples:
            if y_train[nn] != y_train[i]:
                enemy = nn
                break
        w = (
            w
            + np.abs(x_train[i] - x_train[enemy])
            - np.abs(x_train[i] - x_train[friend])
        )

    # Normalize weights
    w_max = np.max(w)

    for idx, w_i in enumerate(w):
        if w_i < 0:
            w[idx] = 0.0
        else:
            w[idx] = w_i / w_max

    return w


#########################################################
################ 5-FOLD CROSS VALIDATION ################
#########################################################


def validate(ds: Dataset, fwl_algo: Callable, seeds: list[int]) -> pd.DataFrame:
    '''
    5-fold cross validation for a dataset
    @param ds: Dataset used to train and validate
    @param fwl_algo: Feature-Weight-Learning algorithm used to fit feature weights
    @return a `pandas.DataFrame` object containing relevant measures about algorithm evaluation on dataset
    '''

    clf = KNN(1)

    '''
    Measures table
    - 4 measures: hit_rate(training), hit_rate(test), reduction_rate, fitness, elapsed_time
    - 5 partitions (folds) of dataset
    '''
    measures = np.array(np.repeat(0, 6 * 5), dtype=np.float32).reshape((6, 5))

    # Either use a general seed for all executions or different seeds per execution
    unique_seed = False
    if len(seeds) == 1:
        unique_seed = True
        np.random.seed(seeds[0])

    for test_part_key in range(1, 6):
        # Initialize seed
        if not unique_seed:
            seed = get_seed(seeds, test_part_key - 1)
            np.random.seed(seed=seed)

        ###### Training stage ######
        # Join 4 partitions for training
        x_train = np.concatenate(
            [
                ds.partitions[i]
                for i in filter(lambda x: x != test_part_key, ds.partitions)
            ]
        )
        y_train = np.concatenate(
            [ds.classes[i] for i in filter(lambda x: x != test_part_key, ds.classes)]
        )

        ### Learn weights
        start = time.monotonic()
        w = fwl_algo(x_train=x_train, y_train=y_train)
        end = time.monotonic()

        ###### Testing stage ######
        test_part = ds.partitions[test_part_key]
        test_class = ds.classes[test_part_key]

        # Take measures for training and test
        _, hit_r_train, _ = F(x_train, y_train, x_train, y_train, w, clf)
        fitness, hit_r_test, red_r = F(x_train, y_train, test_part, test_class, w, clf)

        fwl_elapsed_time = end - start

        measures[test_part_key - 1] = np.array(
            [hit_r_train, hit_r_test, red_r, fitness, fwl_elapsed_time]
        )

    # Calc mean of each statistic
    measures[-1] = np.sum(measures[:-1], axis=0) / (measures.shape[0] - 1)

    # return the df
    rows = np.array(
        [
            'Part. #1',
            'Part. #2',
            'Part. #3',
            'Part. #4',
            'Part. #5',
            'Avg.',
        ]
    )
    cols = np.array(['Train (%)', 'Test (%)', 'Red. (%)', 'Fit.', 'T(s)'])
    df = pd.DataFrame(measures, index=rows, columns=cols)
    return df


######################################################
#################### LOCAL SEARCH ####################
######################################################


def gen_random_solution(num_features: int) -> np.ndarray:
    return np.random.uniform(0, 1, num_features)


def gen_new_neighbour(w: np.ndarray, gene: int) -> np.ndarray:
    z = np.random.normal(MEAN, np.sqrt(VAR))
    new_w = w.copy()
    new_w[gene] = new_w[gene] + z

    # Trunc the feature if necessary
    if new_w[gene] < 0:
        new_w[gene] = 0.0
    elif new_w[gene] > 1:
        new_w[gene] = 1.0

    return new_w


def busqueda_local(x_train: np.ndarray, y_train: np.ndarray) -> np.ndarray:
    '''
    @brief learn weights from a training set using Best-First Local Search
    @param x_train training examples
    @param y_train training classes
    @return w weights learned
    '''

    clf = KNN(k=1)
    eval_sol = lambda x_train, y_train, w, clf: F(
        x_train, y_train, x_train, y_train, w, clf
    )

    # Initial random solution and initial best
    w = gen_random_solution(x_train.shape[1]).astype(np.float32)
    f, _, _ = eval_sol(x_train, y_train, w, clf)

    # Number of F evaluations
    num_evals = 1

    # Number of neighbours generated
    num_gen_neigh = 0

    # First iteration, generate new neighbours
    gen_new_neigh = True

    # Bind to avoid bugs
    genes_to_mutate = np.array([])

    # Repeat until termination criterion
    while num_evals < 15000 and num_gen_neigh < 20 * (x_train.shape[1]):
        if gen_new_neigh:
            # Permute order of genes to mutate
            genes_to_mutate = np.random.permutation(np.arange(x_train.shape[1]))
            gen_new_neigh = False
        else:
            # Generate a new neighbour
            new_neigh = gen_new_neighbour(w, genes_to_mutate[0])
            genes_to_mutate = genes_to_mutate[1:]
            num_gen_neigh += 1

            # Check improvement
            f_new_n, _, _ = eval_sol(x_train, y_train, new_neigh, clf)
            if f_new_n > f:
                f = f_new_n
                w = new_neigh

                num_evals += 1

                # Repeat again for this new solution
                gen_new_neigh = True

        # if there's no more genes that improve, repeat
        if genes_to_mutate.shape[0] == 0:
            gen_new_neigh = True

    return w
