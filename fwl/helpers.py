import numpy as np
from collections import Counter


def is_float(s):
    try:
        float(s)
        return True
    except ValueError:
        return False


def euclidean_dist(sample1: np.ndarray, sample2: np.ndarray, w: np.ndarray) -> float:
    return np.sqrt(np.sum(w * ((sample1 - sample2) ** 2)))


def most_common(arr: np.ndarray) -> float:
    return Counter(arr).most_common(1)[0][0]


def are_equal(e1: np.ndarray, e2: np.ndarray) -> np.bool_:
    return (e1 == e2).all()
