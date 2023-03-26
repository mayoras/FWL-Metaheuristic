import numpy as np


def is_float(s):
    try:
        float(s)
        return True
    except ValueError:
        return False


def euclidean_dist(sample1: np.ndarray, sample2: np.ndarray) -> float:
    return np.sqrt(np.sum((sample1 - sample2) ** 2))
