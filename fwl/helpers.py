import numpy as np
from collections import Counter


def is_float(s):
    try:
        float(s)
        return True
    except ValueError:
        return False


def euclidean_dist(e1, e2, w):
    return np.sqrt(np.sum(w * ((e1 - e2) ** 2), axis=1))


def most_common(arr: np.ndarray) -> float:
    return Counter(arr).most_common(1)[0][0]


def str_solution(w: np.ndarray) -> str:
    return ",".join([str(f) for f in w])
