import numpy as np
from collections import Counter


def is_float(s):
    try:
        float(s)
        return True
    except ValueError:
        return False


def euclidean_dist(e1: np.ndarray, e2: np.ndarray, w: np.ndarray) -> np.ndarray | float:
    return np.sqrt(np.sum(w * ((e1 - e2) ** 2), axis=1))


def most_common(arr: np.ndarray) -> float:
    return Counter(arr).most_common(1)[0][0]


# TODO: dump this thing
def are_equal(e1: np.ndarray, e2: np.ndarray) -> np.bool_:
    return (e1 == e2).all()


def get_seed(seeds: list[int], num_exec: int) -> int:
    return seeds[num_exec]


def str_solution(w: np.ndarray) -> str:
    return ",".join([str(f) for f in w])
