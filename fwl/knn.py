import numpy as np
from fwl.helpers import euclidean_dist, most_common, are_equal


class KNN:
    def __init__(self, k: int = 1):
        self.k = k

    def fit(self, X: np.ndarray, y: np.ndarray, w: np.ndarray) -> None:
        self.X_train = X
        self.y_train = y
        self.w_train = w

    def predict(self, samples: np.ndarray) -> np.ndarray:
        classes: list[float] = []
        for s in samples:
            # get the distances of each training sample to the unclassified sample
            distances = []
            for x in self.X_train:
                # Leave-one-out
                if not are_equal(x, s):
                    distances.append(euclidean_dist(x, s, self.w_train))

            k_idx = np.argsort(distances)[: self.k]

            classes.append(most_common(self.y_train[k_idx]))
        return np.array(classes, dtype=np.float32)
