import sys
import numpy as np
from fwl.helpers import most_common


class KNN:
    def __init__(self, k: int = 1):
        self.k = k

    def fit(self, X: np.ndarray, y: np.ndarray, w: np.ndarray) -> None:
        self.X_train = X
        self.y_train = y
        self.w_train = w

    def predict(self, examples: np.ndarray) -> np.ndarray:
        predicting_training = False
        # Check if we are predicting in training - Leave-one-out
        if examples.shape == self.X_train.shape:
            predicting_training = True

        # Precompute the distances
        dists = self.compute_dists(examples)

        # Leave-one-out (in training). Set diagonal (distance from itself) to INF
        if predicting_training:
            np.fill_diagonal(dists, sys.maxsize)

        # If k=1, then just take the class of example with minimum distance
        if self.k == 1:
            return np.apply_along_axis(
                lambda x: self.y_train[np.argmin(x)], axis=1, arr=dists
            )

        classes: list[float] = []
        for i in range(examples.shape[0]):
            dist = dists[i]

            # get the k-nearest neighbours
            k_idx = np.argsort(dist)[: self.k]

            # if k > 1, the class assigned to the example is the mode of the k classes
            classes.append(most_common(self.y_train[k_idx]))
        return np.array(classes, dtype=np.float32)

    def compute_dists(self, examples):
        # Compute the matrix distance of test examples with training examples.
        # https://sparrow.dev/pairwise-distance-in-numpy/
        dists = np.sqrt(
            np.sum(
                self.w_train
                * (examples[:, np.newaxis, :] - self.X_train[np.newaxis, :, :]) ** 2,
                axis=-1,
            )
        )
        return dists
