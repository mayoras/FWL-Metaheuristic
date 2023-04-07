import numpy as np
from fwl.helpers import euclidean_dist, most_common


class KNN:
    def __init__(self, k: int = 1):
        self.k = k

    def fit(self, X: np.ndarray, y: np.ndarray, w: np.ndarray) -> None:
        self.X_train = X
        self.y_train = y
        self.w_train = w

    # def predict(self, samples: np.ndarray) -> np.ndarray:
    #     classes: list[float] = []
    #     for s in samples:
    #         # get the distances from each training sample to the unclassified sample
    #         distances = []
    #         for x in self.X_train:
    #             # Leave-one-out
    #             if not are_equal(x, s):
    #                 distances.append(euclidean_dist(x, s, self.w_train))

    #         # get k-nearest neighbours
    #         k_idx = np.argsort(distances)[: self.k]

    #         # if k > 1, the class assigned to the example is the mode of the k classes
    #         classes.append(most_common(self.y_train[k_idx]))
    #     return np.array(classes, dtype=np.float32)

    def predict(self, examples: np.ndarray) -> np.ndarray:
        classes: list[float] = []

        predicting_training = False
        # Check if we are predicting in training - Leave-one-out
        if examples.shape == self.X_train.shape and np.allclose(examples, self.X_train):
            predicting_training = True
        for e in examples:
            distances = euclidean_dist(self.X_train, e, self.w_train)

            # get k-nearest neighbours. Leave-one-out (in training) by filtering distances greater than 0 (that is the example itself)
            if predicting_training:
                k_idx = np.argsort(distances)[1 : self.k + 1]
            else:
                k_idx = np.argsort(distances)[: self.k]

            # if k > 1, the class assigned to the example is the mode of the k classes
            classes.append(most_common(self.y_train[k_idx]))
        return np.array(classes, dtype=np.float32)
