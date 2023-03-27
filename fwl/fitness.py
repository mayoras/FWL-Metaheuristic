import numpy as np
from fwl.dataset import Dataset
from fwl.knn import KNN

ALPHA = 0.3


def succ_rate(num_failed: int, total: int) -> float:
    return 100 * ((total - num_failed) / total)


def red_rate(num_ignored: int, num_features: int) -> float:
    return 100 * (num_ignored / num_features)


def fitness(ds: Dataset, clf: KNN) -> float:
    avg = 0
    for i in ds.partitions:
        test_part = ds.partitions[i]
        test_clas = ds.classes[i]

        train_features = []
        test_classes = []
        for j in ds.partitions:
            if j != i:
                train_features.append(ds.partitions[j])
                test_classes.append(ds.classes[j])
        train_features = np.concatenate(train_features)
        train_classes = np.concatenate(test_classes)

        # Fit the classifier
        clf.fit(X=train_features, y=train_classes, w=clf.w_train)

        # Predict the test partition
        predictions = clf.predict(test_part)

        ## Test predictions
        num_failed = 0
        for inp, prediction, label in zip(test_part, predictions, test_clas):
            if prediction != label:
                print(
                    inp, 'has been classified as ', prediction, 'and should be ', label
                )
                num_failed += 1
        num_success = len(test_part) - num_failed

        print('Number of failed classifications:', num_failed)
        print('Number of successful classifications:', num_success)

        ## Calc Fitness
        red_rate = 0.0
        fitness = (
            ALPHA * succ_rate(num_failed=num_failed, total=len(test_part))
            + (1 - ALPHA) * red_rate
        )

        avg += fitness
    avg /= len(ds.partitions)

    return avg
