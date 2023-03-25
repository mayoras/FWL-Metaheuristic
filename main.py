import time
import numpy as np
import fwl.dataset as dataset
from dotenv import load_dotenv

load_dotenv('./.env')


def read_arff(filepath):
    features = []
    classes = []

    with open(filepath, 'r') as f:
        for line in f:
            if len(line) == 0:
                continue
            if line[0] in ('@', '%', ' ', '\n'):
                continue
            line.split(',')
            features.append(line.split(',')[:-1])
            classes.append(line.split(',')[-1])

    return np.array(features, dtype=np.float32), np.array(classes)


def read_dataset(dataset_name):
    feat1, target1 = read_arff(f'data/inst/{dataset_name}_1.arff')
    feat2, target2 = read_arff(f'data/inst/{dataset_name}_2.arff')
    feat3, target3 = read_arff(f'data/inst/{dataset_name}_3.arff')
    feat4, target4 = read_arff(f'data/inst/{dataset_name}_4.arff')
    feat5, target5 = read_arff(f'data/inst/{dataset_name}_5.arff')

    return np.concatenate([feat1, feat2, feat3, feat4, feat5], axis=0), np.concatenate(
        [target1, target2, target3, target4, target5]
    )


def normalize_dataset(features):
    min_values, max_values = np.max(features, axis=0), np.min(features, axis=0)
    features = (features - min_values) / (max_values - min_values)


def main():
    # diabetes_dataset = read_dataset('diabetes')
    # ozone_dataset = read_dataset('ozone-320')
    # spectf_dataset = read_dataset('spectf-heart')

    # normalize_dataset(diabetes_dataset[0])
    # normalize_dataset(ozone_dataset[0])
    # normalize_dataset(spectf_dataset[0])

    start = time.time()

    diabetes_ds = dataset.Dataset('diabetes', normalized=True)
    ozone_ds = dataset.Dataset('ozone-320', normalized=True)
    spectf_ds = dataset.Dataset('spectf-heart', normalized=True)

    end = time.time()

    print(f'{end - start} seconds')


if __name__ == '__main__':
    main()
