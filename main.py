import time
import numpy as np
from fwl.dataset import Dataset
from fwl.knn import KNN
import fwl.fitness
import fwl.helpers as h
from dotenv import load_dotenv

load_dotenv('./.env')


def main():
    # diabetes_dataset = read_dataset('diabetes')
    # ozone_dataset = read_dataset('ozone-320')
    # spectf_dataset = read_dataset('spectf-heart')

    # normalize_dataset(diabetes_dataset[0])
    # normalize_dataset(ozone_dataset[0])
    # normalize_dataset(spectf_dataset[0])

    start = time.time()

    diabetes_ds = Dataset('diabetes', normalized=True)
    ozone_ds = Dataset('ozone-320', normalized=True)
    spectf_ds = Dataset('spectf-heart', normalized=True)

    end = time.time()

    print(f'{end - start} seconds')

    ### Test KNN classifier
    one_nn = KNN(k=1)

    one_nn.fit(
        X=diabetes_ds.partitions[1],
        y=diabetes_ds.classes[1],
        w=np.ones(diabetes_ds.num_features),
    )

    mean_fit = fwl.fitness.fitness(diabetes_ds, one_nn)
    print(mean_fit)


if __name__ == '__main__':
    main()
