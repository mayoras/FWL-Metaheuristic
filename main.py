import time
import numpy as np
import fwl.dataset as dataset
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

    diabetes_ds = dataset.Dataset('diabetes', normalized=True)
    ozone_ds = dataset.Dataset('ozone-320', normalized=True)
    spectf_ds = dataset.Dataset('spectf-heart', normalized=True)

    end = time.time()

    print(f'{end - start} seconds')


if __name__ == '__main__':
    main()
