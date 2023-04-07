import argparse
from fwl.dataset import Dataset
import fwl.fwl as fwl

# def greedy(ds: Dataset) -> tuple[np.ndarray, float]:


def main():
    ds = Dataset(dataset, normalized=True)

    # score_diabetes = fwl.validate2(diabetes_ds, fwl.relief)
    # print(score_diabetes)

    df = fwl.validate(ds=ds, fwl_algo=fwl.relief)
    print(df)


if __name__ == '__main__':
    main()
