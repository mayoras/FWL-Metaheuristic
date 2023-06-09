import argparse
from fwl.dataset import Dataset
import fwl.fwl as fwl

ALGORITHMS = {
    'one-kk': fwl.one_kk,
    'greedy': fwl.greedy,
    'busqueda-local': fwl.busqueda_local,
}


def score(dataset, alg, seed):
    if len(alg) == 1:
        print(alg[0].upper().replace('-', ' '))
        score = fwl.validate(ds=dataset, fwl_algo=ALGORITHMS[alg[0]], seed=seed)
        print(score)
    else:
        for a in ALGORITHMS:
            print(a.upper().replace('-', ' '))
            score = fwl.validate(ds=dataset, fwl_algo=ALGORITHMS[a], seed=seed)
            print(score, '\n')


def main():
    parser = argparse.ArgumentParser(description="MH APC")
    parser.add_argument('--dataset', type=str, default=False, help='choose the dataset')
    parser.add_argument(
        '--algorithm', type=str, default=False, help='choose the algorithm'
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=True,
        help='Seed for the pseudorandom generator',
    )
    parser.add_argument('--list-algo', default=False, action='store_true')

    args = parser.parse_args()

    # List available algorithms
    if args.list_algo:
        print('Available Algorithms:')
        for a in ALGORITHMS:
            print('->', a)
        return

    # Seed
    if args.seed:
        seed = int(args.seed)
    else:
        seed = 0

    # Dataset
    if args.dataset:
        ds_name = args.dataset
        ds = Dataset(ds_name, normalized=True)
        data = [ds]
    else:
        # Use all datasets
        diabetes_ds = Dataset('diabetes', normalized=True)
        ozone_ds = Dataset('ozone-320', normalized=True)
        spectf_ds = Dataset('spectf-heart', normalized=True)

        data = [diabetes_ds, ozone_ds, spectf_ds]

    # Algorithm, and execution
    if args.algorithm:
        alg = [args.algorithm]
    else:
        # Use all algorithms
        alg = [ALGORITHMS[k] for k in ALGORITHMS]

    if len(data) == 1:
        print(f'-------------- {data[0].ds_name} --------------')
        score(data[0], alg, seed)
    else:
        for d in data:
            print(f'-------------- {d.ds_name} --------------')
            score(d, alg, seed)


if __name__ == '__main__':
    main()
