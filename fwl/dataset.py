import os
import re
import numpy as np

from fwl.helpers import is_float


class Dataset:
    '''
    Dataset class:
    - ds_name
    - partitions = List[partitions]
    - labels = List[labels]
    - data = Merge(partitions)
    - labels = List[strings]
    '''

    def __init__(self, ds_name, normalized=False):
        '''
        - ds_name: dataset's unique name
        - normalized: Whether or not to normalize the dataset. Defaults to False
        '''

        DATA_DIR_PATH = os.environ.get("DATA_DIR_PATH")

        # Dataset name
        self.ds_name = ds_name

        # {1: ndarray, 2: ndarray, ...}
        self.partitions = {}

        # {1: ndarray, 2: ndarray, ...}
        self.classes = {}

        # Iterate over dir path
        dir = os.fsencode(DATA_DIR_PATH)
        for file in os.listdir(dir):
            filename = os.fsdecode(file)

            # Just iterate over a specific dataset
            if ds_name not in filename:
                continue

            # features and classes read per file
            features, classes = [], []

            filepath = os.path.join(DATA_DIR_PATH, filename)
            with open(filepath, 'r') as f:
                # print(f.name)
                part_num = int(re.findall('_([1-5]{1}).arff', f.name.split('/')[-1])[0])
                for line in f:
                    if len(line) == 0:
                        continue
                    # if 'class' in line.lower() and line[0] != '%':
                    #     c = [m.strip() for m in re.findall('{(.*)}', line)]
                    #     continue
                    if line[0] in ('@', '%', ' ', '\n'):
                        continue
                    line.split(',')

                    features.append([float(f) for f in line.split(',')[:-1]])
                    classes.append(self.ctof(line.split(',')[-1]))

                self.partitions[part_num] = np.array(features, dtype=np.float64)
                self.classes[part_num] = np.array(classes)
        if normalized:
            self.normalize()

    def normalize(self):
        # we need to concatenate all partitions
        # in order to normalize an entire dataset
        data = np.concatenate(list(self.partitions.values()), axis=0)

        # take min and max from data
        max_values, min_values = np.max(data, axis=0), np.min(data, axis=0)

        # normalized every column of each partition
        for p in self.partitions:
            self.partitions[p] = (self.partitions[p] - min_values) / (
                max_values - min_values
            )

    def ctof(self, c):
        '''
        String to float function.

        if c is not numerical, ad-hoc solution for diabetes:
            - True for 'tested_positive'
            - False for 'tested_negative'
        '''
        if is_float(c):
            return float(c)
        else:
            return True if 'positive' in c else False