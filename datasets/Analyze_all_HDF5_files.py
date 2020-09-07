import numpy as np
import h5py
import os

PATH = 'ee285f-public/'
for path, dirs, files in os.walk(PATH):
    for filename in files:
        fullpath = os.path.join(path, filename)
        if fullpath.endswith('.hdf5'):
            with h5py.File(fullpath, 'r') as hdf:
                ls = list(hdf.keys())
                print('\n+++ File:', fullpath, (100 - (len(fullpath) + 11)) * '+')
                print('List of datasets in this file:', ls)

                print(100 * '-')

                for name in ls:
                    dataset = hdf.get(name)
                    info = np.array(dataset)
                    print('Shape of', dataset, ': ', info.shape)

                print(100 * '-')

                base_items = list(hdf.items())
                print('Items in base directory:\n', base_items)

                print(100 * '-')

                for item in base_items:
                    for name in ls:
                        group = hdf.get(name)
                        items = list(group.items())
                        items_info = np.array(items)
                        print('Shape of', name, 'items in group', item, ':', items_info.shape)
                    print(100 * '-')