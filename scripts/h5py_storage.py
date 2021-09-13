import itertools
import os
from time import perf_counter

import h5py
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


def linear_vs_folded(f, n_tps=300, n_cells=1000):
    f.clear()
    data = np.ones((n_tps, n_cells, 5, 5))

    timepoints, cells, _, _ = np.indices(data.shape)

    f.create_group('folded')
    f.create_dataset('folded/data', data=data)

    f.create_group('linear')
    f['linear'].create_dataset('data', data=data.flatten())
    f['linear'].create_dataset('cells', data=cells.flatten())
    f['linear'].create_dataset('timepoints', data=timepoints.flatten())
    
    tp = n_tps - 1
    t = perf_counter()
    ix = np.where(f['linear/timepoints'][()] == tp)[0]
    blah1 = f['linear/data'][ix]
    lin_time = perf_counter() - t
    
    t = perf_counter()
    blah2 = f['folded/data'][tp]
    fol_time = perf_counter() - t
    
    assert np.isclose(len(blah1), len(blah2.flatten()))
    return lin_time, fol_time


if __name__ == "__main__":
    lin_times = []
    fol_times = []

    indices = sorted(itertools.product(range(100, 500, 50), range(1000, 3000, 100)))

    with h5py.File('test.h5', 'w') as f:
        for n_cells, n_tps in indices:
            lin, fol = linear_vs_folded(f, n_tps=n_tps, n_cells=n_cells)
            lin_times.append(lin)
            fol_times.append(fol)


    results = pd.DataFrame(indices, columns=['n_tps', 'n_cells'])

    results['linear'] = lin_times
    results['folded'] = fol_times

    g = sns.jointplot(data=results, x='folded', y='linear', hue='n_tps')
    g.savefig('h5py_storage_tps.pdf', bbox_inches='tight', transparent=True)
    
    g = sns.jointplot(data=results, x='folded', y='linear', hue='n_cells')
    g.savefig('h5py_storage_cells.pdf', bbox_inches='tight', transparent=True)

    # Delete the hdf5 file
    os.remove('test.h5')
