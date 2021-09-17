import itertools
import logging
import os
from time import perf_counter

import h5py
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


def linear_vs_folded(f_lin_name, f_fol_name, n_tps=300, n_cells=1000):
    f_lin = h5py.File(f_lin_name, 'w')
    f_fol = h5py.File(f_fol_name, 'w')
    
    data = np.ones((n_tps, n_cells, 5, 5))

    timepoints, cells, _, _ = np.indices(data.shape)

    # Create the groups with no initial size
    f_lin.create_group('linear')
    f_lin['linear'].create_dataset('data', maxshape=(None,), shape=(0,))
    f_lin['linear'].create_dataset('cells', maxshape=(None,), shape=(0,))
    f_lin['linear'].create_dataset('timepoints', maxshape=(None,), shape=(0,))
    
    f_fol.create_group('folded')
    f_fol.create_dataset('folded/data', maxshape=(None, None, 5, 5), shape=(0,0, 5,5))
    
    # Test writing time, including a resize
    t = perf_counter()
    f_lin['linear/cells'].resize(len(cells.flatten()), axis=0)
    f_lin['linear/cells'][()] = cells.flatten()
    f_lin['linear/timepoints'].resize(len(timepoints.flatten()), axis=0)
    f_lin['linear/timepoints'][()] = timepoints.flatten()
    f_lin['linear/data'].resize(len(data.flatten()), axis=0)
    f_lin['linear/data'][()] = data.flatten()
    logging.debug(f'Write:linear:{n_cells}:{n_tps}:{perf_counter() - t}:seconds')
    
    t = perf_counter()
    f_fol['folded/data'].resize(data.shape)
    f_fol['folded/data'][()] = data
    logging.debug(f'Write:folded:{n_cells}:{n_tps}:{perf_counter() - t}:seconds')
    
    # Test reading time
    tp = n_tps - 1
    t = perf_counter()
    ix = np.where(f_lin['linear/timepoints'][()] == tp)[0]
    blah1 = f_lin['linear/data'][ix]
    logging.debug(f'Read:linear:{n_cells}:{n_tps}:{perf_counter() - t}:seconds')
    
    t = perf_counter()
    blah2 = f_fol['folded/data'][tp]
    logging.debug(f'Read:folded:{n_cells}:{n_tps}:{perf_counter() - t}:seconds')
    
    assert np.isclose(len(blah1), len(blah2.flatten()))
    
    # Close the files
    f_lin.close()
    f_fol.close()
    
    # Check file sizes
    logging.debug(f'File size:linear:{n_cells}:{n_tps}:{os.stat(f_lin_name).st_size}:bytes')
    logging.debug(f'File size:folded:{n_cells}:{n_tps}:{os.stat(f_fol_name).st_size}:bytes')

    # Delete the files
    os.remove(f_lin_name)
    os.remove(f_fol_name)
    return


if __name__ == "__main__":
#     # Set up the logging 
#     print('Running the benchmark.')
#     logging.basicConfig(filename='h5py_storage.log', level=logging.DEBUG)
    
#     try:
#         indices = sorted(itertools.product(range(100, 500, 50), range(1000, 3000, 100)))
#         for n_cells, n_tps in indices:
#             linear_vs_folded('test_lin.h5', 'test_fol.h5', n_tps=n_tps, n_cells=n_cells) # Run the values
#     finally:
#         for fname in ['test_lin.h5', 'test_fol.h5']:
#             if os.path.exists(fname):
#                 os.remove(fname)
                
    print('Creating visualisations')
    # Parse the log file
    results = pd.read_csv('h5py_storage.log', sep=':', 
            header=None, usecols=[1, 2,3,4,5,6,7], names=['origin', 'Operation', 'Type', 'Cells', 'Timepoints', 'Metric', 'Unit'], )
    dtype={'origin': str, 'Operation': str, 'Type': str, 'Cells': int, 'Timepoints': int, 'Metric': float, 'Unit': str}
    results = results[results['origin'] == 'root'].astype(dtype)
    sns.set_theme(style="ticks", palette="pastel")

    print(results.dtypes)
    for name, group in results.groupby('Unit'):
        print(name)
        g = sns.catplot(x='Operation', y='Metric', hue="Type", kind="box", legend=False, data=group);
        # make grouped stripplot
        g = sns.stripplot(x='Operation', y='Metric', hue='Type', jitter=True, dodge=True, marker='o', alpha=0.5, 
                                data=group)
        # how to remove redundant legends in Python
        # Let us first get legend information from the plot object
        handles, labels = g.get_legend_handles_labels()
        # specify just one legend
        l = plt.legend(handles[0:2], labels[0:2])    
        g.set_ylabel(name)
        sns.despine(offset=10, trim=True)
        g.figure.savefig(f'h5py_storage_{name}_benchmark.pdf', bbox_inches='tight', transparent=True)
        
    g2 = sns.relplot(x='Cells', y='Metric', hue='Timepoints', style='Type', data=results, legend='brief', col='Operation', kind='line',
                facet_kws={'sharey': False, 'sharex': True})
    g2.axes_dict['Write'].set_ylabel('Time (seconds)')
    g2.axes_dict['Read'].set_ylabel('Time (seconds)')
    g2.axes_dict['File size'].set_ylabel('File size (bytes)')
    g2.savefig('h5_storage_linearity.pdf', bbox_inches='tight', transparent=True)
    print('Done.')