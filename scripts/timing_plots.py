## Get Timings

import matplotlib.pyplot as plt

import numpy as np

timings = dict()
# Open the log file
with open('python-pipeline/data/2tozero_Hxts_02/issues.log', 'r') as f:
    # Line by line read
    for line in f.read().splitlines():
        if not line.startswith('DEBUG:root'):
            continue
        words = line.split(':')
        # Only keep lines that include "Timing"
        if 'Timing' in words:
            # Split the last two into key, value
            k,v = words[-2:]
            # Dict[key].append(value)
            if k not in timings: 
                timings[k] = []
            timings[k].append(float(v[:-1]))

for k,v in timings.items():
    if k in ['Segmentation', 'Trap', 'Extraction', 'get_masks', 'save_to_hdf']:
        plt.plot(v, label=k)
plt.legend()

for name in ['MaskIndexing', 'MaskFetching', 'at_time', 'get_masks']:
    y = timings[name]
    x = range(len(y))
    plt.scatter(x, y, label=name)
plt.legend()

import pandas as pd

timings.keys()

import seaborn as sns

plot_data = {x: timings[x] for x in ['Trap', 'Writing', 'Segmentation', 'Extraction']}

import operator

sorted_keys, fixed_Data = zip(*sorted(plot_data.items(), key=operator.itemgetter(1)))

#Set up the graph parameters
sns.set(context='notebook', style='whitegrid')

#Plot the graph
sns.stripplot(data=fixed_data, size=1)
ax = sns.boxplot(data=fixed_data, whis=np.inf, width=.05)
ax.set(xlabel="Stage", ylabel="Time (s)")

plt.xticks(plt.xticks()[0], sorted_keys);

plt.savefig('Timepoint_benchmark.pdf', bbox_inches='tight', transparent=True)
