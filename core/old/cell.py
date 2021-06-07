"""
Main data processing functions. The inputs for this functions must be data series.
"""

import numpy as np
from pandas import Series, DataFrames

def bg_sub(data:Series, background:float):
    return data - background

def ratio(data1:Series, data2:Series):
    return data1 / data2

def norm_tps(data:Series, tps:list, fun=np.nanmean):
    return data / fun(data.loc(axis=1)[tps])

def growth_rate(data:Series, alg=None, filt = 'savgol'):
    if alg is None:
        alg='standard'

    

    
