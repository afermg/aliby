#!/usr/bin/env python3
# c=CellsHDF.from_source("/home/alan/Documents/sync_docs/PhD/tmp/DO6MS2_003store.h5")
import h5py
from core.cells import Cells
import pandas as pd

# f = h5py.File("/home/alan/Documents/sync_docs/PhD/tmp/DO6MS2_003store.h5")
f = h5py.File(
    "/shared_libs/pipeline-core/scripts/data/20191026_ss_experiments_01/DO6MS2_003store.h5"
)
tracks = f["/extraction/general/None/area"][()]
cell = Cells.from_source("/home/alan/Documents/sync_docs/PhD/tmp/DO6MS2_003store.h5")
from postprocessor.core.processes.picker import Picker, PickerParameters

picker = Picker(cells=cell, parameters=PickerParameters.default())
