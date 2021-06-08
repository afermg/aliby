# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.11.2
#   kernelspec:
#     display_name: baby
#     language: python
#     name: baby
# ---

# %%
import h5py
from core.experiment import ExperimentOMERO
from core.pipeline import create_keys
from core.segment import Tiler, from_hdf, TimelapseTiler
from core.baby_client import BabyRunner
from core.cells import ExtractionRunner

import time

# %%
t = time.perf_counter()

expt = ExperimentOMERO(
    18020,  # Experiment ID on OMERO
    "islay.bio.ed.ac.uk",  # OMERO host
    port=4064,  # This is default
    save_dir="./data",
    username="upload",
    password="***REMOVED***",
)
tiler = Tiler(expt)

# TODO pull config out of the metadata
config = {
    "camera": "prime95b",
    "channel": "Brightfield",
    "zoom": "60x",
    "n_stacks": "5z",
    "default_image_size": 96,
}

runner = BabyRunner(tiler, **config)
# extractor = ExtractionRunner(tiler)

# Pipeline

store_name = "store.h5"  # The base name

keys = create_keys(expt)  # Run for full experiment
# For each position in the experiment, create store in expt.run
print(f"Running expt for {keys}")
keys = expt.run(keys, store_name)

# For each position/time-point run the trap location algorithm and then save
# to store
print(f"Running tiler for {keys}")
keys = tiler.run(keys, store_name)  # Raises an error if the store does not
# exist
# stores under /trap_info/

# For each position and timepoint, run the BABY algorithm
run_config = {"with_edgemasks": True, "assign_mothers": True}
runner.run(keys, store_name, verbose=True, **run_config)  # Raises an error if the
# store does    not exist
# stores under /cell_info/

from core.cells import Cells

cells = Cells.from_source("./data/20191026_ss_experiments_01/DO6MS2_003store.h5")
trap = 70
tp = 0
print("Number of masks obtained: ", len(cells.at_time(tp, kind="mask")[trap]))
print(
    "Number of labels obtained: ",
    len(cells["cell_label"][(cells["trap"] == trap) & (cells["timepoint"] == tp)]),
)
