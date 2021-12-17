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
# store does not exist
# stores under /cell_info/

# For each position and time-point, run the extractor
# extractor.run() # Raises an error if the store does not exist
# store under /extraction/

# OPTIONAL
# Run post-processing.
total_time = time.perf_counter() - t
print(f"Total time {total_time:.2f}")

# %%
print(
    f"{total_time/ 60:.2f} minutes for {len(expt.positions)} positions at {expt.shape[1]} timepoints:"
)
per_tp_per_pos = total_time / (len(expt.positions) * expt.shape[1])
print(f"{per_tp_per_pos:.2f}s per time point per position")
print(f"{(per_tp_per_pos * 20 * 200)/3600:.2f}h for an average experiment.")

# %%
# TEST RESULTS
# check results
position_test = expt.positions[0]
with h5py.File(expt.root_dir / f"{position_test}{store_name}", "r") as hfile:
    print(hfile.keys())
    for group in hfile:
        print(group)
        print(hfile[group].keys())

from extraction.core.extractor import Extractor
from extraction.core.parameters import Parameters
from extraction.core.functions.defaults import get_params

params = Parameters(**get_params("batgirl_fast"))
ext = Extractor.from_object(params, object=tiler)
tp0 = ext.extract_exp()

expt.close()
