# %%
import csv
import h5py
from core.experiment import ExperimentOMERO
from core.pipeline import create_keys
from core.segment import Tiler, from_hdf, TimelapseTiler
from core.baby_client import BabyRunner
from core.cells import ExtractionRunner

import time

# %%
t = time.perf_counter()

# works
# 19916
expt = ExperimentOMERO(
    19813,  # Experiment ID on OMERO
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

store_name = "store.h5"  # The base name

keys = create_keys(expt, positions=[expt.positions[0]], timepoints=list(range(200)))


# For each position in the experiment, create store in expt.run
print(f"Running expt for {keys}")
keys = expt.run(keys, store_name)

# For each position/time-point run the trap location algorithm and then save
# to store
print(f"Running tiler for {keys}")
import cProfile

tiler_profile = cProfile.Profile()
seg_profile = cProfile.Profile()

tiler_profile.enable()
keys = tiler.run(keys, store_name)  # Raises an error if the store does not
tiler_profile.disable()
# exist
# stores under /trap_info/

# For each position and timepoint, run the BABY algorithm
run_config = {"with_edgemasks": True, "assign_mothers": True}
seg_profile.enable()
runner.run(keys, store_name, verbose=True, **run_config)  # Raises an error if the
seg_profile.disable()

seg_profile.dump_stats("rf_nojob_seg.prof")
tiler_profile.dump_stats("rf_nojob_tiler.prof")

from postprocessor.core.io.base import BridgeH5

b = BridgeH5(
    "/shared_libs/pipeline-core/scripts/data/gluStarv_2_0_2_dual_phl__mig1_msn2_ura7_ura8__wt_00/phl_mig1_001store.h5"
)

print(b.get_npairs(nstepsback=3))

expt.close()


# Data processing and plotting
with open("seg_time.csv") as csvfile:
    reader = csv.reader(csvfile, delimiter=" ")
    rt = np.array([float(row[0]) for row in reader])

ncalls = b.get_npairs_over_time(nstepsback=runner.brain.tracker.cell_tracker.nstepsback)


norm_rt = rt / max(rt)
norm_ncalls = ncalls / max(ncalls)

plt.plot(range(200), norm_rt, label="Time")
plt.plot(range(199), norm_ncalls, label="Number of tracking predictions")
plt.legend()
plt.xlabel("Time point")
plt.ylabel("Normalised value")
plt.title("Segmentation performance. Experiment 19813")
plt.show()
