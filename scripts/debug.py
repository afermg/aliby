from time import perf_counter

import numpy as np
from tqdm import tqdm

from core.experiment import ExperimentOMERO
from core.segment import Tiler
from core.traps import all_tiles

print("Initialising")
raw_expt = ExperimentOMERO(14702, host='islay.bio.ed.ac.uk', save_dir='./data',
                           username='upload', password='***REMOVED***')

tiler = Tiler(raw_expt)

print("Parameter initialisation")
trap_id = 0

channels = [0, 1]
z = list(range(5))
t = None
tile_size = 96

print("Running fast version")
t1 = perf_counter()
## The amount of time it is supposed to take
# Set the defaults (list is mutable)
channels = channels if channels is not None else [0]
z_positions = z if z is not None else [0]
times = t if t is not None else np.arange(raw_expt.shape[1])  # TODO choose sub-set of time points
shape = (len(channels), len(times), tile_size, tile_size, len(z_positions))
# Get trap location for that id:
trap_locations = tiler.trap_locations
zct_tiles, slices, _ = all_tiles(trap_locations, shape, raw_expt, z_positions,
                                 channels, times, [trap_id])

t2 = perf_counter()
print(f"Listing tiles: {t2 - t1}")
##time
#images = raw_expt.current_position.pixels.getTiles(zct_tiles)

# DEFINING A LAZY GET PLANE WITH DASk
from dask import delayed
import dask.array as da

get_tile = delayed(lambda idx: raw_expt.current_position.pixels.getTile(*idx))

def get_lazy_tile(zcttile, shape):
    return da.from_delayed(get_tile(zcttile), shape=shape, dtype=float)

# images = [get_lazy_tile(idx) for idx in zct_tiles]

images = raw_expt.current_position.pixels.getTiles(zct_tiles)
timelapse = np.full(shape, np.nan)
##time
total = len(zct_tiles)
for (z, c, t, tile), (y, x), image in tqdm(zip(zct_tiles, slices, images),
                                    total=total):
    ch = channels.index(c)
    tp = times.tolist().index(t)
    z_pos = z_positions.index(z)
    #get_lazy_tile(( # z, c, # t, tile), # shape=(x[1]-x[0], # y[1]-y[0]))
    timelapse[ch, tp, x[0]:x[1], y[0]:y[1], z_pos] = image
t5 = perf_counter()


# timelapse = np.full(shape, np.nan)
# ##time
# total = len(zct_tiles)
# for (z, c, t, _), (y, x), image in tqdm(zip(zct_tiles, slices, images),
#                                         total=total):
#     ch = channels.index(c)
#     tp = times.tolist().index(t)
#     z_pos = z_positions.index(z)
#     timelapse[ch, tp, x[0]:x[1], y[0]:y[1], z_pos] = image
# t5 = perf_counter()

print(f"Total time: {t5 - t1}")

t1 = perf_counter()
# The amount of time it actually takes
#lprun -u 1 -f  get_trap_timelapse_omero 
timelapse = tiler.get_trap_timelapse(trap_id, tile_size, channels=channels,
                                     z=z_positions)
print(f"Time in pipeline: {perf_counter() - t1}")