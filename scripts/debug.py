import numpy as np

from core.experiment import ExperimentOMERO
from core.segment import Tiler
from core.traps import all_tiles

raw_expt = ExperimentOMERO(10863, host='islay.bio.ed.ac.uk', save_dir='./data',
        username='upload', password='***REMOVED***')

tiler = Tiler(raw_expt)

# Parameter initialisation
trap_id =  0
tile_size=96

channels=[0, 1]
z=list(range(5))
t=np.arange(100)
tile_size=96

## The amount of time it is supposed to take
# Set the defaults (list is mutable)
channels = channels if channels is not None else [0]
z_positions = z if z is not None else [0]
times = t if t is not None else np.arange(raw_expt.shape[1])  # TODO choose sub-set of time points
shape = (len(channels), len(times), tile_size, tile_size, len(z_positions))
# Get trap location for that id:
zct_tiles, slices, _ = all_tiles(trap_locations, shape, raw_expt, z_positions, channels, times, [trap_id])

##time
images = raw_expt.current_position.pixels.getTiles(zct_tiles)

##time
images = list(images)

timelapse = np.full(shape, np.nan)
##time
for (z, c, t, _), (y, x), image in zip(zct_tiles, slices, images):
    ch = channels.index(c)
    tp = times.tolist().index(t)
    z_pos = z_positions.index(z)
    timelapse[ch, tp, x[0]:x[1], y[0]:y[1], z_pos] = image


# The amount of time it actually takes
#lprun -u 1 -f  get_trap_timelapse_omero 
timelapse = tiler.get_trap_timelapse(trap_id, tile_size, channels=channels, z=z, t=t)
