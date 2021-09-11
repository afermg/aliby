# Working with MATLAB files

Working with MATLAB files is possible but very likely to cause issues. 

```python
from core.experiment import Experiment

omero_expt = Experiment.from_source(19310, #Experiment ID on OMERO
                                    'upload', #OMERO Username
                                    '***REMOVED***', #OMERO Password
                                    'islay.bio.ed.ac.uk', #OMERO host
                                    port=4064, #This is default
                                    save_dir='../doc/' #Directory to cache files
                                    )

# Download and save the MATLAB files
omero_expt.cache_annotations(matlab=True)
```

## Create a Tiler using the MATLAB information
Use the suffix to your MATLAB files as the input argument for `matlab`. For
 example , if your cTimelapse files are named 
 `experiment_and_position_namecTimelapse_001.mat` then you should use
 `'cTimelapse_001.mat'` as an argument, as below.
 
```python
from core.segment import Tiler

seg_expt = Tiler(omero_expt, matlab='cTimelapse_001.mat')
channels = [0] #Get only the first channel, this is also the default
z = [0]# 1, 2, 3, 4] #Get all z-positions
trap_id = 0
tile_size = 117


# Get trap images at a given timepoint
seg_expt.get_traps_timepoint(0, tile_size=tile_size, channels=channels, z=z)
```

## Create a Cells object using the MATLAB information
At the moment a `Cells` object can take a `cTimelapse` as input and no other
format. In the future we may create an experiment-wide `Cells` object where
the position is set like in the `Experiment` or `Tiler` objects.

The following will work if you have downloaded the matlab objects as
 decribed above.
 
```python
from core.cells import Cells
# Currently hacky way to get the correct cells object. Note you will have to
# re-create a cells object each time you change position
mat_source = omero_expt.root_dir / (seg_expt.current_position + seg_expt.matlab)
cells = Cells.from_source(mat_source)
```

You can combine the images with the segmentations at one timepoint by doing: 
```python
timepoint = 0
for image, segmentations in zip(seg_expt.get_traps_timepoint(timepoint
), cells.at_time(timepoint)):
    run_analysis(image, segmentations)
```
