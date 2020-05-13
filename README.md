# Pipeline core

The core classes and methods for the python microfluidics, microscopy, and 
analysis pipeline.

### Installation
See [INSTALL.md](./INSTALL.md) for installation instructions.

## Development guidelines
In order to separate the python2, python3, and "currently working" versions 
(\#socialdistancing) of the pipeline, please use the branches:
* python2.7: for any development on the 2 version
* python3.6-dev: for any added features for the python3 version
* master: very sparingly and only for changes that need to be made in both
 versions as I will be merging changes from master into the development
 branches frequently
    * Ideally for adding features into any branch, espeically master, create
     a new branch first, then create a pull request (from within Gitlab) before 
     merging it back so we can check each others' code. This is just to make
     sure that we can always use the code that is in the master branch without
     any issues.
 
Branching cheat-sheet:
```git
git branch my_branch # Create a new branch called branch_name from master
git branch my_branch another_branch #Branch from another_branch, not master
git checkout -b my_branch # Create my_branch and switch to it

# Merge changes from master into your branch
git pull #get any remote changes in master
git checkout my_branch
git merge master

# Merge changes from your branch into another branch
git checkout another_branch
git merge my_branch #check the doc for --no-ff option, you might want to use it
```

## Quickstart Documentation
### Setting up a server
For testing and development, the easiest way to set up an OMERO server is by
using Docker images. 
[The software carpentry](https://software-carpentry.org/) and the [Open
 Microscopy Environment](https://www.openmicroscopy.org), have provided
[instructions](https://ome.github.io/training-docker/) to do this.

The `docker-compose.yml` file can be used to create an OMERO server with an
accompanying PostgreSQL database, and an OMERO web server.
It is described in detail 
[here](https://ome.github.io/training-docker/12-dockercompose/).

Our version of the `docker-compose.yml` has been adapted from the above to
use version 5.6 of OMERO.

To start these containers (in background):
```shell script
cd pipeline-core
docker-compose up -d
```
Omit the `-d` to run in foreground.

To stop them, in the same directory, run:
```shell script
docker-compose stop
```

### Raw data access
Raw data access can be found in `core.experiment` and `core.timelapse`, and 
is organised inspired by the Swain Lab MATLAB pipeline.
 
The `Experiment` classes are basically the only ones that really need to be 
accessed by a user. The `ExperimentOMERO` and `ExperimentLocal` classes 
implement the different possible sources of the data. 
If the experiment is saved locally, we expect the organisation to be as in
[this repository](https://github.com/SynthSys/omero_connect_demo/tree/master/test_data)
`Experiment` cannot be instantiated as it is an abstract class, but calling 
`Experiment.from_source()` will instantiate either an `ExperimentOMERO` or an 
`ExperimentLocal` class depending on the arguments, and the differences between
the two are invisible to the user from then on. 

```python
from core.experiment import Experiment
local_expt = Experiment.from_source('path/to/data/directory/')
omero_expt = Experiment.from_source(10421, #Experiment ID on OMERO
                                    'user', #OMERO Username
                                    'password', #OMERO Password
                                    'host.omero.ed.ac.uk', #OMERO host
                                    port=4064 #This is default
                                    )
```
 
Data is organised in each experiment as `Timelapse` classes. These are also
separated into `TimelapseOMERO` and `TimelapseLocal`.
The main function of these objects is to give a direct interface to the raw
data, whatever form it is saved in. 
These objects are sliceable, meaning that data can be accessed like a numpy
array (with some reservations). This can be done directly through the
 `Experiment` object. 

 ```python
bf_1 = expt[0, 0, :, :, :] # First channel, first timepoint, all x,y,z
```
 
Aside from the argument parsing, this is implemented through the
`get_hypercube()` function, which can be called directly from the `Experiment` 
object.

```python
x, y, width, height, z_positions, channels, timepoints = [None]*7 #Get full pos
expt.get_hypercube(x, y, width, height, z_positions, channels,
                      timepoints)
```
To change position (`Timelapse`), one simply needs to set `Experiment
.curent_position` to the desired position name. 

```python
position = expt.positions[0] #This is the default position when expt initalized
expt.current_position = positions
```

### Tiling the raw data

The tiling of raw data is done through a `Tiler` object. 
It takes a raw `Experiment` object as an argument.

```python
from core.segment import Tiler
seg_expt = Tiler(expt)
```

The initialization should take a few seconds, as it needs to align the images
in time. 

#### Get a timelapse for a given trap
From there, you can obtain a timelapse for a single trap as follows:
```python
channels = [0] #Get only the first channel, this is also the default
z = [0, 1, 2, 3, 4] #Get all z-positions
trap_id = 0
tile_size = 117

# Get a timelapse of the trap
# The default trap size is 96 by 96
# The trap is in the center of the image, except for edge cases
# The output has shape (C, T, X, Y, Z), so in this example: (1, T, 96, 96, 5)
timelapse = seg_expt.get_trap_timelapse(trap_id, tile_size=tile_size, 
                                        channels=channels, z=z)
```

This can take several seconds at the moment.
For a speed-up: take fewer z-positions if you can.

If you're not sure what indices to use:
```python
seg_expt.channels # Get a list of channels
channel = 'Brightfield'
ch_id = seg_expt.get_channel_index(channel)

n_traps = seg_expt.n_traps # Get the number of traps 
```

#### Get the traps for a given time point
Alternatively, if you want to get all the traps at a given timepoint:

```python
timepoint = 0
seg_expt.get_traps_timepoints(timepoint, tile_size=96, channels=None, 
                                z=[0,1,2,3,4])
```



