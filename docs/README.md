# ALIBY (Analyser of Live-cell Imaging for Budding Yeast)

[![PyPI version](https://badge.fury.io/py/aliby.svg)](https://badge.fury.io/py/aliby)
[![readthedocs](https://readthedocs.org/projects/aliby/badge/?version=latest)](https://aliby.readthedocs.io/en/latest)
[![pipeline status](https://git.ecdf.ed.ac.uk/swain-lab/aliby/aliby/badges/master/pipeline.svg)](https://git.ecdf.ed.ac.uk/swain-lab/aliby/aliby/-/pipelines)

The core classes and methods for the python microfluidics, microscopy, data analysis and reporting.

### Installation
See [INSTALL.md](./INSTALL.md) for installation instructions.

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

 ```python
from aliby.io.dataset import Dataset
from aliby.io.image import Image

server_info= {
            "host": "host_address",
            "username": "user",
            "password": "xxxxxx"}
expt_id = XXXX
tps = [0, 1] # Subset of positions to get.

with Dataset(expt_id, **server_info) as conn:
    image_ids = conn.get_images()

#To get the first position
with Image(list(image_ids.values())[0], **server_info) as image:
    dimg = image.data
    imgs = dimg[tps, image.metadata["channels"].index("Brightfield"), 2, ...].compute()
    # tps timepoints, Brightfield channel, z=2, all x,y
```
 
### Tiling the raw data

A `Tiler` object performs trap registration. It is built in different ways, the easiest one is using an image and a the default parameters set.

```python
from aliby.tile.tiler import Tiler, TilerParameters
with Image(list(image_ids.values())[0], **server_info) as image:
    tiler = Tiler.from_image(image, TilerParameters.default())
```

The initialisation should take a few seconds, as it needs to align the images
in time. 

It fetches the metadata from the Image object, and uses the TilerParameters values (all Processes in aliby depend on an associated Parameters class, which is in essence a dictionary turned into a class.)

#### Get a timelapse for a given trap
TODO: Update this
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

## TODO
### Tests
* test full pipeline with OMERO experiment (no download.)

