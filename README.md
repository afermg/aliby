# ALIBY (Analyser of Live-cell Imaging for Budding Yeast)

[![docs](https://readthedocs.org/projects/aliby/badge/?version=master)](https://aliby.readthedocs.io/en/latest)
[![PyPI version](https://badge.fury.io/py/aliby.svg)](https://badge.fury.io/py/aliby)
[![pipeline](https://git.ecdf.ed.ac.uk/swain-lab/aliby/aliby/badges/master/pipeline.svg?key_text=master)](https://git.ecdf.ed.ac.uk/swain-lab/aliby/aliby/-/pipelines)
[![dev pipeline](https://git.ecdf.ed.ac.uk/swain-lab/aliby/aliby/badges/dev/pipeline.svg?key_text=dev)](https://git.ecdf.ed.ac.uk/swain-lab/aliby/aliby/-/commits/dev)

End-to-end processing of cell microscopy time-lapses. ALIBY automates segmentation, tracking, lineage predictions, post-processing and report production. It leverages the existing Python ecosystem and open-source scientific software available to produce seamless and standardised pipelines.

## Quickstart Documentation

We use (and recommend) [OMERO](https://www.openmicroscopy.org/omero/) to manage our microscopy database, but ALIBY can process both locally-stored experiments and remote ones hosted on a server.

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

### Installation

See our [installation instructions]( https://aliby.readthedocs.io/en/latest/INSTALL.html ) for more details.

### Raw data access

ALIBY's tooling can also be used as an interface to OMERO servers, taking care of fetching data when needed.
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

A `Tiler` object performs trap registration. It may be built in different ways but the simplest one is using an image and a the default parameters set.

```python
from aliby.tile.tiler import Tiler, TilerParameters
with Image(list(image_ids.values())[0], **server_info) as image:
    tiler = Tiler.from_image(image, TilerParameters.default())
    tiler.run_tp(0)
```

The initialisation should take a few seconds, as it needs to align the images
in time.

It fetches the metadata from the Image object, and uses the TilerParameters values (all Processes in aliby depend on an associated Parameters class, which is in essence a dictionary turned into a class.)

#### Get a timelapse for a given trap
```python
fpath = "h5/location"

trap_id = 9
trange = list(range(0, 30))
ncols = 8

riv = remoteImageViewer(fpath)
trap_tps = riv.get_trap_timepoints(trap_id, trange, ncols)
```

This can take several seconds at the moment.
For a speed-up: take fewer z-positions if you can.

#### Get the traps for a given time point
Alternatively, if you want to get all the traps at a given timepoint:

```python
timepoint = 0
seg_expt.get_tiles_timepoints(timepoint, tile_size=96, channels=None,
                                z=[0,1,2,3,4])
```


### Contributing
See [CONTRIBUTING](https://aliby.readthedocs.io/en/latest/INSTALL.html) on how to help out or get involved.
