# Pipeline core

The core classes and methods for the python microfluidics, microscopy, and analysis pypline pipeline.

## References
* [OMERO python bindings](https://docs.openmicroscopy.org/omero/5.4.0/developers/Python.html)
* [Zeroc-ice python](https://pypi.org/project/zeroc-ice/3.6.5/) 

## Installation 
How to set up access to OMERO from python:
* Install bzip headers : `sudo apt-get install libbz2-dev`
* Install openssl headers version 1.0.2: `sudo apt-get install libssl1.0-dev`
* Install the corresponding openssl (as default is 1.1.1): `conda install openssl==1.0.2k`
* Make sure you are in an environment that uses python 2.7 
* Install zeroc-ice from PyPI, which includes Ice: `pip install zeroc-ice==3.6.0`
* Run `connect_to_omero.py` as a test (TODO TESTS)

Tested on Ubuntu Bionic-Beaver (18.04): the `install.sh` file should run the
above steps if run as root and in an environment with python 2.7

Tested on MacOSX Mojave: Just installing zeroc-ice should be enough (the OSX
defaults include much of the above) as long as you make sure you are 
running in a python 2.7 environment. Not tested on 2.6 

## Disclaimers
TLDR: Most of this stuff is deprecated. 
Using OMERO 5.2.5 means that we need to use deprecated python 2.7 (EOL 2020
.01.01), and we need to use `zeroc-ice` version 3.6 which is also dropped in 
the newer version of OMERO.
The local version of `openssl` (`conda`) needs to fit the headers of 
`libssl-dev` (`apt-get`). 
By default conda will install OpenSSL version 1.1.1 as all the others are no 
longer maintained. 
However, using the headers of verions 1.0.2 means that we have to downgrade 
OpenSSL to version 1.0.2 also. 

It is highly recommended that we upgrade OMERO to 5.6 in order to use Python 3,
in which case it will even be possible to get OMERO.py directly from PyPI 
with easy installation, [omero-py](https://pypi.org/project/omero-py/)

## Requirements
`pyOmeroUpload`: https://github.com/SynthSys/pyOmeroUpload.git

Clone the repository: `git clone https://github.com/SynthSys/pyOmeroUpload.git`
Then install with `pip install pyOmeroUpload`

Note: as is, the `pyOmeroUpload` package ignores the DIC channel and cannot 
read most of the `*log.txt` files that I've tested so we're relying mostly 
on the metadata in the `*Acq.txt` files.

## Raw data access
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
separated into `TimelapseOMERO` and `TimelapseLocal`, but the main 
function of these objects is `get_hypercube()`, which can be called 
directly from the `Experiment` object.

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

