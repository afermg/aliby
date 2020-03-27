# Installation 

## Requirements
### Access to OMERO 5.2
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

####Disclaimers

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

### Python Requirements
`pyOmeroUpload`: https://github.com/SynthSys/pyOmeroUpload.git

Clone the repository: `git clone https://github.com/SynthSys/pyOmeroUpload.git`
Then install with `pip install pyOmeroUpload`

Note: as is, the `pyOmeroUpload` package ignores the DIC channel and cannot 
read most of the `*log.txt` files that I've tested so we're relying mostly 
on the metadata in the `*Acq.txt` files.

## Local installation

Run the following from outside of the pipline directory to install an editable
version of the pipeline on your local machine. 

It is recommended to install in a virtual/conda environment.

```
pip install -e pipeline-core
```

## References
* [OMERO python bindings](https://docs.openmicroscopy.org/omero/5.4.0/developers/Python.html)
* [Zeroc-ice python](https://pypi.org/project/zeroc-ice/3.6.5/) 

