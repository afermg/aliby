# Pipeline core

The core classes and methods for the python microfluidics, microscopy, and analysis pypline pipeline.

## References
* [OMERO python bindings](https://docs.openmicroscopy.org/omero/5.4.0/developers/Python.html)
* [Zeroc-ice python](https://pypi.org/project/zeroc-ice/3.6.5/) 

## Installation 
How to set up access to OMERO from python:
* Install openssl headers version 1.0.2: `sudo apt-get install libssl1.0-dev`
* Install the corresponding openssl (as default is 1.1.1): `conda install openssl==1.0.2k`
* Install bzip headers : `sudo apt-get install libbz2-dev`
* Install zeroc-ice from PyPI, which includes Ice: `pip install zeroc-ice==3.6.0`
* Run `connect_to_omero.py` as a test (TODO TESTS)

## Disclaimers
TLDR: Most of this stuff is depcrecated. 
Using OMERO 5.2.5 means that we need to use depreacted python 2.7 (EOL 2020.01.01), 
and we need to use `zeroc-ice` version 3.6 which is also dropped in the newer version of OMERO.
The local version of `openssl` (`conda`) needs to fit the headers of `libssl-dev` (`apt-get`). 
By default conda will install OpenSSL version 1.1.1 as all the others are no longer maintained. 
However, using the headers of verions 1.0.2 means that we have to downgrade OpenSSL to version 1.0.2 also. 

It is highly recommended that we upgrade OMERO to 5.6 in order to use Python 3, in which case it will even 
be possible to get OMERO.py directly from PyPI with easy installation, [omero-py](https://pypi.org/project/omero-py/)

