# Installation

## Requirements
We strongly recommend installing within a python environment as there are many dependencies that you may not want polluting your regular python environment.
Make sure you are using python 3.

An environment can be created using [Anaconda](https://www.anaconda.com/):

    $ conda create --name <env>
    $ conda activate <env>

Which you can deactivate with:

    $ conda deactivate

Or using virtualenv:

    $ python -m virtualenv /path/to/venv/
    $ source /path/to/venv/bin/activate

This will download all of your packages under `/path/to/venv` and then activate it.
Deactivate using

    $ deactivate

You will also need to make sure you have a recent version of pip.
In your local environment, run:

    $ pip install --upgrade pip

Or using [pyenv](https://github.com/pyenv/pyenv) with pyenv-virtualenv:

    $ pyenv install 3.8.14
    $ pyenv virtualenv 3.8.14 aliby
    $ pyenv local aliby


## Pipeline installation

### Pip version
Once you have created and activated your virtual environment, run:

If you are not using an OMERO server setup:

    $ pip install aliby

Otherwise, if you are contacting an OMERO server:

    $ pip install aliby[network]

NOTE: Support for OMERO servers in GNU/Linux computers requires building ZeroC-Ice, thus it requires build tools. The versions for Windows and MacOS are provided as Python wheels and thus installation is faster.

### FAQ
- Installation fails during zeroc-ice compilation (Windows and MacOS).


For Windows, the simplest way to install it is using conda (or mamba). You can install the (OMERO) network components separately:

    $ conda create -n aliby -c conda-forge python=3.8 omero-py
    $ conda activate aliby
    $ cd c:/Users/Public/Repos/aliby
    $ \PATH\TO\POETRY\LOCATION\poetry install

  - MacOS
  Under work (See issue https://github.com/ome/omero-py/issues/317)

### Git version

Install [ poetry ](https://python-poetry.org/docs/#installation) for dependency management.

In case you want to have local version:

    $ git clone git@git.ecdf.ed.ac.uk:swain-lab/aliby/aliby.git
    $ cd aliby && poetry install --all-extras

This will automatically install the [ BABY ](https://git.ecdf.ed.ac.uk/swain-lab/aliby/baby) segmentation software. Support for additional segmentation and tracking algorithms is under development.

## Omero Server

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

### Troubleshooting

Segmentation has been tested on: Mac OSX Mojave, Ubuntu 20.04 and Arch Linux.
Data processing has been tested on all the above and Windows 11.

