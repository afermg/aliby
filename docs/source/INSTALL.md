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
  For local access and processing, follow the same instructions as Linux. Remote access to OMERO servers depends on some issues in one of our depedencies being solved (See issue https://github.com/ome/omero-py/issues/317)

### Git version

Install [ poetry ](https://python-poetry.org/docs/#installation) for dependency management.

In case you want to have local version:

    $ git clone git@gitlab.com/aliby/aliby.git
    $ cd aliby
    
 and then either

    $$ poetry install --all-extras

for everything, including tools to access OMERO servers, or

    $$ poetry install

for a version with only local access, or

    $$ poetry install --with dev

to install with compatible versions of the development tools we use, such as black.

These commands will automatically install the [ BABY ](https://gitlab.com/aliby/baby) segmentation software. Support for additional segmentation and tracking algorithms is under development.

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

### Detailed Windows installation
#### Create environment
Open anaconda powershell as administrator
```shell  script
conda create -n devaliby2 -c conda-forge python=3.8 omero-py
conda activate devaliby2
```

#### Install poetry
    You may have to specify the python executable to get this to work :
```shell script
(Invoke-WebRequest -Uri https://install.python-poetry.org -UseBasicParsing).Content | C:\Users\USERNAME\Anaconda3\envs\devaliby2\python.exe -

```    Also specify full path when running poetry (there must be a way to sort this)

- Clone the repository (Assuming you have ssh properly set up)
```shell script
git clone git@gitlab.com:aliby/aliby.git
cd aliby
poetry install --all-extras
```

You may need to run the full poetry path twice - first time gave an error message, worked second time

```shell script
C:\Users\v1iclar2\AppData\Roaming\Python\Scripts\poetry install --all-extras
```

confirm installation of aliby - python...import aliby - get no error message

#### Access the virtual environment from the IDE (e.g., PyCharm)
New project
In location - navigate to the aliby folder (eg c::/Users/Public/Repos/aliby

- Select the correct python interpreter
click the interpreter name at the bottom right
click add local interpreter
on the left click conda environment
click the 3 dots to the right of the interpreter path and navigate to the python executable from the environment created above (eg C:\Users\v1iclar2\Anaconda3\envs\devaliby2\python.exe)

#### Potential Windows issues
- Sometimes the library pywin32 gives trouble, just install it using pip or conda 
