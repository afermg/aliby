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

If you are analysing data locally:

    $ pip install aliby

If you are contacting an OMERO server:

    $ pip install aliby[network]

NOTE: Support for OMERO servers in GNU/Linux computers requires building ZeroC-Ice, thus it requires build tools. The versions for Windows and MacOS are provided as Python wheels and thus installation is faster.

### Git version

We use [ poetry ](https://python-poetry.org/docs/#installation) for dependency management.

In case you want to have local version:

    $ git clone git@git.ecdf.ed.ac.uk:swain-lab/aliby/aliby.git
    $ cd aliby && poetry install --all-extras

This will automatically install the [ BABY ](https://git.ecdf.ed.ac.uk/swain-lab/aliby/baby) segmentation software. Support for additional segmentation and tracking algorithms is under development.

### Troubleshooting

Segmentation has been tested on: Mac OSX Mojave, Ubuntu 20.04 and Arch Linux.
Data processing has been tested on all the above and Windows 11.

