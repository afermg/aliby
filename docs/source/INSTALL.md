# Installation

Tested on: Mac OSX Mojave and Ubuntu 20.04

## Requirements
We strongly recommend installing within a python environment as there are many dependencies that you may not want polluting your regular python environment.
Make sure you are using python 3.

An environment can be created with using the conda package manager:

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

    $ pyenv install 3.8.13
    $ pyenv virtualenv 3.8.13 aliby
    $ pyenv local aliby


## Pipeline installation

### Pip version
Once you have created your local environment, run:

    $ cd aliby
    $ pip install -e ./


### Git version

We use [ poetry ](https://python-poetry.org/docs/#installation) for dependency management.


In case you want to have local versions (usually for development) the main three aliby dependencies you must install them in a specific order:

    $ git clone git@git.ecdf.ed.ac.uk:swain-lab/aliby/aliby.git
    $ git clone git@git.ecdf.ed.ac.uk:swain-lab/aliby/postprocessor.git
    $ git clone git@git.ecdf.ed.ac.uk:swain-lab/aliby/agora.git

    $ cd aliby && poetry install
    $ cd ../postprocessor && poetry install
    $ cd ../agora && poetry install

And that should install all three main dependencies in an editable mode. The same process can be used for [BABY](https://git.ecdf.ed.ac.uk/swain-lab/aliby/baby)
