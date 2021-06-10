# Installation 

Tested on: Mac OSX Mojave

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

## Pipeline installation

Once you have created your local environment, run:

    $ cd pipeline-core
    $ pip install -e ./

You will be asked to put in your GitLab credentials for a couple of the packages installed as dependencies.
The `-e` option will install in 'editable' mode: all of the changes that are made to the code in the repository will immediately be reflected in the installation. 
This is very useful to keep up with any changes and branching that we make. 
That way, if you want to keep up with the most recent updates, just run: 

    $ git pull

If you would rather be more in charge of which updates you install and which you don't, remove `-e` from your installation command. 
In this case (or if you run into a dependency error) in order to update your installed version you will have to run: 

    $ cd pipeline-core
    $ git pull
    $ pip install ./

