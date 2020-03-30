# Installation 

Tested on: Mac OSX Mojave

## Requirements
Required packages are specified in `environment.yml`.
An environment can be created with the required packages using the conda 
package manager:

    $ conda create --name <env> --file environment.yml`

Alternatively, you can install the package locally as described below.

## Local installation

Run the following from outside of the pipline directory to install an editable
version of the pipeline on your local machine. 

It is still recommended to install in a conda or virtualenv environments.

```
conda create -n <env> python=3.6
conda activate <env>
pip install -e pipeline-core
```

