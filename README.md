# ALIBYlite (Analyser of Live-cell Imaging for Budding Yeast)

End-to-end processing of cell microscopy time-lapses. ALIBY automates segmentation, tracking, lineage predictions and post-processing.

## Installation

We recommend installing both ALIBY and WELA. 

To begin you should install [miniconda](https://docs.anaconda.com/free/miniconda/index.html) and [poetry](https://python-poetry.org).


Once poetry is installed, we suggest running

```bash
poetry config virtualenvs.create false
 ```

so that only conda creates virtual environments.

Then

- Create and activate an alibylite virtual environment

```bash
conda create -n alibylite python=3.10
conda activate alibylite
 ```

- Git clone alibylite, change to the alibylite directory, and use poetry to install:

```bash 
poetry install
 ```

- Git clone wela, change to the wela directory, and use poetry to install:

```bash 
poetry install
 ```

- Use pip to install your usual Python working environment. For example:

```bash 
pip install ipython seaborn
 ```

- Install omero-py.

For a Mac, use:

```bash 
conda install -c conda-forge zeroc-ice==3.6.5
conda install omero-py
 ```

 For everything else, use:

 ```bash 
poetry install --all-extras
 ```

- You may have an issue with Matlablib crashing.
Use conda to install a different version:

```bash 
conda search -f matplotlib       
 ```

 and, for example, 

 ```bash 
conda install matplotlib=3.8.0 
 ```
