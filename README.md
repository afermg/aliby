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

- Git clone alibylite, change to the alibylite directory with the poetry.lock file, and use poetry to install:

```bash 
poetry install
 ```

- Git clone wela, change to the wela directory with the poetry.lock file, and use poetry to install:

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

- You may have an issue with Matplotlib crashing.
Use conda to install a different version:

```bash 
conda search -f matplotlib       
 ```

 and, for example, 

 ```bash 
conda install matplotlib=3.8.0 
 ```

 - On an M1 Mac, these commands proved helpful

 ```bash 
conda install openblas
conda uninstall numpy
conda install numpy    
 ```

 and reinstalling tensorflow

 ```bash 
python -m pip install tensorflow-macos==2.9.0 tensorflow-metal==0.5.0 --force-reinstall
```

## Nix installation (experimental)
This is still under works and it is not guaranteed to work as-is on MacOS, but in Linux running the Nix package manager it should make the setup trivial.

For reproducible environments using [Nix](https://github.com/NixOS/nix) flakes and [envrc](https://github.com/numtide/devshell).

To run a self-contained virtual environment shell session
```bash
nix develop . --impure
```
For convenience, to start the environment every time you access the project you automatically set the environment
```bash
direnv allow . 
```

### IDE Integration
#### Doom Emacs
Activate the `direnv` module (under `:tools`) on [Doom](https://github.com/doomemacs/doomemacs)'s `init.el`. Then add this to your `config.el`.
```elisp
(require 'envrc)
(add-hook! (add-hook 'change-major-mode-after-body-hook 'envrc-mode))
```
That should ensure that Emacs finds the activated environment when running `(run-python)` from a Python file inside the environment.
