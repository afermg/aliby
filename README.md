# ALIBYlite (Analyser of Live-cell Imaging for Budding Yeast)

End-to-end processing of cell microscopy time-lapses. ALIBY automates segmentation, tracking, lineage predictions and post-processing.

## Installation

We recommend using [uv](https://github.com/astral-sh/uv) to install aliby

Once uv is installed, we suggest running

```bash
git clone git@github.com:afermg/aliby.git
cd aliby
uv sync
```

## Nix installation (experimental)
This is still under works and it is not guaranteed to work as-is on MacOS, but in Linux running the Nix package manager it should make the setup trivial.

For reproducible environments using [Nix](https://github.com/NixOS/nix) flakes and [envrc](https://github.com/numtide/devshell).

To run a self-contained virtual environment shell session
```bash
nix develop . 
```
For convenience, to start the environment every time you access the project you automatically set the environment
```bash
direnv allow . 
```

