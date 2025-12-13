# ALIBY: End-to-end processing for high throughput microscopy

This framework orchestrates a multitude of different tools to process imaging data.

NOTE: This past year ALIBY went through heavy refactoring. The documentation is under (re)construction.

By modality:
- Time series microscopy
- Fluorescence microscopy (e.g., Cell Painting)

By procedure
- Segmentation (e.g., [Cellpose](https://github.com/MouseLand/cellpose)) + Feature (e.g., [cp_measure](https://github.com/afermg/cp_measure), engineered features)
- Deep Learning models (e.g., DinoV3, OpenPhenom)

By backend
- Within the same environment (Cellpose -> cp_measure)
- Communicating with other environments (via [Nahual](https://github.com/afermg/nahual))

## Installation

We recommend using [uv](https://github.com/astral-sh/uv) to install aliby

Once uv is installed, we suggest running

```bash
git clone git@github.com:afermg/aliby.git
cd aliby
uv sync --all-extras
```

## Nix installation
For reproducible environments using [Nix](https://github.com/NixOS/nix) flakes and [envrc](https://github.com/numtide/devshell).

To run a self-contained virtual environment shell session
```bash
nix develop . 
```
For convenience, to start the environment every time you access the project you automatically set the environment
```bash
direnv allow . 
```

