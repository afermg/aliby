"""
Functions to load and reshape examples for extraction development

The basic format for data will be pair data/masks pairs. Data will be
assumed to be a single slide, given that this reduction is expected
to happen beforehand.

The most basic functions were copied from Swain Lab's baby module,
specifically baby/io.py
"""

from importlib_resources import files
import json
import re
from itertools import groupby
from pathlib import Path
from typing import Callable

import numpy as np
from imageio import imread

from extraction.core.functions.distributors import reduce_z


def load_tiled_image(filename):
    tImg = imread(filename)
    info = json.loads(tImg.meta.get("Description", "{}"))
    tw, th = info.get("tilesize", tImg.shape[0:2])
    nt = info.get("ntiles", 1)
    nr, nc = info.get("layout", (1, 1))
    nc_final_row = np.mod(nt, nc)
    img = np.zeros((tw, th, nt), dtype=tImg.dtype)
    for i in range(nr):
        i_nc = nc_final_row if i + 1 == nr and nc_final_row > 0 else nc
        for j in range(i_nc):
            ind = i * nc + j
            img[:, :, ind] = tImg[i * tw : (i + 1) * tw, j * th : (j + 1) * th]
    return img, info


def load_paired_images(filenames, typeA="Brightfield", typeB="segoutlines"):
    re_imlbl = re.compile(r"^(.*)_(" + typeA + r"|" + typeB + r")\.png$")
    # For groupby to work, the list needs to be sorted; also has the side
    # effect of ensuring filenames is no longer a generator
    filenames = sorted(filenames)
    matches = [re_imlbl.match(f.name) for f in filenames]
    valid = filter(lambda m: m[0], zip(matches, filenames))
    grouped = {
        k: {m.group(2): f for m, f in v}
        for k, v in groupby(valid, key=lambda m: m[0].group(1))
    }
    valid = [
        set(v.keys()).issuperset({typeA, typeB}) for v in grouped.values()
    ]
    if not all(valid):
        raise Exception
    return {
        lbl: {t: load_tiled_image(f) for t, f in g.items()}
        for lbl, g in grouped.items()
    }


def load(path=None):
    """
    Loads annotated pngs into memory. Only designed for GFP and brightfield.

    input
    :path: Folder used to look for images

    returns
    list of dictionaries containing GFP, Brightfield and segoutlines channel
    """
    if path is None:

        # FUTURE can be replaced by importlib.resources.files('aliby') after upgrading to 3.11
        path = (
            files("aliby").parent.parent
            / "examples"
            / "extraction"
            / "pairs_data"
        )

    image_dir = Path(path)
    channels = ["Brightfield", "GFP"]
    imset = {"segoutlines": {}}
    for ch in channels:
        imset[ch] = {}
        pos = load_paired_images(image_dir.glob("*.png"), typeA=ch)
        for key, img in pos.items():
            imset[ch][key] = img[ch][0]
            imset["segoutlines"][key] = img["segoutlines"][0].astype(bool)

    return [{ch: imset[ch][pos] for ch in imset.keys()} for pos in pos.keys()]


def load_1z(fun: Callable = np.maximum, path: str = None):
    """
    ---
        fun: Function used to reduce the multiple stacks
        path: Path to pass to load function


    """
    dsets = load(path=path)
    dsets_1z = []
    for dset in dsets:
        tmp = {}
        for ch, img in dset.items():
            if ch == "segoutlines":
                tmp[ch] = img
            else:
                tmp[ch] = reduce_z(img, fun)

        dsets_1z.append(tmp)

    return dsets_1z
