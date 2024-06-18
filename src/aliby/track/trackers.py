#!/usr/bin/env jupyter

import numpy as np
from cellpose.utils import stitch3D

"""
If masks are 2d the tracking function returns the new masks relabelled.
If masks are 3d it returns the indices, which correspond to the first axis.
"""


def stitch(masks: np.ndarray) -> np.ndarray:
    """
    Wrapper that returns the new masks. It only returns the latest mask
    and performs no error correction.
    """
    return stitch3D(masks)[-1]
