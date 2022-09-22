#!/usr/bin/env python3

import numpy as np
from matplotlib import cm, colors


def generate_palette_map(df):
    """Create a palette map based on the strains in a dataframe"""
    strain_list = np.unique(df.index.get_level_values("strain"))
    palette_cm = cm.get_cmap("Set1", len(strain_list) + 1)
    palette_rgb = [
        colors.rgb2hex(palette_cm(index / len(strain_list))[:3])
        for index, _ in enumerate(strain_list)
    ]
    palette_map = dict(zip(strain_list, palette_rgb))
    return palette_map
