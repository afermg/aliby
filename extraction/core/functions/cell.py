"""
Base functions to extract information from a single cell

These functions are automatically read, so only add new functions with
the same arguments as the existing ones.

Input:
:cell_mask: (x,y) 2-D cell mask
:trap_image: (x,y) 2-D or (x,y,z) 3-D cell mask


np.where is used to cover for cases where z>1
"""
import math

import numpy as np
from scipy import ndimage
from sklearn.cluster import KMeans


def area(cell_mask, trap_image=None):
    return np.sum(cell_mask, dtype=int)


def mean(cell_mask, trap_image):
    return np.mean(trap_image[np.where(cell_mask)], dtype=float)


def median(cell_mask, trap_image):
    return np.median(trap_image[np.where(cell_mask)])


def max2p5pc(cell_mask, trap_image):
    npixels = cell_mask.sum()
    top_pixels = int(np.ceil(npixels * 0.025))

    sorted_vals = np.sort(trap_image[np.where(cell_mask)], axis=None)
    top_vals = sorted_vals[-top_pixels:]
    max2p5pc = np.mean(top_vals, dtype=float)

    return max2p5pc


def max5px(cell_mask, trap_image):
    sorted_vals = np.sort(trap_image[np.where(cell_mask)], axis=None)
    top_vals = sorted_vals[-5:]
    max5px = np.mean(top_vals, dtype=float)

    return max5px


def std(cell_mask, trap_image):
    return np.std(trap_image[np.where(cell_mask)], dtype=float)


def k2_top_median(cell_mask, trap_image):
    # Use kmeans to cluster the contents of a cell in two, return the high median
    # Useful when a big non-tagged organelle (e.g. vacuole) occupies a big fraction
    # of the cell
    if not np.any(cell_mask):
        return np.nan

    X = trap_image[np.where(cell_mask)].reshape(-1, 1)
    kmeans = KMeans(n_clusters=2, random_state=0).fit(X)
    high_clust_id = kmeans.cluster_centers_.argmax()
    major_cluster = X[kmeans.predict(X) == high_clust_id]

    k2_top_median = np.median(major_cluster, axis=None)
    return k2_top_median


def membraneMax5(cell_mask, trap_image):
    pass


def membraneMedian(cell_mask, trap_image):
    pass


def volume(cell_mask, trap_image=None):
    """Volume from a cell mask, assuming an ellipse.

    Assumes the mask is the median plane of the ellipsoid.
    Assumes rotational symmetry around the major axis.
    """
    min_ax, maj_ax = min_maj_approximation(cell_mask, trap_image)
    return (4 * math.pi * min_ax**2 * maj_ax) / 3


def conical_volume(cell_mask, trap_image=None):
    padded = np.pad(cell_mask, 1, mode='constant', constant_values=0)
    nearest_neighbor = ndimage.morphology.distance_transform_edt(
        padded == 1) * padded
    return 4 * (nearest_neighbor.sum())


def spherical_volume(cell_mask, trap_image=None):
    area = cell_mask.sum()
    r = np.sqrt(area / np.pi)
    return (4 * np.pi * r**3) / 3


def min_maj_approximation(cell_mask, trap_image=None):
    """Length approximation of minor and major axes of an ellipse from mask.


    :param cell_mask:
    :param trap_image:
    :return:
    """
    padded = np.pad(cell_mask,1, mode='constant', constant_values=0)
    nn = ndimage.morphology.distance_transform_edt(padded == 1) * padded
    dn = ndimage.morphology.distance_transform_edt(nn - nn.max()) * padded
    cone_top = ndimage.morphology.distance_transform_edt(dn == 0) * padded
    min_ax = np.round(nn.max())
    maj_ax = np.round(dn.max() + cone_top.sum()/2)
    return min_ax, maj_ax


def eccentricity(cell_mask, trap_image=None):
    min_ax, maj_ax = min_maj_approximation(cell_mask)
    return np.sqrt(maj_ax**2 - min_ax**2)/maj_ax


