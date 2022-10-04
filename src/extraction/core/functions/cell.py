"""
Base functions to extract information from a single cell

These functions are automatically read by extractor.py, and so can only have the cell_mask and trap_image as inputs and must return only one value.

They assume that there are no NaNs in the image.
 We use bottleneck when it performs faster than numpy:
- Median
- values containing NaNs (We make sure this does not happen)

"""
import math
import typing as t

import bottleneck as bn
import faiss
import numpy as np
from scipy import ndimage


def area(cell_mask) -> int:
    """
    Find the area of a cell mask

    Parameters
    ----------
    cell_mask: 2d array
        Segmentation mask for the cell
    """
    return np.sum(cell_mask)


def eccentricity(cell_mask) -> float:
    """
    Find the eccentricity using the approximate major and minor axes

    Parameters
    ----------
    cell_mask: 2d array
        Segmentation mask for the cell
    """
    min_ax, maj_ax = min_maj_approximation(cell_mask)
    return np.sqrt(maj_ax**2 - min_ax**2) / maj_ax


def mean(cell_mask, trap_image) -> float:
    """
    Finds the mean of the pixels in the cell.

    Parameters
    ----------
    cell_mask: 2d array
        Segmentation mask for the cell
    trap_image: 2d array
    """
    return np.mean(trap_image[cell_mask])


def median(cell_mask, trap_image) -> int:
    """
    Finds the median of the pixels in the cell.

    Parameters
    ----------
    cell_mask: 2d array
        Segmentation mask for the cell
    trap_image: 2d array
    """
    return bn.median(trap_image[cell_mask])


def max2p5pc(cell_mask, trap_image) -> float:
    """
    Finds the mean of the brightest 2.5% of pixels in the cell.

    Parameters
    ----------
    cell_mask: 2d array
        Segmentation mask for the cell
    trap_image: 2d array
    """
    # number of pixels in mask
    npixels = np.sum(cell_mask)
    n_top = int(np.ceil(npixels * 0.025))
    # sort pixels in cell and find highest 2.5%
    pixels = trap_image[cell_mask]
    top_values = bn.partition(pixels, len(pixels) - n_top)[-n_top:]

    # find mean of these highest pixels
    return np.mean(top_values)


def max5px(cell_mask, trap_image) -> float:
    """
    Finds the mean of the five brightest pixels in the cell.

    Parameters
    ----------
    cell_mask: 2d array
        Segmentation mask for the cell
    trap_image: 2d array
    """
    # sort pixels in cell
    pixels = trap_image[cell_mask]
    top_values = bn.partition(pixels, len(pixels) - 5)[-5:]
    # find mean of five brightest pixels
    max5px = np.mean(top_values)
    return max5px


def std(cell_mask, trap_image):
    """
    Finds the standard deviation of the values of the pixels in the cell.

    Parameters
    ----------
    cell_mask: 2d array
        Segmentation mask for the cell
    trap_image: 2d array
    """
    return np.std(trap_image[cell_mask])


def k2_major_median(cell_mask, trap_image):
    """
    Finds the medians of the major cluster after clustering the pixels in the cell into two clusters.

    Parameters
    ----------
    cell_mask: 2d array
        Segmentation mask for the cell
    trap_image: 2d array

    Returns
    -------
    median: float
        The median of the major cluster of two clusters
    """
    if bn.anynan(trap_image):
        cell_mask[np.isnan(trap_image)] = False
    X = trap_image[cell_mask].reshape(-1, 1).astype(np.float32)
    # cluster pixels in cell into two clusters
    indices = faiss.IndexFlatL2(X.shape[1])
    # (n_clusters=2, random_state=0).fit(X)
    _, indices = indices.search(X, k=2)
    high_indices = np.argmax(indices, axis=1).astype(bool)
    # find the median of pixels in the largest cluster
    # high_masks = np.logical_xor(  # Use casting to obtain masks
    #     high_indices.reshape(-1, 1), np.tile((0, 1), X.shape[0]).reshape(-1, 2)
    # )
    major_median = bn.median(X[high_indices])
    return major_median


def volume(cell_mask) -> float:
    """
    Estimates the volume of the cell assuming it is an ellipsoid with the mask providing a cross-section through the median plane of the ellipsoid.

    Parameters
    ----------
    cell_mask: 2d array
        Segmentation mask for the cell
    """
    min_ax, maj_ax = min_maj_approximation(cell_mask)
    return (4 * np.pi * min_ax**2 * maj_ax) / 3


def conical_volume(cell_mask):
    """
    Estimates the volume of the cell

    Parameters
    ----------
    cell_mask: 2D array
        Segmentation mask for the cell
    """
    padded = np.pad(cell_mask, 1, mode="constant", constant_values=0)
    nearest_neighbor = (
        ndimage.morphology.distance_transform_edt(padded == 1) * padded
    )
    return 4 * np.sum(nearest_neighbor)


def spherical_volume(cell_mask):
    """
    Estimates the volume of the cell assuming it is a sphere with the mask providing a cross-section through the median plane of the sphere.

    Parameters
    ----------
    cell_mask: 2d array
        Segmentation mask for the cell
    """
    total_area = area(cell_mask)
    r = math.sqrt(total_area / np.pi)
    return (4 * np.pi * r**3) / 3


def min_maj_approximation(cell_mask) -> t.Tuple[int]:
    """
    Finds the lengths of the minor and major axes of an ellipse from a cell mask.

    Parameters
    ----------
    cell_mask: 3d array
        Segmentation masks for cells
    """
    # pad outside with zeros so that the distance transforms have no edge artifacts
    padded = np.pad(cell_mask, 1, mode="constant", constant_values=0)
    # get the distance from the edge, masked
    nn = ndimage.morphology.distance_transform_edt(padded == 1) * padded
    # get the distance from the top of the cone, masked
    dn = ndimage.morphology.distance_transform_edt(nn - nn.max()) * padded
    # get the size of the top of the cone (points that are equally maximal)
    cone_top = ndimage.morphology.distance_transform_edt(dn == 0) * padded
    # minor axis = largest distance from the edge of the ellipse
    min_ax = np.round(np.max(nn))
    # major axis = largest distance from the cone top
    # + distance from the center of cone top to edge of cone top
    maj_ax = np.round(np.max(dn) + np.sum(cone_top) / 2)
    return min_ax, maj_ax
