"""
Base functions to extract information from a single cell.

These functions are automatically read by extractor.py, and
so can only have the cell_mask and trap_image as inputs. They
must return only one value.

They assume that there are no NaNs in the image.
"""

import math
import typing as t

import numpy as np
from scipy import ndimage


def area(cell_mask) -> int:
    """
    Find the area of a cell mask.

    Parameters
    ----------
    cell_mask: 2d array
        Segmentation mask for the cell.
    """
    return np.sum(cell_mask)


def eccentricity(cell_mask) -> float:
    """
    Find the eccentricity using the approximate major and minor axes.

    Parameters
    ----------
    cell_mask: 2d array
        Segmentation mask for the cell.
    """
    min_ax, maj_ax = min_maj_approximation(cell_mask)
    return np.sqrt(maj_ax**2 - min_ax**2) / maj_ax


def mean(cell_mask, trap_image) -> float:
    """
    Find the mean of the pixels in the cell.

    Parameters
    ----------
    cell_mask: 2d array
        Segmentation mask for the cell.
    trap_image: 2d array
    """
    return np.mean(trap_image[cell_mask])


def total(cell_mask, trap_image) -> float:
    """
    Find the sum of the pixels in the cell.

    Parameters
    ----------
    cell_mask: 2d array
        Segmentation mask for the cell.
    trap_image: 2d array
    """
    return np.sum(trap_image[cell_mask])


def total_squared(cell_mask, trap_image) -> float:
    """
    WARNING: produces overflow error when converted to float16

    Find the sum of the square of the pixels in the cell.

    For finding variances.

    Parameters
    ----------
    cell_mask: 2d array
        Segmentation mask for the cell.
    trap_image: 2d array
    """
    return np.sum(trap_image[cell_mask] ** 2)


def median(cell_mask, trap_image) -> int:
    """
    Find the median of the pixels in the cell.

    Parameters
    ----------
    cell_mask: 2d array
         Segmentation mask for the cell.
    trap_image: 2d array
    """
    return np.median(trap_image[cell_mask])


def max2p5pc(cell_mask, trap_image) -> float:
    """
    Find the mean of the brightest 2.5% of pixels in the cell.

    Parameters
    ----------
    cell_mask: 2d array
        Segmentation mask for the cell.
    trap_image: 2d array
    """
    # number of pixels in mask
    npixels = np.sum(cell_mask)
    n_top = int(np.ceil(npixels * 0.025))
    # sort pixels in cell and find highest 2.5%
    pixels = trap_image[cell_mask]
    top_values = np.partition(pixels, len(pixels) - n_top)[-n_top:]
    # find mean of these highest pixels
    return np.mean(top_values)


def max5px_median(cell_mask, trap_image) -> float:
    """
    Estimate the degree of localisation.

    Find the mean of the five brightest pixels in the cell divided by the
    median of all pixels.

    Parameters
    ----------
    cell_mask: 2d array
        Segmentation mask for the cell.
    trap_image: 2d array
    """
    # sort pixels in cell
    pixels = trap_image[cell_mask]
    if len(pixels) > 5:
        top_values = np.partition(pixels, len(pixels) - 5)[-5:]
        # find mean of five brightest pixels
        max5px = np.mean(top_values)
        med = np.median(pixels)
        if med == 0:
            return np.nan
        else:
            return max5px / np.median(pixels)
    else:
        return np.nan


def std(cell_mask, trap_image):
    """
    Find the standard deviation of the values of the pixels in the cell.

    Parameters
    ----------
    cell_mask: 2d array
        Segmentation mask for the cell.
    trap_image: 2d array
    """
    return np.std(trap_image[cell_mask])


def volume(cell_mask) -> float:
    """
    Estimate the volume of the cell.

    Assumes the cell is an ellipsoid with the mask providing
    a cross-section through its median plane.

    Parameters
    ----------
    cell_mask: 2d array
        Segmentation mask for the cell.
    """
    min_ax, maj_ax = min_maj_approximation(cell_mask)
    return (4 * np.pi * min_ax**2 * maj_ax) / 3


def conical_volume(cell_mask):
    """
    Estimate the volume of the cell.

    Parameters
    ----------
    cell_mask: 2D array
        Segmentation mask for the cell
    """
    padded = np.pad(cell_mask, 1, mode="constant", constant_values=0)
    nearest_neighbor = ndimage.morphology.distance_transform_edt(padded == 1) * padded
    return 4 * np.sum(nearest_neighbor)


def spherical_volume(cell_mask):
    """
    Estimate the volume of the cell.

    Assumes the cell is a sphere with the mask providing
    a cross-section through its median plane.

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
    Find the lengths of the minor and major axes of an ellipse from a cell mask.

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


def moment_of_inertia(cell_mask, trap_image):
    """
    Find moment of inertia - a measure of homogeneity.

    From iopscience.iop.org/article/10.1088/1742-6596/1962/1/012028
    which cites ieeexplore.ieee.org/document/1057692.
    """
    # set pixels not in cell to zero
    trap_image[~cell_mask] = 0
    x = trap_image
    if np.any(x):
        # x-axis : column=x-axis
        columnvec = np.arange(1, x.shape[1] + 1, 1)[:, None].T
        # y-axis : row=y-axis
        rowvec = np.arange(1, x.shape[0] + 1, 1)[:, None]
        # find raw moments
        M00 = np.sum(x)
        M10 = np.sum(np.multiply(x, columnvec))
        M01 = np.sum(np.multiply(x, rowvec))
        # find centroid
        Xm = M10 / M00
        Ym = M01 / M00
        # find central moments
        Mu00 = M00
        Mu20 = np.sum(np.multiply(x, (columnvec - Xm) ** 2))
        Mu02 = np.sum(np.multiply(x, (rowvec - Ym) ** 2))
        # find invariants
        Eta20 = Mu20 / Mu00 ** (1 + (2 + 0) / 2)
        Eta02 = Mu02 / Mu00 ** (1 + (0 + 2) / 2)
        # find moments of inertia
        moi = Eta20 + Eta02
        return moi
    else:
        return np.nan


def ratio(cell_mask, trap_image):
    """Find the median ratio between two fluorescence channels."""
    if trap_image.ndim == 3 and trap_image.shape[-1] == 2:
        fl_0 = trap_image[..., 0][cell_mask]
        fl_1 = trap_image[..., 1][cell_mask]
        if np.any(fl_1 == 0):
            div = np.nan
        else:
            div = np.median(fl_0 / fl_1)
    else:
        div = np.nan
    return div


def centroid(cell_mask):
    """Find the cell's centroid."""
    weights_c = np.arange(1, cell_mask.shape[1] + 1, 1).reshape(1, cell_mask.shape[1])
    weights_v = np.arange(1, cell_mask.shape[0] + 1, 1).reshape(cell_mask.shape[0], 1)
    # moments
    M00 = np.sum(cell_mask)
    M10 = np.sum(np.multiply(cell_mask, weights_c))
    M01 = np.sum(np.multiply(cell_mask, weights_v))
    # centroid
    Xm = M10 / M00
    Ym = M01 / M00
    return (Xm, Ym)


def centroid_x(cell_mask):
    """Return x coordinate of a cell's centroid."""
    return centroid(cell_mask)[0]


def centroid_y(cell_mask):
    """Return y coordinate of a cell's centroid."""
    return centroid(cell_mask)[1]
