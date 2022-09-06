"""
Base functions to extract information from a single cell

These functions are automatically read by extractor.py, and so can only have the cell_mask and trap_image as inputs and must return only one value.
"""
import numpy as np
from scipy import ndimage
from sklearn.cluster import KMeans


def area(cell_mask):
    """
    Find the area of a cell mask

    Parameters
    ----------
    cell_mask: 2d array
        Segmentation mask for the cell
    """
    return np.sum(cell_mask, dtype=int)


def eccentricity(cell_mask):
    """
    Find the eccentricity using the approximate major and minor axes

    Parameters
    ----------
    cell_mask: 2d array
        Segmentation mask for the cell
    """
    min_ax, maj_ax = min_maj_approximation(cell_mask)
    return np.sqrt(maj_ax**2 - min_ax**2) / maj_ax


def mean(cell_mask, trap_image):
    """
    Finds the mean of the pixels in the cell.

    Parameters
    ----------
    cell_mask: 2d array
        Segmentation mask for the cell
    trap_image: 2d array
    """
    return np.mean(trap_image[np.where(cell_mask)], dtype=float)


def median(cell_mask, trap_image):
    """
    Finds the median of the pixels in the cell.

    Parameters
    ----------
    cell_mask: 2d array
        Segmentation mask for the cell
    trap_image: 2d array
    """
    return np.median(trap_image[np.where(cell_mask)])


def max2p5pc(cell_mask, trap_image):
    """
    Finds the mean of the brightest 2.5% of pixels in the cell.

    Parameters
    ----------
    cell_mask: 2d array
        Segmentation mask for the cell
    trap_image: 2d array
    """
    # number of pixels in mask
    npixels = cell_mask.sum()
    top_pixels = int(np.ceil(npixels * 0.025))
    # sort pixels in cell
    sorted_vals = np.sort(trap_image[np.where(cell_mask)], axis=None)
    # find highest 2.5%
    top_vals = sorted_vals[-top_pixels:]
    # find mean of these highest pixels
    max2p5pc = np.mean(top_vals, dtype=float)
    return max2p5pc


def max5px(cell_mask, trap_image):
    """
    Finds the mean of the five brightest pixels in the cell.

    Parameters
    ----------
    cell_mask: 2d array
        Segmentation mask for the cell
    trap_image: 2d array
    """
    # sort pixels in cell
    sorted_vals = np.sort(trap_image[np.where(cell_mask)], axis=None)
    top_vals = sorted_vals[-5:]
    # find mean of five brightest pixels
    max5px = np.mean(top_vals, dtype=float)
    return max5px


def max5px_med(cell_mask, trap_image):
    """
    Finds the mean of the five brightest pixels in the cell divided by the median pixel value.

    Parameters
    ----------
    cell_mask: 2d array
        Segmentation mask for the cell
    trap_image: 2d array
    """
    # sort pixels in cell
    sorted_vals = np.sort(trap_image[np.where(cell_mask)], axis=None)
    top_vals = sorted_vals[-5:]
    # find mean of five brightest pixels
    max5px = np.mean(top_vals, dtype=float)
    # find the median
    med = np.median(sorted_vals)
    if med == 0:
        return np.nan
    else:
        return max5px / med


def max2p5pc_med(cell_mask, trap_image):
    """
    Finds the mean of the brightest 2.5% of pixels in the cell
    divided by the median pixel value.

    Parameters
    ----------
    cell_mask: 2d array
        Segmentation mask for the cell
    trap_image: 2d array
    """
    # number of pixels in mask
    npixels = cell_mask.sum()
    top_pixels = int(np.ceil(npixels * 0.025))
    # sort pixels in cell
    sorted_vals = np.sort(trap_image[np.where(cell_mask)], axis=None)
    # find highest 2.5%
    top_vals = sorted_vals[-top_pixels:]
    # find mean of these highest pixels
    max2p5pc = np.mean(top_vals, dtype=float)
    med = np.median(sorted_vals)
    if med == 0:
        return np.nan
    else:
        return max2p5pc / med


def std(cell_mask, trap_image):
    """
    Finds the standard deviation of the values of the pixels in the cell.

    Parameters
    ----------
    cell_mask: 2d array
        Segmentation mask for the cell
    trap_image: 2d array
    """
    return np.std(trap_image[np.where(cell_mask)], dtype=float)


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
    if np.any(cell_mask):
        X = trap_image[np.where(cell_mask)].reshape(-1, 1)
        # cluster pixels in cell into two clusters
        kmeans = KMeans(n_clusters=2, random_state=0).fit(X)
        high_clust_id = kmeans.cluster_centers_.argmax()
        # find the median of pixels in the largest cluster
        major_cluster = X[kmeans.predict(X) == high_clust_id]
        major_median = np.median(major_cluster, axis=None)
        return major_median
    else:
        return np.nan

def k2_minor_median(cell_mask, trap_image):
    """
    Finds the median of the minor cluster after clustering the pixels in the cell into two clusters.

    Parameters
    ----------
    cell_mask: 2d array
        Segmentation mask for the cell
    trap_image: 2d array

    Returns
    -------
    median: float
        The median of the minor cluster.
    """
    if np.any(cell_mask):
        X = trap_image[np.where(cell_mask)].reshape(-1, 1)
        # cluster pixels in cell into two clusters
        kmeans = KMeans(n_clusters=2, random_state=0).fit(X)
        low_clust_id = kmeans.cluster_centers_.argmin()
        # find the median of pixels in the smallest cluster
        minor_cluster = X[kmeans.predict(X) == low_clust_id]
        minor_median = np.median(minor_cluster, axis=None)
        return minor_median
    else:
        return np.nan


def volume(cell_mask):
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
    return 4 * (nearest_neighbor.sum())


def spherical_volume(cell_mask):
    '''
    Estimates the volume of the cell assuming it is a sphere with the mask providing a cross-section through the median plane of the sphere.

    Parameters
    ----------
    cell_mask: 2d array
        Segmentation mask for the cell
    '''
    area = cell_mask.sum()
    r = np.sqrt(area / np.pi)
    return (4 * np.pi * r**3) / 3


def min_maj_approximation(cell_mask):
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
    min_ax = np.round(nn.max())
    # major axis = largest distance from the cone top
    # + distance from the center of cone top to edge of cone top
    maj_ax = np.round(dn.max() + cone_top.sum() / 2)
    return min_ax, maj_ax
