""" How to do the nuc Est Conv from MATLAB
Based on the code in MattSegCode/Matt Seg
GUI/@timelapseTraps/extractCellDataStacksParfor.m

Especially lines 342 to  399.
This part only replicates the method to get the nuc_est_conv values
"""
import numpy as np
import scipy
import skimage


def matlab_style_gauss2D(shape=(3, 3), sigma=0.5):
    """
    2D gaussian mask - should give the same result as MATLAB's
    fspecial('gaussian',[shape],[sigma])
    """
    m, n = [(ss - 1.0) / 2.0 for ss in shape]
    y, x = np.ogrid[-m : m + 1, -n : n + 1]
    h = np.exp(-(x * x + y * y) / (2.0 * sigma * sigma))
    h[h < np.finfo(h.dtype).eps * h.max()] = 0
    sumh = h.sum()
    if sumh != 0:
        h /= sumh
    return h


def gauss3D(shape=(3, 3, 3), sigma=(0.5, 0.5, 0.5)):
    """3D gaussian mask - based on MATLAB's fspecial but made 3D."""
    m, n, p = [(ss - 1.0) / 2.0 for ss in shape]
    z, y, x = np.ogrid[-p : p + 1, -m : m + 1, -n : n + 1]
    sigmax, sigmay, sigmaz = sigma
    xx = (x ** 2) / (2 * sigmax)
    yy = (y ** 2) / (2 * sigmay)
    zz = (z ** 2) / (2 * sigmaz)
    h = np.exp(-(xx + yy + zz))
    h[h < np.finfo(h.dtype).eps * h.max()] = 0  # Truncate
    sumh = h.sum()
    if sumh != 0:
        h /= sumh
    return h


def small_peaks_conv(cell_mask, trap_image):
    cell_fluo = trap_image[cell_mask]
    # Get the number of pixels in the cell
    num_cell_fluo = len(np.nonzero(cell_fluo)[0])
    # Sort cell pixels in descending fluorescence order
    ratio_overlap = num_cell_fluo * 0.025  # TODO what is this?

    # Small Peak Conv
    # Convolution parameters
    conv_matrix = np.zeros((3, 3))
    # This makes a matrix with zeros in the corners and ones every where else
    # Basically the minimal disk.
    conv_matrix[1, :] = 1
    conv_matrix[:, 1] = 1
    # Reshape so that it is the size of a fifth of the cell, which is what we
    # expect the size of the nucleus to be.
    # TODO directly get a disk of that size?
    # new_shape = tuple(x * ratio_overlap / 5 for x in conv_matrix.shape)
    # conv_matrix = scipy.misc.imresize(conv_matrix, new_shape)
    conv_matrix = skimage.morphology.disk(3 * ratio_overlap / 5)
    # Apply convolution to the image
    # TODO maybe rename 'conv_matrix' to 'kernel'
    fluo_peaks = scipy.signal.convolve(trap_image, conv_matrix, "same")
    fluo_peaks = fluo_peaks[cell_mask]
    small_peak_conv = np.max(fluo_peaks)
    return small_peak_conv


def nuc_est_conv(cell_mask, trap_image):
    """
    :param cell_mask: the segmentation mask of the cell (filled)
    :param trap_image: the image for the trap in which the cell is (all
    channels)
    """
    cell_loc = cell_mask  # np.where(cell_mask)[0]
    cell_fluo = trap_image[cell_mask]
    num_cell_fluo = len(np.nonzero(cell_fluo)[0])

    # Nuc Est Conv
    alpha = 0.95
    approx_nuc_radius = np.sqrt(0.085 * num_cell_fluo / np.pi)
    chi2inv = scipy.stats.distributions.chi2.ppf(alpha, df=2)
    sd_est = approx_nuc_radius / np.sqrt(chi2inv)

    nuc_filt_hw = np.ceil(2 * approx_nuc_radius)
    nuc_filter = matlab_style_gauss2D((2 * nuc_filt_hw + 1,) * 2, sd_est)

    cell_image = trap_image - np.median(cell_fluo)
    cell_image[~cell_loc] = 0

    nuc_conv = scipy.signal.convolve(cell_image, nuc_filter, "same")
    nuc_est_conv = np.max(nuc_conv)
    nuc_est_conv /= np.sum(nuc_filter ** 2) * alpha * np.pi * chi2inv * sd_est ** 2
    return nuc_est_conv


def nuc_conv_3d(cell_mask, trap_image, pixel_size=0.23, spacing=0.6):
    cell_mask = np.dstack([cell_mask] * trap_image.shape[-1])
    ratio = spacing / pixel_size
    cell_fluo = trap_image[cell_mask]
    num_cell_fluo = len(np.nonzero(cell_fluo)[0])
    # Nuc Est Conv
    alpha = 0.95
    approx_nuc_radius = np.sqrt(0.085 * num_cell_fluo / np.pi)
    chi2inv = scipy.stats.distributions.chi2.ppf(alpha, df=2)
    sd_est = approx_nuc_radius / np.sqrt(chi2inv)
    nuc_filt_hw = np.ceil(2 * approx_nuc_radius)
    nuc_filter = gauss3D((2 * nuc_filt_hw + 1,) * 3, (sd_est, sd_est, sd_est * ratio))
    cell_image = trap_image - np.median(cell_fluo)
    cell_image[~cell_mask] = 0
    nuc_conv = scipy.signal.convolve(cell_image, nuc_filter, "same")
    nuc_est_conv = np.max(nuc_conv)
    nuc_est_conv /= np.sum(nuc_filter ** 2) * alpha * np.pi * chi2inv * sd_est ** 2
    return nuc_est_conv
