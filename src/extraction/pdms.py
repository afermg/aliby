"""Find the PDMS trap within a tile, which is not background."""

import logging
import typing as t
import warnings

import numpy as np
from scipy.ndimage import (
    binary_closing,
    binary_dilation,
    binary_fill_holes,
    label as label_regions,
)
from skimage.filters import threshold_otsu
from skimage.morphology import disk


def remove_small_regions(mask: np.ndarray, min_size: int) -> np.ndarray:
    """
    Remove connected regions comprising fewer than min_size pixels.

    Parameters
    ----------
    mask: array
        A boolean array.
    min_size: int
        Smallest number of pixels a region may have.
    """
    labelled, no_regions = label_regions(mask)
    if no_regions:
        sizes = np.bincount(labelled.ravel())
        small = np.flatnonzero(sizes < min_size)
        mask = mask & ~np.isin(labelled, small)
    return mask


def compute_pdms_mask(
    brightfield: np.ndarray,
    masks: t.List[np.ndarray],
    min_no_tiles: int = 10,
) -> t.Optional[np.ndarray]:
    """
    Find the PDMS trap within a tile from the brightfield images.

    The PDMS is autofluorescent and so is not background, but it lies
    outside the cells and is therefore included in any estimate of the
    background unless it is explicitly excluded.

    Tiles are corrected for drift and are centred on the traps, and so
    the trap lies at the same place in every tile both over time and
    from one tile to the next. Taking the median over tiles therefore
    reinforces the trap. Cells are blanked rather than averaged away
    because cells are held in the trap's pocket and so lie in similar
    places in different tiles.

    Parameters
    ----------
    brightfield: array
        Brightfield images for all tiles, shape (no_tiles, Y, X).
    masks: list of arrays
        Segmentation masks per tile, each (no_cells, Y, X).
    min_no_tiles: int (optional)
        Minimum number of usable tiles needed to find the trap.

    Returns
    -------
    mask: array or None
        A boolean array, shape (Y, X), that is True for the pixels of
        the PDMS trap, or None if the trap cannot be found reliably.
    """
    logger = logging.getLogger("aliby")
    if brightfield is None or not len(brightfield):
        return None
    stack = np.array(brightfield, dtype=float)
    # blank the cells
    for tile_id, mask_set in enumerate(masks):
        if len(mask_set):
            stack[tile_id][np.any(mask_set, axis=0)] = np.nan
    # ignore tiles that are padded or wholly covered by cells
    stack = stack[~np.all(np.isnan(stack), axis=(1, 2))]
    if len(stack) < min_no_tiles:
        logger.warning(
            f"Extractor: only {len(stack)} tiles are available, and so "
            "the PDMS trap will not be masked."
        )
        return None
    with warnings.catch_warnings():
        # some pixels may be covered by cells in every tile
        warnings.simplefilter("ignore", RuntimeWarning)
        static = np.nanmedian(stack, axis=0)
    level = np.nanmedian(static)
    static = np.where(np.isnan(static), level, static)
    # the trap both absorbs and refracts and so appears either darker
    # or brighter than the medium
    deviation = np.abs(static - level)
    mask = deviation > threshold_otsu(deviation)
    # the trap's interior may match the medium, lying between its edges;
    # treat pixels beyond a tile as trap so that a trap running into a
    # tile's edge is not eroded, erring towards masking too much
    mask = binary_closing(mask, disk(3), border_value=1)
    mask = binary_fill_holes(mask)
    mask = remove_small_regions(mask, max(9, mask.size // 1000))
    # allow for drift and for tiles being imperfectly centred on traps
    mask = binary_dilation(mask, disk(2))
    fraction = mask.mean()
    if fraction > 0.6:
        logger.warning(
            f"Extractor: the PDMS trap appears to cover {fraction:.0%} of "
            "a tile, which is implausible, and so it will not be masked."
        )
        return None
    elif fraction > 0.4:
        logger.warning(
            f"Extractor: the PDMS trap covers {fraction:.0%} of a tile. "
            "Background estimates may be unreliable."
        )
    else:
        logger.info(
            f"Extractor: the PDMS trap covers {fraction:.0%} of a tile."
        )
    return mask
