"""
Make the masks an extraction needs and the images less background.

Give the cells' masks, labels and outlines for each trap, the masks of
their vacuoles and cytoplasms, and the masks of the background.
"""

import bottleneck as bn
import numpy as np
from extraction.functions.cell_functions import identify_vacuole
from sooth import tile_shape


def get_masks(tp, masks, cells):
    """Get masks as a list with an array of masks per trap."""
    if masks is None:
        raw_masks = cells.at_time(tp, kind="mask")
        masks = {trap_id: [] for trap_id in range(cells.ntraps)}
        for trap_id, trap_cells in raw_masks.items():
            if len(trap_cells):
                masks[trap_id] = np.stack(np.array(trap_cells)).astype(bool)
    # one array of shape (n_cells, Y, X) per trap
    masks = [np.array(v) for v in masks.values()]
    return masks


def get_cell_labels(tp, cell_labels, cells):
    """Get cell labels per trap as a dict with trap_ids as keys."""
    if cell_labels is None:
        raw_cell_labels = cells.labels_at_time(tp)
        cell_labels = {
            trap_id: raw_cell_labels.get(trap_id, [])
            for trap_id in range(cells.ntraps)
        }
    return cell_labels


def get_outlines(tp, cell_labels, cells):
    """Get labelled cell outlines as a dict with trap_ids as keys."""
    # read once: at_time reads every trap's masks from the h5 file
    edgemasks_at_tp = cells.at_time(tp, kind="edgemask")
    cell_mask_dict = {
        trap_id: edgemasks_at_tp.get(trap_id, [])
        for trap_id in range(cells.ntraps)
    }
    cell_labels = get_cell_labels(tp, cell_labels, cells)
    outlines_dict = {}
    for trap_id in range(cells.ntraps):
        edgemasks = cell_mask_dict[trap_id]
        labels = cell_labels[trap_id]
        if edgemasks and labels:
            outlines_dict[trap_id] = np.stack(
                [
                    cell_mask * label
                    for cell_mask, label in zip(edgemasks, labels)
                ]
            ).max(axis=0)
        else:
            outlines_dict[trap_id] = np.array([])
    return outlines_dict


def compute_intracellular_masks(outlines, masks, img):
    """
    Compute vacuole and cytoplasm sub-masks from brightfield.

    Parameters
    ----------
    outlines: list of arrays
        Labelled-outline images per trap, each (Y, X) with
        integer cell labels.
    masks: list of arrays
        Cell masks per trap, each (ncells, Y, X).
    img: dict
        Image dict containing "Brightfield" key with shape
        (ntraps, Z, Y, X); the vacuole CNN projects over z.

    Returns
    -------
    vac_masks: list of arrays
        Vacuole masks per trap with same structure as masks.
    cyt_masks: list of arrays
        Cytoplasm masks per trap with same structure as masks.
    """
    bf = img["Brightfield"]
    vac_masks = []
    cyt_masks = []
    for trap_outline, trap_masks, bf_trap in zip(outlines, masks, bf):
        if len(trap_masks):
            # single binary vacuole image for the whole trap
            trap_vac = identify_vacuole(trap_outline, bf_trap)
            # split into per-cell masks
            vac_list, cyt_list = [], []
            for cell_mask in trap_masks:
                cell_vac = cell_mask & trap_vac
                if cell_vac.any():
                    vac_list.append(cell_vac)
                    cyt_list.append(cell_mask & ~trap_vac)
                else:
                    # no vacuole found; cytoplasm is also undefined
                    vac_list.append(np.zeros_like(cell_mask))
                    cyt_list.append(np.zeros_like(cell_mask))
            vac_masks.append(np.stack(vac_list))
            cyt_masks.append(np.stack(cyt_list))
        else:
            vac_masks.append(np.array([]))
            cyt_masks.append(np.array([]))
    return vac_masks, cyt_masks


def get_background_masks(masks, tile_size, pdms_mask=None):
    """
    Generate boolean background masks for pre-reduction subtraction.

    Combine cell masks per trap and take the logical inverse to
    identify background pixels. These masks are used in
    ``get_imgs_background_subtract`` to compute and subtract the
    median background independently for each z-slice, before the
    z-reduction step.

    Parameters
    ----------
    masks: list of arrays
        Segmentation masks per trap, each (N_traps, Y, X).
    tile_size: int
        Size of each tile in pixels.
    pdms_mask: array (optional)
        Boolean mask of the PDMS trap, shape (Y, X), which is
        autofluorescent and is not background.

    Returns
    -------
    bgs: array
        Boolean background masks with shape (N_traps, Y, X).
    """
    combined = np.array(
        [
            (
                np.any(m, axis=0)
                if len(m)
                else np.zeros(tile_shape(tile_size), dtype=bool)
            )
            for m in masks
        ]
    )
    bgs = ~combined
    if pdms_mask is not None:
        bgs = bgs & ~pdms_mask
    return bgs


def get_imgs_background_subtract(params, tree_dict, tiles, bgs):
    """
    Get raw and background-subtracted fluorescence images.

    Subtract the median background per trap per z-slice before
    z-reduction, so each slice has its own background level
    removed. This differs from the ``imBackground`` trap function,
    which reports the background of an already z-reduced image.

    Parameters
    ----------
    params: ExtractorParameters
        Gives the channels to subtract a background from, the
        background channels and the multichannel functions.
    tree_dict: dict
        Extraction tree specifying channels and functions.
    tiles: array
        Tile images with shape (traps, channels, 1, Z, Y, X).
    bgs: array
        Boolean background masks with shape (traps, Y, X) from
        ``get_background_masks``.

    Returns
    -------
    img: dict
        Raw images keyed by channel name.
    img_bgsub: dict
        Background-subtracted images keyed by channel + '_bgsub',
        or None for channels without background subtraction.
    """
    img = {}
    img_bgsub = {}
    # a background channel, such as Cy5, is not extracted less its
    # background, so its image less background is made only for a
    # function of several channels that names it
    unwanted = set(params.background_channels) - {
        ch for spec in params.multichannel_funs.values() for ch in spec[0]
    }
    for ch in tree_dict["channels_tree"]:
        if tiles is not None and len(tiles):
            # image data for all traps for a particular channel
            # and time point arranged as (traps, Z, Y, X)
            ch_idx = tree_dict["channels"].index(ch)
            img[ch] = tiles[:, ch_idx, 0, ...]
            if (
                bgs.any()
                and ch in params.subtract_background
                and ch not in unwanted
                and img[ch] is not None
            ):
                # subtract median background per trap per z
                # img[ch] has shape (traps, Z, Y, X)
                # bgs has shape (traps, Y, X)
                ch_img = img[ch]
                n_traps, n_z = ch_img.shape[:2]
                bg_medians = np.empty((n_traps, n_z))
                for i in range(n_traps):
                    for j in range(n_z):
                        bg_medians[i, j] = bn.nanmedian(ch_img[i, j][bgs[i]])
                # subtract: broadcast (traps, Z) over (Y, X)
                img_bgsub[ch + "_bgsub"] = (
                    ch_img - bg_medians[:, :, np.newaxis, np.newaxis]
                )
            else:
                img_bgsub[ch + "_bgsub"] = None
        else:
            img[ch] = None
            img_bgsub[ch + "_bgsub"] = None
    return img, img_bgsub
