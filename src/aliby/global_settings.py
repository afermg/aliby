"""Global parameters and settings."""

# parameters to stop the pipeline when exceeded
earlystop = dict(
    min_tp=100,
    thresh_pos_clogged=0.4,
    thresh_trap_ncells=8,
    thresh_trap_area=0.9,
    ntps_to_eval=5,
)

# imaging properties of the microscope
imaging_specifications = {
    "pixel_size": 0.236,
    "z_size": 0.6,
    "z_spacing": 0.6,
    "magnification": 60,
    "tile_size": 117,
}

# possible imaging channels excluding Brightfield
possible_imaging_channels = [
    "Citrine",
    "GFP",
    "GFPFast",
    "mCherry",
    "Flavin",
    "Citrine",
    "mKO2",
    "Cy5",
    "cy5",
    "pHluorin405",
    "pHluorin488",
]

# functions to apply to cell outlines
outline_functions = [
    "area",
    "volume",
    "eccentricity",
    "centroid_x",
    "centroid_y",
]


# functions to apply to the fluorescence of each cell
fluorescence_functions = [
    "mean",
    "median",
    "std",
    "imBackground",
    "max5px_median",
]


def detect_brightfield_channels(
    pixels,
    ratio: float = 3.0,
) -> tuple[list[int], list[int]]:
    """Detect brightfield vs fluorescence channels by max pixel intensity.

    Brightfield channels have max intensity much higher than fluorescence.
    Uses ratio-based detection: a channel is brightfield if its max is
    at least ``ratio`` times the median of all channel maxima.

    Parameters
    ----------
    pixels : array-like
        Image data with shape (..., C, Z, Y, X) or (C, Z, Y, X).
    ratio : float
        A channel is brightfield if its max exceeds ``ratio`` times the
        median max across channels.

    Returns
    -------
    tuple of (list[int], list[int])
        (brightfield_indices, fluorescence_indices)
    """
    import numpy as np

    arr = np.asarray(pixels)
    # Normalize to (C, ...) by collapsing leading dims
    while arr.ndim > 4:  # Remove T and any extra leading dims
        arr = arr[0]
    # arr is now (C, Z, Y, X)
    n_channels = arr.shape[0]
    maxima = np.array([int(np.max(arr[ch])) for ch in range(n_channels)])
    sorted_maxima = np.sort(maxima)[::-1]

    bf, fluo = [], []
    for ch in range(n_channels):
        ch_max = maxima[ch]
        # A channel is BF if it has the highest max AND is at least
        # ratio times higher than the second-highest channel
        others = np.delete(maxima, ch)
        second_highest = np.max(others) if len(others) > 0 else 0
        if second_highest > 0 and ch_max >= ratio * second_highest:
            bf.append(ch)
        else:
            fluo.append(ch)
    return bf, fluo


# default time interval in seconds
default_time_interval = 300

# maximum possible size of data frame in h5 files
h5_max_ncells = 2e5
h5_max_tps = 1500
