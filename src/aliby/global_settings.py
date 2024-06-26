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
