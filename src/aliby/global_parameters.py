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
    "spacing": 0.6,
}

# possible imaging channels
possible_imaging_channels = [
    "Citrine",
    "GFP",
    "GFPFast",
    "mCherry",
    "Flavin",
    "Citrine",
    "mKO2",
    "Cy5",
    "pHluorin405",
    "pHluorin488",
]

# functions to apply to the fluorescence of each cell
fluorescence_functions = [
    "mean",
    "median",
    "std",
    "imBackground",
    "max5px_median",
]

# default fraction of time a cell must be in the experiment to be kept by Signal
signal_retained_cutoff = 0.8
