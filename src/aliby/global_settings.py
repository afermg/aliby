"""Global parameters and settings."""


class GlobalSettings:
    """Define all hard-coded parameters used by aliby."""

    def __init__(self):
        """Define global parameters."""
        # parameters to stop the pipeline when exceeded
        self.earlystop = {
            "min_tp": 100,
            "thresh_pos_clogged": 0.4,
            "thresh_trap_ncells": 8,
            "thresh_trap_area": 0.9,
            "ntps_to_eval": 5,
        }

        # microscope and camera properties
        self.imaging_specifications = {
            "pixel_size": 0.236,
            "z_size": 0.6,
            "z_spacing": 0.6,
            "magnification": 60,
            "tile_size": 117,
            "ref_z": 0,
        }

        # possible imaging channels excluding Brightfield
        self.possible_imaging_channels = [
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
            "NADH",
        ]

        # functions to apply to cell outlines
        self.outline_functions = [
            "area",
            "volume",
            "eccentricity",
            "centroid_x",
            "centroid_y",
        ]

        # functions to apply to the fluorescence of each cell
        self.fluorescence_functions = [
            "mean",
            "median",
            "std",
            "imBackground",
        ]

        # default time interval in seconds
        self.default_time_interval = 300

    def get(self, global_property: str):
        """Get a global setting."""
        return getattr(self, global_property)

    @property
    def get_all(self):
        """Get all attributes with their values."""
        return vars(self)


global_settings = GlobalSettings()
