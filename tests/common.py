REGEX_PARAMETERS = (
    (
        "crop_cellpainting_256",
        ".*__([A-Z][0-9]{2})__([0-9])__([A-Za-z]+).tif",
        "WFC",
    ),
    (
        "crop_timeseries_alcatras_round_diff_dims_293",
        ".*/([^/]+)/.+_([0-9]{6})_([A-Za-z0-9]+)_(?:.*_)?([0-9]+).tif",
        "FTCZ",
    ),
    (
        "crop_timeseries_alcatras_square_same_channels_293",
        ".*/([^/]+)/.+_([0-9]{6})_([A-Za-z0-9]+)_(?:.*_)?([0-9]+).tif",
        "FTCZ",
    ),
)
