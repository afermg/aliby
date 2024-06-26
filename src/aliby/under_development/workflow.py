#!/usr/bin/env jupyter
from aliby.io.dataset import DatasetIndFiles
from aliby.io.image import ImageIndFiles
from aliby.segment.dispatch import dispatch_segmenter
from aliby.tile.tiler import Tiler, TilerParameters
from aliby.track.dispatch import dispatch_tracker
from extraction.core.extractor import Extractor, ExtractorParameters

path = "/home/amunoz/gsk/batches/ELN201687_subset/ELN201687_subset/H00DJKJread1BF48hrs_20230926_095825/"

# Load dataset from a regular expression

dif = DatasetIndFiles(
    path,
    regex=".+\/(.+)\/_.+([A-P][0-9]{2}).*_T([0-9]{4})F([0-9]{3}).*Z([0-9]{2}).*[0-9].tif",
    dimorder="CTZ",
)

wildcard, image_filenames = list(dif.get_position_ids().values())[0]

# Load Multidimensional image from multiple individual files using
image = ImageIndFiles(wildcard, image_filenames=image_filenames)

# Tiling

tiler = Tiler.from_image(image, TilerParameters.default(tile_size=None, ref_channel=0))
drifts_info = tiler.run_tp(0)
pixels = tiler.get_tp_data(0, 0)

# Segmentation
segment = dispatch_segmenter("nuclei", diameter=None, channels=[0, 0])
masks = segment(pixels)

# Tracking

## track the masks against themselves
track = dispatch_tracker("stitch")
tracked_mask = track(masks, masks)

# Extraction


# The extraction tree indicates how to combine masks and images to produce quantities
# Example extraction tree; there are many ways to build them
def gen_minimal_tree(
    fluorescence_channels=list[str],
) -> dict[str, dict[str, dict[str, str]]]:
    """
    Generate a basic extraciton tree based on the type of channels (fluorescence or not).
    """
    tree = {
        "channels_tree": {
            "general": {
                "None": [
                    "area",
                    "eccentricity",
                    "centroid_x",
                    "centroid_y",
                ],
            }
        },
        "multichannel_ops": {},
    }
    return tree


tree = gen_minimal_tree()
extractor = Extractor(ExtractorParameters(tree=tree), tiler=tiler)
extracted_tp = extractor.run_tp([0], tree=tree, masks=[masks], save=False)
"""
extracted_tp
Out[8]: 
{'general/None/area':                    0
 trap cell_label     
 0    1           130
      2           993,
 'general/None/eccentricity':                         0
 trap cell_label          
 0    1           0.866025
      2           0.877268,
 'general/None/centroid_x':                           0
 trap cell_label            
 0    1           385.123077
      2           807.623364,
 'general/None/centroid_y':                          0
 trap cell_label           
 0    1           43.038462
      2           91.235650}
"""
