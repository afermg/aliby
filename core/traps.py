"""
A set of utilities for dealing with ALCATRAS traps
"""

#TODO portfolio of trap templates
import numpy as np
from skimage import transform, feature


def identify_trap_locations(image, trap_template, optimize_scale=True,
                            downscale=0.3,
                            trap_rotation=0):
    """
    Identify the traps in a single image based on a trap template.
    This assumes a trap template that is similar to the image in question
    (same camera, same magification; ideally same experiment).

    This method speeds up the search by downscaling both the image and
    the trap template before running the template match.
    It also optimizes the scale and the rotation of the trap template.

    :param image:
    :param trap_template:
    :param optimize_scale:
    :param downscale:
    :param trap_rotation:
    :return:
    """
    img = transform.rescale(image, downscale)
    temp = transform.rescale(trap_template, downscale)
    # TODO reimplement optimize rotation
    temp = transform.rotate(temp, trap_rotation,
                            cval=np.median(img))
    # TODO optimize over templates?

    if optimize_scale:
        scales = np.linspace(0.5, 2, 10)
        matches = matches = {scale: feature.match_template(
            img, transform.rescale(temp, scale),
            pad_input=True,
            mode='median')
            for scale in scales}
        best_scale = max(matches, key=lambda x: np.max(matches[x]))
        matched = matches[best_scale]
    else:
        matched = feature.match_template(img, temp, pad_input=True,
                                         mode='median')

    coordinates = feature.peak_local_max(
        transform.rescale(matched, 1 / downscale),
        min_distance=trap_template.shape[0],
        exclude_border=trap_template.shape[0] // 3)
    return coordinates

