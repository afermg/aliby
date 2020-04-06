"""
A set of utilities for dealing with ALCATRAS traps
"""

# TODO portfolio of trap templates
import numpy as np
from skimage import transform, feature


def identify_trap_locations(image, trap_template, optimize_scale=True,
                            downscale=0.3, trap_size=None):
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
    trap_size = trap_size if trap_size is not None else trap_template.shape[0]
    img = transform.rescale(image, downscale)
    temp = transform.rescale(trap_template, downscale)

    # TODO random search hyperparameter optimization
    # optimize rotation
    matches = {rotation: feature.match_template(
        img, transform.rotate(temp,
                              rotation,
                              cval=np.median(img)),
        pad_input=True,
        mode='median'
    )**2 for rotation in [0, 90, 180, 270]}
    best_rotation = max(matches, key=lambda x: np.max(matches[x]))
    temp = transform.rotate(temp, best_rotation, cval=np.median(img))

    if optimize_scale:
        scales = np.linspace(0.5, 2, 10)
        matches = {scale: feature.match_template(
            img, transform.rescale(temp, scale),
            pad_input=True,
            mode='median')**2
            for scale in scales}
        best_scale = max(matches, key=lambda x: np.max(matches[x]))
        matched = matches[best_scale]
    else:
        matched = feature.match_template(img, temp, pad_input=True,
                                         mode='median')

    coordinates = feature.peak_local_max(
        transform.rescale(matched, 1 / downscale),
        min_distance=trap_template.shape[0]*0.80,
        exclude_border=trap_size // 3)
    return coordinates
