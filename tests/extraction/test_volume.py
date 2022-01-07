import pytest
from skimage.morphology import disk, erosion
from skimage import draw
import numpy as np

from extraction.core.functions.cell import volume
from extraction.core.functions.cell import min_maj_approximation
from extraction.core.functions.cell import eccentricity

threshold = 0.01
radii = list(range(10, 100, 10))
circularities = np.arange(0.4, 1., 0.1)
eccentricities = np.arange(0, 0.9, 0.1)
rotations = [10, 20, 30, 40, 50, 60, 70, 80, 90]


def ellipse(x, y, rotate=0):
    shape = (4 * x, 4 * y)
    img = np.zeros(shape, dtype=np.uint8)
    rr, cc = draw.ellipse(2 * x, 2 * y, x, y, rotation=np.deg2rad(rotate))
    img[rr, cc] = 1
    return img


def maj_from_min(min_ax, ecc):
    y = np.sqrt(min_ax ** 2 / (1 - ecc ** 2))
    return np.round(y).astype(int)


@pytest.mark.parametrize('r', radii)
def test_volume_circular(r):
    im = disk(r)
    v = volume(im)
    real_v = (4 * np.pi * r ** 3) / 3
    err = np.abs(v - real_v) / real_v
    assert err < threshold
    assert np.isclose(v, real_v, rtol=threshold * real_v)


@pytest.mark.parametrize('x', radii)
@pytest.mark.parametrize('ecc', eccentricities)
@pytest.mark.parametrize('rotation', rotations)
def test_volume_ellipsoid(x, ecc, rotation):
    y = maj_from_min(x, ecc)
    im = ellipse(x, y, rotation)
    v = volume(im)
    real_v = (4 * np.pi * x * y * x) / 3
    err = np.abs(v - real_v) / real_v
    assert err < threshold
    assert np.isclose(v, real_v, rtol=threshold * real_v)
    return v, real_v


@pytest.mark.parametrize('x', radii)
@pytest.mark.parametrize('ecc', eccentricities)
@pytest.mark.parametrize('rotation', rotations)
def test_approximation(x, ecc, rotation):
    y = maj_from_min(x, ecc)
    im = ellipse(x, y, rotation)
    min_ax, maj_ax = min_maj_approximation(im)
    assert np.allclose([min_ax, maj_ax], [x, y],
                       rtol=threshold * min(np.array([x, y])))


@pytest.mark.parametrize('x', radii)
@pytest.mark.parametrize('ecc', eccentricities)
@pytest.mark.parametrize('rotation', rotations)
def test_roundness(x, ecc, rotation):
    y = maj_from_min(x, ecc)
    real_ecc = np.sqrt(y ** 2 - x ** 2) / y
    im = ellipse(x, y, rotation)
    e = eccentricity(im)
    assert np.isclose(real_ecc, e, rtol=threshold * real_ecc)

