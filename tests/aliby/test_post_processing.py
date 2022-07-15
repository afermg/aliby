import pytest
import unittest

import matplotlib.pyplot as plt
import numpy as np
import skimage.morphology as morph
from scipy import ndimage
from skimage import draw

# from aliby.post_processing import (
#     circle_outline,
#     conical,
#     ellipse_perimeter,
#     union_of_spheres,
#     volume_of_sphere,
# )


@pytest.mark.skip(
    reason="No longer usable, post_processing unused inside aliby. Kept temporarily"
)
class VolumeEstimation(unittest.TestCase):
    def test_conical(self):
        radius = np.random.choice(range(60, 100))
        con = conical(circle_outline(radius))
        b_sum = morph.ball(radius).sum()
        # Close relative to the value.
        print(radius, con, b_sum)
        self.assertAlmostEqual(abs(con - b_sum) / b_sum, 0, delta=0.10)

    @pytest.mark.skip(
        reason="No longer usable, post_processing unused inside aliby. Kept temporarily"
    )
    def test_conical_ellipse(self):
        e = ellipse_perimeter(4, 5)
        con = conical(e)
        true = draw.ellipsoid_stats(4, 5, 4)[0]
        print(con, true)

    @pytest.mark.skip(
        reason="No longer usable, post_processing unused inside aliby. Kept temporarily"
    )
    def test_sphere_error(self):
        radii = range(3, 30)
        con = [conical(circle_outline(radius)) for radius in radii]
        spheres = [union_of_spheres(circle_outline(r)) for r in radii]
        true = [4 * (r**3) * np.pi / 3 for r in radii]
        mVol = [
            4 / 3 * np.pi * np.sqrt(morph.disk(radius).sum() / np.pi) ** 3
            for radius in radii
        ]
        plt.scatter(true, con, label="Conical")
        plt.scatter(true, spheres, label="Spheres")
        plt.scatter(true, mVol, label="mVol")
        plt.plot(true, true, "k-")
        plt.xlabel("Analytical")
        plt.ylabel("Estimated")
        plt.title("Disk")
        plt.legend()
        # plt.show()

    @pytest.mark.skip(
        reason="No longer usable, post_processing unused inside aliby. Kept temporarily"
    )
    def test_ellipse_error(self):
        x_radii = range(3, 30)
        y_radii = [np.ceil(2.5 * r) for r in x_radii]
        ellipses = [
            ellipse_perimeter(x_r, y_r) for x_r, y_r in zip(x_radii, y_radii)
        ]
        con = [conical(ellipse) for ellipse in ellipses]
        spheres = [union_of_spheres(ellipse) for ellipse in ellipses]
        mVol = np.array(
            [
                4
                / 3
                * np.pi
                * np.sqrt(ndimage.binary_fill_holes(ellipse).sum() / np.pi)
                ** 3
                for ellipse in ellipses
            ]
        )
        true = np.array(
            [
                4 * np.pi * x_r * y_r * x_r / 3
                for x_r, y_r in zip(x_radii, y_radii)
            ]
        )
        plt.scatter(true, con, label="Conical")
        plt.scatter(true, spheres, label="Spheres")
        plt.scatter(true, mVol, label="mVol")
        plt.plot(true, true, "k-")
        plt.xlabel("Analytical")
        plt.ylabel("Estimated")
        plt.title("Ellipse")
        plt.legend()
        # plt.show()

    @pytest.mark.skip(
        reason="No longer usable, post_processing unused inside aliby. Kept temporarily"
    )
    def test_minor_major_error(self):
        r = np.random.choice(list(range(3, 30)))
        x_radii = np.linspace(r / 3, r, 20)
        y_radii = r**2 / x_radii

        ellipses = [
            ellipse_perimeter(x_r, y_r) for x_r, y_r in zip(x_radii, y_radii)
        ]
        con = np.array([conical(ellipse) for ellipse in ellipses])
        spheres = np.array([union_of_spheres(ellipse) for ellipse in ellipses])
        mVol = np.array(
            [
                4
                / 3
                * np.pi
                * np.sqrt(ndimage.binary_fill_holes(ellipse).sum() / np.pi)
                ** 3
                for ellipse in ellipses
            ]
        )

        true = np.array(
            [
                4 * np.pi * x_r * y_r * x_r / 3
                for x_r, y_r in zip(x_radii, y_radii)
            ]
        )

        ratio = y_radii / x_radii
        plt.scatter(ratio, con / true, label="Conical")
        plt.scatter(ratio, spheres / true, label="Spheres")
        plt.scatter(ratio, mVol / true, label="mVol")
        plt.xlabel("Major/Minor")
        plt.ylabel("Estimated / Analytical")
        plt.title(f"Error by circularity, r = {r}")
        plt.legend()
        # plt.show()


if __name__ == "__main__":
    unittest.main()
