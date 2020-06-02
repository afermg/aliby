import matplotlib.pyplot as plt
import numpy as np
import skimage.morphology as morph
from skimage import draw
import unittest

from core.post_processing import conical, ellipse_perimeter, union_of_spheres


class VolumeEstimation(unittest.TestCase):
    def test_conical(self):
        radius = np.random.choice(range(60, 100))
        con = conical(morph.disk(radius))
        b_sum = morph.ball(radius).sum()
        # Close relative to the value.
        print(radius, con, b_sum)
        self.assertAlmostEqual(abs(con - b_sum)/b_sum, 0, delta=0.10)

    def test_conical_ellipse(self):
        e = ellipse_perimeter(4, 5)
        con = conical(e)
        true = draw.ellipsoid_stats(4, 4, 5)[0]
        print(con, true)

    def test_sphere_error(self):
        radii = range(3, 30)
        con = [conical(morph.disk(radius)) for radius in radii]
        spheres = [union_of_spheres(ellipse_perimeter(r, r)) for r in radii]
        true = [4*(r**3)*np.pi/3 for r in radii]
        mVol = [4 / 3 * np.pi * np.sqrt(morph.disk(radius).sum() / np.pi)**3
                for radius in radii]
        plt.scatter(true, con, label='Conical')
        plt.scatter(true, spheres, label='Spheres')
        plt.scatter(true, mVol, label='mVol')
        plt.plot(true, true, 'k-' )
        plt.xlabel("Analytical")
        plt.ylabel("Estimated")
        plt.title("Disk")
        plt.legend()
        plt.show()


    def test_ellipse_error(self):
        x_radii = range(3, 30)
        y_radii = [np.ceil(1.2*r) for r in x_radii]
        ellipses = [ellipse_perimeter(x_r, y_r)
                    for x_r, y_r in zip(x_radii, y_radii)]
        con = [conical(ellipse) for ellipse in ellipses]
        spheres = [union_of_spheres(ellipse) for ellipse in ellipses]
        mVol = [(4 * np.pi * np.sqrt(ellipse.sum() / np.pi)**3) / 3
                for ellipse in ellipses]
        true = [draw.ellipsoid_stats(x_r, y_r, x_r)[0]
                for x_r, y_r in zip(x_radii, y_radii)]
        plt.scatter(true, con, label='Conical')
        plt.scatter(true, spheres, label='Spheres')
        plt.scatter(true, mVol, label='mVol')
        plt.plot(true, true, 'k-')
        plt.xlabel("Analytical")
        plt.ylabel("Estimated")
        plt.title("Ellipse")
        plt.legend()
        plt.show()

    def test_mixed_error(self):
        pass





if __name__ == '__main__':
    unittest.main()
