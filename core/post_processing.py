"""
Post-processing utilities
"""
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from scipy import ndimage
from skimage.morphology import erosion, ball
from skimage import measure, draw, feature

def ellipse_perimeter(x, y):
    im_shape = int(2*max(x, y) + 1)
    img = np.zeros((im_shape, im_shape), dtype=np.uint8)
    rr, cc = draw.ellipse_perimeter(int(im_shape//2), int(im_shape//2),
                                    int(x), int(y))
    img[rr, cc] = 1
    return img

def capped_cylinder(x, y):
    max_size = (y + 2*x + 2)
    pixels = np.zeros((max_size, max_size))

    rect_start = ((max_size-x)//2, x + 1)
    rr, cc = draw.rectangle_perimeter(rect_start, extent=(x, y),
                                     shape=(max_size, max_size))
    pixels[rr, cc] = 1
    circle_centres = [(max_size//2 - 1, x),
                      (max_size//2 - 1, max_size - x - 1 )]
    for r, c in circle_centres:
        rr, cc = draw.circle_perimeter(r, c, (x + 1)//2,
                                       shape=(max_size, max_size))
        pixels[rr, cc] = 1
    pixels = ndimage.morphology.binary_fill_holes(pixels)
    pixels ^= erosion(pixels)
    return pixels

# Volume estimation
def union_of_spheres(outline, debug=False):
    filled = ndimage.binary_fill_holes(outline)
    nearest_neighbor = ndimage.morphology.distance_transform_edt(
        outline == 0) * filled
    voxels = np.zeros((filled.shape[0], filled.shape[1], max(filled.shape)))
    for x,y in zip(*np.where(filled)):
        radius = nearest_neighbor[(x,y)]
        if radius > 0:
            b = ball(radius)
            centre_b = ndimage.measurements.center_of_mass(b)

            I,J,K = np.ogrid[:b.shape[0], :b.shape[1], :b.shape[2]]
            c_z = voxels.shape[2]//2
            voxels[I + int(x - centre_b[0]), J + int(y - centre_b[1]),
                   K + int(c_z - centre_b[2])] += b
    if debug:
        verts, faces, normals, values = measure.marching_cubes_lewiner(
            voxels, 0)
        fig = plt.figure(figsize=(10, 10))
        ax = fig.add_subplot(111, projection='3d')
        mesh = Poly3DCollection(verts[faces])
        mesh.set_edgecolor('k')
        ax.add_collection3d(mesh)
        ax.set_xlim(0, filled.shape[0])
        ax.set_ylim(0, filled.shape[1])
        ax.set_zlim(0, max(filled.shape))
        plt.tight_layout()
        plt.show()
    return voxels.astype(bool).sum()

def conical(outline, debug=False, selem=None):
    filled = ndimage.binary_fill_holes(outline)
    cone = [filled]
    while filled.sum() > 0 :
        filled = erosion(filled, selem=selem)
        cone.append(filled)
        if debug:
            plt.imshow(filled)
            plt.show()
    cone = np.dstack(cone)
    return 4 * np.sum(cone) #* 0.95 #To make the circular version work

def volume(outline, method='spheres'):
    if method=='conical':
        return conical(outline)
    elif method=='spheres':
        return union_of_spheres(outline)
    else:
        raise ValueError(f"Method {method} not implemented.")

def circularity(outline):
    pass