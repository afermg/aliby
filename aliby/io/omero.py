import dask.array as da
import numpy as np
from dask import delayed
from omero.gateway import BlitzGateway
from omero.model import enums as omero_enums

# convert OMERO definitions into numpy types
PIXEL_TYPES = {
    omero_enums.PixelsTypeint8: np.int8,
    omero_enums.PixelsTypeuint8: np.uint8,
    omero_enums.PixelsTypeint16: np.int16,
    omero_enums.PixelsTypeuint16: np.uint16,
    omero_enums.PixelsTypeint32: np.int32,
    omero_enums.PixelsTypeuint32: np.uint32,
    omero_enums.PixelsTypefloat: np.float32,
    omero_enums.PixelsTypedouble: np.float64,
}


class Argo:
    """
    Base class to interact with OMERO.
    See
    https://docs.openmicroscopy.org/omero/5.6.0/developers/Python.html
    """

    def __init__(
        self,
        host="islay.bio.ed.ac.uk",
        username="upload",
        password="***REMOVED***",
    ):
        """
        Parameters
        ----------
        host : string
            web address of OMERO host
        username: string
        password : string
        """
        self.conn = None
        self.host = host
        self.username = username
        self.password = password

    # standard method required for Python's with statement
    def __enter__(self):
        self.conn = BlitzGateway(
            host=self.host, username=self.username, passwd=self.password
        )
        self.conn.connect()
        self.conn.c.enableKeepAlive(60)

        return self

    # standard method required for Python's with statement
    def __exit__(self, *exc):
        for e in exc:
            if e is not None:
                print(e)

        self.conn.close()
        return False


###


def get_data_lazy(image) -> da.Array:
    """
    Get 5D dask array, with delayed reading from OMERO image.
    """
    nt, nc, nz, ny, nx = [getattr(image, f"getSize{x}")() for x in "TCZYX"]
    pixels = image.getPrimaryPixels()
    dtype = PIXEL_TYPES.get(pixels.getPixelsType().value, None)
    # using dask
    get_plane = delayed(lambda idx: pixels.getPlane(*idx))

    def get_lazy_plane(zct):
        return da.from_delayed(get_plane(zct), shape=(ny, nx), dtype=dtype)

    # 5D stack: TCZXY
    t_stacks = []
    for t in range(nt):
        c_stacks = []
        for c in range(nc):
            z_stack = []
            for z in range(nz):
                z_stack.append(get_lazy_plane((z, c, t)))
            c_stacks.append(da.stack(z_stack))
        t_stacks.append(da.stack(c_stacks))

    return da.stack(t_stacks)
