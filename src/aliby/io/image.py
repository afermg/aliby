#!/usr/bin/env python3

import typing as t
from datetime import datetime
from pathlib import Path, PosixPath

import dask.array as da
import numpy as np
import xmltodict
from agora.io.bridge import BridgeH5
from dask import delayed
from dask.array.image import imread
from omero.model import enums as omero_enums
from tifffile import TiffFile
from yaml import safe_load

from aliby.io.omero import BridgeOmero

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


def get_image_class(source: t.Union[str, int, t.Dict[str, str], PosixPath]):
    """
    Wrapper to pick the appropiate Image class depending on the source of data.
    """
    if isinstance(source, int):
        instatiator = Image
    elif isinstance(source, dict) or (
        isinstance(source, (str, PosixPath)) and Path(source).is_dir()
    ):
        instatiator = ImageDirectory
    elif isinstance(source, str) and Path(source).is_file():
        instatiator = ImageLocal
    else:
        raise Exception(f"Invalid data source at {source}")

    return instatiator


class ImageLocal:
    def __init__(self, path: str, dimorder=None):
        self.path = path
        self.image_id = str(path)

        meta = dict()
        try:
            with TiffFile(path) as f:
                self.meta = xmltodict.parse(f.ome_metadata)["OME"]

            for dim in self.dimorder:
                meta["size_" + dim.lower()] = int(
                    self.meta["Image"]["Pixels"]["@Size" + dim]
                )
                meta["channels"] = [
                    x["@Name"] for x in self.meta["Image"]["Pixels"]["Channel"]
                ]
                meta["name"] = self.meta["Image"]["@Name"]
                meta["type"] = self.meta["Image"]["Pixels"]["@Type"]

        except Exception as e:
            print("Metadata not found: {}".format(e))
            assert (
                self.dimorder or self.meta.get("dims") is not None
            ), "No dimensional info provided."

            # Mark non-existent dimensions for padding
            base = "TCZXY"
            self.base = base
            self.ids = [base.index(i) for i in dimorder]

            self._dimorder = dimorder

        self._meta = meta

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        for e in exc:
            if e is not None:
                print(e)
        return False

    @property
    def name(self):
        return self._meta["name"]

    @property
    def data(self):
        return self.get_data_lazy_local()

    @property
    def date(self):
        date_str = [
            x
            for x in self.meta["StructuredAnnotations"]["TagAnnotation"]
            if x["Description"] == "Date"
        ][0]["Value"]
        return datetime.strptime(date_str, "%d-%b-%Y")

    @property
    def dimorder(self):
        """Order of dimensions in image"""
        if not hasattr(self, "_dimorder"):
            self._dimorder = self.meta["Image"]["Pixels"]["@DimensionOrder"]
        return self._dimorder

    @dimorder.setter
    def dimorder(self, order: str):
        self._dimorder = order
        return self._dimorder

    @property
    def metadata(self):
        return self._meta

    def get_data_lazy_local(self) -> da.Array:
        """Return 5D dask array. For lazy-loading local multidimensional tiff files"""

        if not hasattr(self, "formatted_img"):
            if not hasattr(self, "ids"):  # Standard dimension order
                img = (imread(str(self.path))[0],)
            else:  # Custom dimension order, we rearrange the axes for compatibility
                img = imread(str(self.path))[0]
                for i, d in enumerate(self._dimorder):
                    self._meta["size_" + d.lower()] = img.shape[i]

                target_order = (
                    *self.ids,
                    *[
                        i
                        for i, d in enumerate(self.base)
                        if d not in self.dimorder
                    ],
                )
                reshaped = da.reshape(
                    img,
                    shape=(
                        *img.shape,
                        *[1 for _ in range(5 - len(self.dimorder))],
                    ),
                )
                img = da.moveaxis(
                    reshaped, range(len(reshaped.shape)), target_order
                )

            self._formatted_img = da.rechunk(
                img,
                chunks=(1, 1, 1, self._meta["size_y"], self._meta["size_x"]),
            )
        return self._formatted_img


class ImageDirectory(ImageLocal):
    """
    Image class for case where all images are split in one or multiple folders with time-points and channels as independent files.
    It inherits from Imagelocal so we only override methods that are critical.

    Assumptions:
    - Assumes individual folders for individual channels. If only one path provided it assumes it to be brightfield.
    - Assumes that images are flat.
    - Provides Dimorder as TCZYX
    """

    def __init__(self, path: t.Union[str, t.Dict[str, str]]):
        if isinstance(path, str):
            path = {"Brightfield": path}

        self.path = path
        self.image_id = str(path)
        self._meta = dict(channels=path.keys(), name=list(path.values())[0])

        # Parse name if necessary
        # Build lazy-loading array using dask?

    def get_data_lazy_local(self) -> da.Array:
        """Return 5D dask array. For lazy-loading local multidimensional tiff files"""

        img = da.stack([imread(v) for v in self.path.values()])
        if (
            img.ndim < 5
        ):  # Files do not include z-stack: Add and swap with time dimension.
            img = da.stack((img,)).swapaxes(0, 2)

            # TODO check whether x and y swap is necessary

        # Use images to redefine axes
        for i, dim in enumerate(("t", "c", "z", "y", "x")):
            self._meta["size_" + dim] = img.shape[i]

        self._formatted_img = da.rechunk(
            img,
            chunks=(1, 1, 1, self._meta["size_y"], self._meta["size_x"]),
        )
        return self._formatted_img


class Image(BridgeOmero):
    """
    Loads images from OMERO and gives access to the data and metadata.
    """

    def __init__(self, image_id: int, **server_info):
        """
        Establishes the connection to the OMERO server via the Argo
        base class.

        Parameters
        ----------
        image_id: integer
        server_info: dictionary
            Specifies the host, username, and password as strings
        """
        self.ome_id = image_id
        super().__init__(**server_info)

    def init_interface(self, ome_id: int):
        self.set_id(ome_id)
        self.ome_class = self.conn.getObject("Image", ome_id)

    @classmethod
    def from_h5(
        cls,
        filepath: t.Union[str, PosixPath],
    ):
        """Instatiate Image from a hdf5 file.

        Parameters
        ----------
        cls : Image
            Image class
        filepath : t.Union[str, PosixPath]
            Location of hdf5 file.

        Examples
        --------
        FIXME: Add docs.

        """
        # metadata = load_attributes(filepath)
        bridge = BridgeH5(filepath)
        image_id = bridge.meta_h5["image_id"]
        # server_info = safe_load(bridge.meta_h5["parameters"])["general"][
        #     "server_info"
        # ]
        return cls(image_id, **cls.server_info_from_h5(filepath))

    @property
    def name(self):
        return self.ome_class.getName()

    @property
    def data(self):
        return get_data_lazy(self.ome_class)

    @property
    def metadata(self):
        """
        Store metadata saved in OMERO: image size, number of time points,
        labels of channels, and image name.
        """
        meta = dict()
        meta["size_x"] = self.ome_class.getSizeX()
        meta["size_y"] = self.ome_class.getSizeY()
        meta["size_z"] = self.ome_class.getSizeZ()
        meta["size_c"] = self.ome_class.getSizeC()
        meta["size_t"] = self.ome_class.getSizeT()
        meta["channels"] = self.ome_class.getChannelLabels()
        meta["name"] = self.ome_class.getName()
        return meta


class UnsafeImage(Image):
    """
    Loads images from OMERO and gives access to the data and metadata.
    This class is a temporary solution while we find a way to use
    context managers inside napari. It risks resulting in zombie connections
    and producing freezes in an OMERO server.

    """

    def __init__(self, image_id, **server_info):
        """
        Establishes the connection to the OMERO server via the Argo
        base class.

        Parameters
        ----------
        image_id: integer
        server_info: dictionary
            Specifies the host, username, and password as strings
        """
        super().__init__(image_id, **server_info)
        self.create_gate()
        self.init_wrapper()

    @property
    def data(self):
        try:
            return get_data_lazy(self.ome_class)
        except Exception as e:
            print(f"ERROR: Failed fetching image from server: {e}")
            self.conn.connect(False)


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
