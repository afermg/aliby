#!/usr/bin/env python3

from pathlib import Path
from datetime import datetime

import xmltodict
from tifffile import TiffFile

import dask.array as da
from dask.array.image import imread

from aliby.io.omero import Argo, get_data_lazy


class ImageLocal:
    def __init__(self, path: str, dimorder=None, *args, **kwargs):
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
            assert "dims" != None, "No dimensional info provided."

            # Mark non-existent dimensions for padding
            base = "TCZXY"
            self.base = base
            self.ids = {base.index(i) for i in dimorder}

            self._dimorder = dimorder

        self._meta = meta

    def __enter__(self):
        return self

    def __exit__(self, *exc):
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


class Image(Argo):
    """
    Loads images from OMERO and gives access to the data and metadata.
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
        super().__init__(**server_info)
        self.image_id = image_id
        # images from OMERO
        self._image_wrap = None

    @property
    def image_wrap(self):
        """
        Get images from OMERO
        """
        if self._image_wrap is None:
            # get images using OMERO
            self._image_wrap = self.conn.getObject("Image", self.image_id)
        return self._image_wrap

    @property
    def name(self):
        return self.image_wrap.getName()

    @property
    def data(self):
        return get_data_lazy(self.image_wrap)

    @property
    def metadata(self):
        """
        Store metadata saved in OMERO: image size, number of time points,
        labels of channels, and image name.
        """
        meta = dict()
        meta["size_x"] = self.image_wrap.getSizeX()
        meta["size_y"] = self.image_wrap.getSizeY()
        meta["size_z"] = self.image_wrap.getSizeZ()
        meta["size_c"] = self.image_wrap.getSizeC()
        meta["size_t"] = self.image_wrap.getSizeT()
        meta["channels"] = self.image_wrap.getChannelLabels()
        meta["name"] = self.image_wrap.getName()
        return meta
