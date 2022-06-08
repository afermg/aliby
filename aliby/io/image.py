#!/usr/bin/env python3

from pathlib import Path
from datetime import datetime

import xmltodict
from tifffile import TiffFile

# dask extends numpy to multi-core machines and distributed clusters
# and allows data to be stored that is larger than the RAM by
# sharing between RAM and a hard disk
import dask.array as da
from dask.array.image import imread

from aliby.io.omero import Argo, get_data_lazy


class ImageLocal:
    def __init__(self, path: str, *args, **kwargs):
        self.path = path
        self.image_id = str(path)

        with TiffFile(path) as f:
            self.meta = xmltodict.parse(f.ome_metadata)["OME"]

        meta = dict()
        try:
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
            raise e

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
        return self.meta["Image"]["Pixels"]["@DimensionOrder"]

    @property
    def metadata(self):
        return self._meta

    def get_data_lazy_local(self) -> da.Array:
        """Return 5D dask array. For lazy-loading local multidimensional tiff files"""
        return da.rechunk(
            imread(str(self.path))[0],
            chunks=(1, 1, 1, self._meta["size_y"], self._meta["size_x"]),
        )


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

    # version with local file processing
    def get_data_lazy_local(path: str) -> da.Array:
        """
        For lazy-loading - loading on demand only -- local,
        multidimensional tiff files.

        Parameters
        ----------
        path: string

        Returns
        -------
            5D dask array
        """
        return da.from_delayed(imread(str(path))[0], shape=())

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
