#!/usr/bin/env python3

from aliby.experiment import get_data_lazy_local
from pathlib import Path
from tifffile import TiffFile
from pathlib import Path
import xmltodict
from aliby.experiment import get_data_lazy
from aliby.io.omero import Argo


class ImageLocal:
    def __init__(self, path: str):
        self.path = path
        self.image_id = str(path)

        try:
            with TiffFile(fpath) as f:
                self.meta = xmltodict.parse(f.ome_metadata)["OME"]
            meta = dict()
            for dim in self.dimorder:
                meta["size_" + dim.lower()] = int(
                    self.meta["Image"]["Pixels"]["@Size" + dim]
                )
            meta["channels"] = [
                x["@Name"] for x in self.meta["Image"]["Pixels"]["Channel"]
            ]
            meta["name"] = self.meta["Image"]["@Name"]
            meta["type"] = self.meta["Image"]["Pixels"]["@Type"]

            self._meta = meta

        except Exception as e:
            # TODO implement default meta when no image metadata
            print("Could not fetch metadata: {}".format(e))

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
            # chunks="auto",
        )


class Image(Argo):
    """"""

    def __init__(self, image_id, **server_info):
        super().__init__(**server_info)
        self.image_id = image_id
        self._image_wrap = None

    @property
    def image_wrap(self):
        # TODO check that it is alive/ connected
        if self._image_wrap is None:
            self._image_wrap = self.conn.getObject("Image", self.image_id)
        return self._image_wrap

    # Version with local file processing
    def get_data_lazy_local(path: str) -> da.Array:
        """Return 5D dask array. For lazy-loading local multidimensional tiff files"""
        return da.from_delayed(imread(str(path))[0], shape=())

    @property
    def name(self):
        return self.image_wrap.getName()

    @property
    def data(self):
        return get_data_lazy(self.image_wrap)

    @property
    def metadata(self):
        meta = dict()
        meta["size_x"] = self.image_wrap.getSizeX()
        meta["size_y"] = self.image_wrap.getSizeY()
        meta["size_z"] = self.image_wrap.getSizeZ()
        meta["size_c"] = self.image_wrap.getSizeC()
        meta["size_t"] = self.image_wrap.getSizeT()
        meta["channels"] = self.image_wrap.getChannelLabels()
        meta["name"] = self.image_wrap.getName()
        return meta
