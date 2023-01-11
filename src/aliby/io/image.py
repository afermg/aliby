#!/usr/bin/env python3

import typing as t
from abc import ABC, abstractproperty
from datetime import datetime
from pathlib import Path, PosixPath

import dask.array as da
import xmltodict
from dask.array.image import imread
from tifffile import TiffFile

from agora.io.metadata import dir_to_meta


def get_image_class(source: t.Union[str, int, t.Dict[str, str], PosixPath]):
    """
    Wrapper to pick the appropiate Image class depending on the source of data.
    """
    if isinstance(source, int):
        from aliby.io.omero import Image

        instatiator = Image
    elif isinstance(source, dict) or (
        isinstance(source, (str, PosixPath)) and Path(source).is_dir()
    ):
        instatiator = ImageDir
    elif isinstance(source, str) and Path(source).is_file():
        instatiator = ImageLocalOME
    else:
        raise Exception(f"Invalid data source at {source}")

    return instatiator


class BaseLocalImage(ABC):
    """
    Base class to set path and provide context management method.
    """

    _default_dimorder = "tczxy"

    def __init__(self, path: t.Union[str, PosixPath]):
        # If directory, assume contents are naturally sorted
        self.path = Path(path)

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        for e in exc:
            if e is not None:
                print(e)
        return False

    def rechunk_data(self, img):
        # Format image using x and y size from metadata.

        self._rechunked_img = da.rechunk(
            img,
            chunks=(
                1,
                1,
                1,
                self._meta["size_x"],
                self._meta["size_y"],
            ),
        )
        return self._rechunked_img

    @abstractproperty
    def name(self):
        pass

    @abstractproperty
    def dimorder(self):
        pass

    @property
    def data(self):
        return self.get_data_lazy()

    @property
    def metadata(self):
        return self._meta


class ImageLocalOME(BaseLocalImage):
    """
    Fetch image from OMEXML data format, in which a multidimensional tiff image contains the metadata.
    """

    def __init__(self, path: str, dimorder=None):
        super().__init__(path)
        self._id = str(path)

        meta = dict()
        try:
            with TiffFile(path) as f:
                self._meta = xmltodict.parse(f.ome_metadata)["OME"]

            for dim in self.dimorder:
                meta["size_" + dim.lower()] = int(
                    self._meta["Image"]["Pixels"]["@Size" + dim]
                )
                meta["channels"] = [
                    x["@Name"]
                    for x in self._meta["Image"]["Pixels"]["Channel"]
                ]
                meta["name"] = self._meta["Image"]["@Name"]
                meta["type"] = self._meta["Image"]["Pixels"]["@Type"]

        except Exception as e:  # Images not in OMEXML

            print("Warning:Metadata not found: {}".format(e))
            print(
                f"Warning: No dimensional info provided. Assuming {self._default_dimorder}"
            )

            # Mark non-existent dimensions for padding
            self.base = self._default_dimorder
            # self.ids = [self.index(i) for i in dimorder]

            self._dimorder = base

            self._meta = meta

    @property
    def name(self):
        return self._meta["name"]

    @property
    def date(self):
        date_str = [
            x
            for x in self._meta["StructuredAnnotations"]["TagAnnotation"]
            if x["Description"] == "Date"
        ][0]["Value"]
        return datetime.strptime(date_str, "%d-%b-%Y")

    @property
    def dimorder(self):
        """Order of dimensions in image"""
        if not hasattr(self, "_dimorder"):
            self._dimorder = self._meta["Image"]["Pixels"]["@DimensionOrder"]
        return self._dimorder

    @dimorder.setter
    def dimorder(self, order: str):
        self._dimorder = order
        return self._dimorder

    def get_data_lazy(self) -> da.Array:
        """Return 5D dask array. For lazy-loading  multidimensional tiff files"""

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

        return self.rechunk_data(img)


class ImageDir(BaseLocalImage):
    """
    Image class for the case in which all images are split in one or
    multiple folders with time-points and channels as independent files.
    It inherits from Imagelocal so we only override methods that are critical.

    Assumptions:
    - One folders per position.
    - Images are flat.
    - Channel, Time, z-stack and the others are determined by filenames.
    - Provides Dimorder as it is set in the filenames, or expects order during instatiation
    """

    def __init__(self, path: t.Union[str, PosixPath], **kwargs):
        super().__init__(path)
        self.image_id = str(self.path.stem)

        self._meta = dir_to_meta(self.path)

    def get_data_lazy(self) -> da.Array:
        """Return 5D dask array. For lazy-loading local multidimensional tiff files"""

        img = imread(str(self.path / "*.tiff"))

        # If extra channels, pick the first stack of the last dimensions

        while len(img.shape) > 3:
            img = img[..., 0]

        if self._meta:
            self._meta["size_x"], self._meta["size_y"] = img.shape[-2:]

            # Reshape using metadata
            # img = da.reshape(img, (*self._meta, *img.shape[1:]))
            img = da.reshape(img, self._meta.values())
            original_order = [
                i[-1] for i in self._meta.keys() if i.startswith("size")
            ]
            # Swap axis to conform with normal order
            target_order = [
                self._default_dimorder.index(x) for x in original_order
            ]
            img = da.moveaxis(
                img,
                list(range(len(original_order))),
                target_order,
            )
            pixels = self.rechunk_data(img)
        return pixels

    @property
    def name(self):
        return self.path.stem

    @property
    def dimorder(self):
        # Assumes only dimensions start with "size"
        return [
            k.split("_")[-1] for k in self._meta.keys() if k.startswith("size")
        ]
