#!/usr/bin/env python3
"""
Image: Loads images and registers them.

Image instances loads images from a specified directory into an object that
also contains image properties such as name and metadata.  Pixels from images
are stored in dask arrays; the standard way is to store them in 5-dimensional
arrays: T(ime point), C(channel), Z(-stack), Y, X.

This module consists of a base Image class (BaseLocalImage).  ImageLocalOME
handles local OMERO images.  ImageDir handles cases in which images are split
into directories, with each time point and channel having its own image file.
ImageDummy is a dummy class for silent failure testing.
"""

import typing as t
from abc import ABC, abstractmethod, abstractproperty
from datetime import datetime
from importlib_resources import files
from pathlib import Path, PosixPath

import dask.array as da
import xmltodict
from dask.array.image import imread
from tifffile import TiffFile

from agora.io.metadata import dir_to_meta


def get_examples_dir():
    """Get examples directory which stores dummy image for tiler"""
    return files("aliby").parent.parent / "examples" / "tiler"


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
    Base Image class to set path and provide context management method.
    """

    _default_dimorder = "tczyx"

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
                self._meta["size_y"],
                self._meta["size_x"],
            ),
        )
        return self._rechunked_img

    @abstractmethod
    def get_data_lazy(self) -> da.Array:
        pass

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


class ImageDummy(BaseLocalImage):
    """
    Dummy Image class.

    ImageDummy mimics the other Image classes in such a way that it is accepted
    by Tiler.  The purpose of this class is for testing, in particular,
    identifying silent failures.  If something goes wrong, we should be able to
    know whether it is because of bad parameters or bad input data.

    For the purposes of testing parameters, ImageDummy assumes that we already
    know the tiler parameters before Image instances are instantiated.  This is
    true for a typical pipeline run.
    """

    def __init__(self, tiler_parameters: dict):
        """Builds image instance

        Parameters
        ----------
        tiler_parameters : dict
            Tiler parameters, in dict form. Following
            aliby.tile.tiler.TilerParameters, the keys are: "tile_size" (size of
            tile), "ref_channel" (reference channel for tiling), and "ref_z"
            (reference z-stack, 0 to choose a default).
        """
        self.ref_channel = tiler_parameters["ref_channel"]
        self.ref_z = tiler_parameters["ref_z"]

    # Goal: make Tiler happy.
    @staticmethod
    def pad_array(
        image_array: da.Array,
        dim: int,
        n_empty_slices: int,
        image_position: int = 0,
    ):
        """Extends a dimension in a dask array and pads with zeros

        Extends a dimension in a dask array that has existing content, then pads
        with zeros.

        Parameters
        ----------
        image_array : da.Array
            Input dask array
        dim : int
            Dimension in which to extend the dask array.
        n_empty_slices : int
            Number of empty slices to extend the dask array by, in the specified
            dimension/axis.
        image_position : int
            Position within the new dimension to place the input arary, default 0
            (the beginning).

        Examples
        --------
        ```
        extended_array = pad_array(
            my_da_array, dim = 2, n_empty_slices = 4, image_position = 1)
        ```
        Extends a dask array called `my_da_array` in the 3rd dimension
        (dimensions start from 0) by 4 slices, filled with zeros.  And puts the
        original content in slice 1 of the 3rd dimension
        """
        # Concats zero arrays with same dimensions as image_array, and puts
        # image_array as first element in list of arrays to be concatenated
        zeros_array = da.zeros_like(image_array)
        return da.concatenate(
            [
                *([zeros_array] * image_position),
                image_array,
                *([zeros_array] * (n_empty_slices - image_position)),
            ],
            axis=dim,
        )

    # Logic: We want to return a image instance
    def get_data_lazy(self) -> da.Array:
        """Return 5D dask array. For lazy-loading multidimensional tiff files. Dummy image."""
        examples_dir = get_examples_dir()
        # TODO: Make this robust to having multiple TIFF images, one for each z-section,
        # all falling under the same "pypipeline_unit_test_00_000001_Brightfield_*.tif"
        # naming scheme.  The aim is to create a multidimensional dask array that stores
        # the z-stacks.
        img_filename = "pypipeline_unit_test_00_000001_Brightfield_003.tif"
        img_path = examples_dir / img_filename
        # img is a dask array has three dimensions: z, x, y
        # TODO: Write a test to confirm this: If everything worked well,
        # z = 1, x = 1200, y = 1200
        img = imread(str(img_path))
        # Adds t & c dimensions
        img = da.reshape(
            img, (1, 1, img.shape[-3], img.shape[-2], img.shape[-1])
        )
        # Pads t, c, and z dimensions
        img = self.pad_array(
            img, dim=0, n_empty_slices=199
        )  # 200 timepoints total
        img = self.pad_array(img, dim=1, n_empty_slices=2)  # 3 channels
        img = self.pad_array(
            img, dim=2, n_empty_slices=4, image_position=self.ref_z
        )  # 5 z-stacks
        return img

    @property
    def name(self):
        pass

    @property
    def dimorder(self):
        pass


class ImageLocalOME(BaseLocalImage):
    """
    Local OMERO Image class.

    This is a derivative Image class. It fetches an image from OMEXML data format,
    in which a multidimensional tiff image contains the metadata.
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
    It inherits from BaseLocalImage so we only override methods that are critical.

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
