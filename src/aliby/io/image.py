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
from pathlib import Path

import dask.array as da
import numpy as np
import xmltodict
import zarr
from aliby.io.omero import Image
from dask.array.image import imread

try:
    from importlib_resources import files
except ModuleNotFoundError:
    from importlib.resources import files

from agora.io.metadata import parse_microscopy_logs
from tifffile import TiffFile


def instantiate_image(
    source: t.Union[str, int, t.Dict[str, str], Path], **kwargs
):
    """
    Instantiate the image.

    Parameters
    ----------
    source : t.Union[str, int, t.Dict[str, str], Path]
        Image identifier

    Examples
    --------
    image_path = "path/to/image"
    with instantiate_image(image_path) as img:
        print(imz.data, img.metadata)
    """
    return dispatch_image(source)(source, **kwargs)


def dispatch_image(source: t.Union[str, int, t.Dict[str, str], Path]):
    """Pick the appropriate Image class for the source of data."""
    if isinstance(source, (int, np.int64)):
        instantiator = Image
    elif isinstance(source, dict) or (
        isinstance(source, (str, Path)) and Path(source).is_dir()
    ):
        # zarr files are considered directories
        if Path(source).suffix == ".zarr":
            instantiator = ImageZarr
        else:
            instantiator = ImageDir
    elif isinstance(source, (str, Path)) and Path(source).is_file():
        instantiator = ImageLocalOME
    else:
        raise Exception(f"Invalid data source at {source}.")
    return instantiator


def files_to_image_sizes(path: Path, suffix="tiff"):
    """Deduce image sizes from the naming convention of tiff files."""
    filenames = list(path.glob(f"*.{suffix}"))
    try:
        # deduce order from filenames
        dimorder = "".join(
            map(lambda x: x[0], filenames[0].stem.split("_")[1:])
        )
        dim_value = list(
            map(
                lambda f: filename_to_dict_indices(f.stem),
                path.glob("*.tiff"),
            )
        )
        maxes = [max(map(lambda x: x[dim], dim_value)) for dim in dimorder]
        mins = [min(map(lambda x: x[dim], dim_value)) for dim in dimorder]
        dim_shapes = [
            max_val - min_val + 1 for max_val, min_val in zip(maxes, mins)
        ]
        meta = {
            "size_" + dim: shape for dim, shape in zip(dimorder, dim_shapes)
        }
    except Exception as e:
        print("Warning: files_to_image_sizes failed." f"\nError: {e}")
        meta = {}
    return meta


def filename_to_dict_indices(stem: str):
    """Split string into a dict."""
    return {
        dim_number[0]: int(dim_number[1:])
        for dim_number in stem.split("_")[1:]
    }


class BaseLocalImage(ABC):
    """Set path and provide method for context management."""

    # default image order
    default_dimorder = "tczyx"

    def __init__(self, path: t.Union[str, Path]):
        # If directory, assume contents are naturally sorted
        self.path = Path(path)

    def __enter__(self):
        """For entering 'with' statements."""
        return self

    def __exit__(self, *exc):
        """For exiting from 'with' statements."""
        for e in exc:
            # print exceptions - do not crash
            if e is not None:
                print(e)
        return False

    def rechunk_data(self, img):
        """Format image using x and y size from metadata."""
        self._rechunked_img = da.rechunk(
            img,
            chunks=(
                1,
                1,
                1,
                self.meta["size_y"],
                self.meta["size_x"],
            ),
        )
        return self._rechunked_img

    @property
    def data(self):
        """Get data."""
        return self.get_data_lazy()

    @property
    def metadata(self):
        """Get metadata."""
        return self.meta

    def set_meta(self):
        """Load metadata using parser dispatch."""
        logs_metadata = parse_microscopy_logs(self.path)
        if logs_metadata is None:
            # try to deduce metadata
            self.meta = files_to_image_sizes(self.path)
        else:
            self.meta = logs_metadata

    @abstractmethod
    def get_data_lazy(self) -> da.Array:
        """Define in child class."""
        pass

    @abstractproperty
    def name(self):
        """Define in child class."""
        pass

    @abstractproperty
    def dimorder(self):
        """Define in child class."""
        pass


class ImageLocalOME(BaseLocalImage):
    """
    Local OMERO Image class.

    This is a derivative Image class. It fetches an image from
    OMEXML data format, in which a multidimensional tiff image
    contains the metadata.
    """

    def __init__(self, path: str, dimorder=None, **kwargs):
        """Initialise using file name."""
        super().__init__(path)
        self._id = str(path)
        self.set_meta(str(path))

    def set_meta(self, path):
        """Get metadata from the associated tiff file."""
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
            # images not in OMEXML
            print("Warning:Metadata not found: {}".format(e))
            print(
                "Warning: No dimensional info provided. "
                f"Assuming {self.default_dimorder}"
            )
            # mark non-existent dimensions for padding
            self.base = self.default_dimorder
            self.dimorder = self.base
            self.meta = meta

    @property
    def name(self):
        return self.meta["name"]

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
        """Return order of dimensions in the image."""
        if not hasattr(self, "dimorder"):
            self.dimorder = self.meta["Image"]["Pixels"]["@DimensionOrder"]
        return self.dimorder

    @dimorder.setter
    def dimorder(self, order: str):
        self.dimorder = order
        return self.dimorder

    def get_data_lazy(self) -> da.Array:
        """Return 5D dask array via lazy-loading of tiff files."""
        if not hasattr(self, "formatted_img"):
            if not hasattr(self, "ids"):
                # standard order of image dimensions
                img = (imread(str(self.path))[0],)
            else:
                # bespoke order, so rearrange axes for compatibility
                img = imread(str(self.path))[0]
                for i, d in enumerate(self.dimorder):
                    self.meta["size_" + d.lower()] = img.shape[i]
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
    Standard image class for tiff files.

    Image class for the case in which all images are split in one or
    multiple folders with time-points and channels as independent files.
    It inherits from BaseLocalImage so we only override methods that are critical.

    Assumptions:
    - One folder per position.
    - Images are flat.
    - Channel, Time, z-stack and the others are determined by filenames.
    - Provides Dimorder as it is set in the filenames, or expects order
    """

    def __init__(self, path: t.Union[str, Path], **kwargs):
        """Initialise using file name."""
        super().__init__(path)
        self.image_id = str(self.path.stem)
        self.meta = files_to_image_sizes(self.path)

    def get_data_lazy(self) -> da.Array:
        """Return 5D dask array."""
        img = imread(str(self.path / "*.tiff"))
        # If extra channels, pick the first stack of the last dimensions
        while len(img.shape) > 3:
            img = img[..., 0]
        if self.meta:
            self.meta["size_x"], self.meta["size_y"] = img.shape[-2:]
            # Reshape using metadata
            img = da.reshape(img, self.meta.values())
            original_order = [
                i[-1] for i in self.meta.keys() if i.startswith("size")
            ]
            # Swap axis to conform with normal order
            target_order = [
                self.default_dimorder.index(x) for x in original_order
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
        """Return name of image directory."""
        return self.path.stem

    @property
    def dimorder(self):
        # Assumes only dimensions start with "size"
        return [
            k.split("_")[-1] for k in self.meta.keys() if k.startswith("size")
        ]


class ImageZarr(BaseLocalImage):
    """
    Read zarr compressed files.

    These files are generated by the script
    skeletons/scripts/howto_omero/convert_clone_zarr_to_tiff.py
    """

    def __init__(self, path: t.Union[str, Path], **kwargs):
        """Initialise using file name."""
        super().__init__(path)
        self.set_meta()
        try:
            self._img = zarr.open(self.path)
            self.add_size_to_meta()
        except Exception as e:
            print(f"ImageZarr: Could not add size info to metadata: {e}.")

    def get_data_lazy(self) -> da.Array:
        """Return 5D dask array for lazy-loading local multidimensional zarr files."""
        return self._img

    def add_size_to_meta(self):
        """Add shape of image array to metadata."""
        self.meta.update(
            {
                f"size_{dim}": shape
                for dim, shape in zip(self.dimorder, self._img.shape)
            }
        )

    @property
    def name(self):
        """Return name of zarr directory."""
        return self.path.stem

    @property
    def dimorder(self):
        """Impose a hard-coded order of dimensions based on the zarr compression script."""
        return "TCZYX"
