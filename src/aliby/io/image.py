#!/usr/bin/env python3
"""
Image: Loads images and registers them.

Image instances loads images from a specified directory into an object that
also contains image properties such as name and metadata.  Pixels from images
are stored in dask arrays; the standard way is to store them in 5-dimensional
arrays: T(ime point), C(channel), Z(-stack), Y, X.

This module consists of a base Image class (BaseLocalImage). ImageDir
handles cases in which images are split into directories,
with each time point and channel having its own image file.
ImageDummy is a dummy class for silent failure testing.
"""

import hashlib
import re
import typing as t
from abc import ABC, abstractmethod, abstractproperty
from functools import cached_property
from glob import glob
from pathlib import Path

import dask
import dask.array as da
import imageio
import numpy as np
import zarr
from dask.array.image import imread

# from tifffile import TiffFile


def instantiate_image(source: t.Union[str, int, t.Dict[str, str], Path], **kwargs):
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


def dispatch_image(source: str or int or dict[str, str] or Path):
    """Pick the appropriate Image class for the source of data."""
    img_type = None
    if isinstance(source, int):
        from aliby.io.omero import Image

        img_type = Image
    elif isinstance(source, (list, tuple)):  # Local files
        img_type = ImageList
    else:
        match Path(source):
            case s if "*" in str(s):  # Wildcard
                img_type = ImageList
            case s if s.suffix == ".zarr":
                img_type = ImageZarr
            case s if s.is_dir() and s.exists():
                img_type = ImageDir
    return img_type


def files_to_image_sizes(path: Path, suffix="tiff"):
    """
    Deduce image sizes from the naming convention of tiff files.
    """
    filenames = list(path.glob(f"*.{suffix}"))
    try:
        # deduce order from filenames
        dimorder = "".join(map(lambda x: x[0], filenames[0].stem.split("_")[1:]))
        dim_value = list(
            map(
                lambda f: filename_to_dict_indices(f.stem),
                path.glob("*.tiff"),
            )
        )
        maxes = [max(map(lambda x: x[dim], dim_value)) for dim in dimorder]
        mins = [min(map(lambda x: x[dim], dim_value)) for dim in dimorder]
        dim_shapes = [max_val - min_val + 1 for max_val, min_val in zip(maxes, mins)]
        meta = {"size_" + dim: shape for dim, shape in zip(dimorder, dim_shapes)}
    except Exception as e:
        print(f"Warning: files_to_image_sizes failed.\nError: {e}")
        meta = {}
    return meta


def filename_to_dict_indices(stem: str):
    """Split string into a dict."""
    return {dim_number[0]: int(dim_number[1:]) for dim_number in stem.split("_")[1:]}


class BaseLocalImage(ABC):
    """Set path and provide method for context management."""

    # default image order
    default_dimorder = "tczyx"

    def __init__(self, path: t.Union[str, Path, list[str]]):
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

    @abstractproperty
    def meta(self):
        pass

    def set_meta(self):
        """Load metadata using parser dispatch."""
        from agora.io.metadata import parse_microscopy_logs

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
            original_order = [i[-1] for i in self.meta.keys() if i.startswith("size")]
            # Swap axis to conform with normal order
            target_order = [self.default_dimorder.index(x) for x in original_order]
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
        return [k.split("_")[-1] for k in self.meta.keys() if k.startswith("size")]


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
        self.meta.update({
            f"size_{dim}": shape for dim, shape in zip(self.dimorder, self._img.shape)
        })

    @property
    def name(self):
        """Return name of zarr directory."""
        return self.path.stem

    @property
    def dimorder(self):
        """Impose a hard-coded order of dimensions based on the zarr compression script."""
        return "TCZYX"


class ImageMultiTiff(BaseLocalImage):
    """
    Read multidimensional tiff files.
    """

    def __init__(
        self, path: t.Union[str, Path], capture_order: str, dimorder: str = None
    ):
        """Initialise using file name."""
        super().__init__(path)
        self.capture_order = capture_order
        self.dimorder = dimorder or "TCZYX"

        shape = imread(path).shape

        self.missing_dims = set(dimorder).difference(capture_order)
        for dim in self.missing_dims:
            self.meta[f"size_{dim}"] = 1

        self.meta = {
            f"size_{dim}": v for dim, v in zip(self.capture_order, shape) if dim
        }

        lazy = dask.array.imread(self.path)
        self._img = self.adjust_dimensions(lazy)

    def get_data_lazy(self) -> da.Array:
        """Return 5D dask array for lazy-loading local multidimensional zarr files."""
        return self._img

    def add_size_to_meta(self):
        """Add shape of image array to metadata."""
        self.meta.update({
            f"size_{dim}": shape for dim, shape in zip(self.dimorder, self._img.shape)
        })

    @property
    def name(self):
        """Return name of zarr directory."""
        return self.path.stem

    @property
    def dimorder(self):
        """Impose a hard-coded order of dimensions based on the zarr compression script."""
        return "TCZYX"


class ImageList(BaseLocalImage):
    """
    Image subclass that uses a wildcard or a (pre-sorted) list of images to associate images to channels. Assumes that every file is a 2-D array and the metadata that pertains to dimensions is encoded
    in the filename. the metadata must be extracted using regular expressions
    """

    def __init__(
        self,
        source: str or tuple[str],
        regex: str,
        capture_order: str,
        dimorder: str or None = None,  # output dimorder
        **kwargs,
    ):
        """
        Initialise using a directory and parse the files inside of it using a regex."""
        # super().__init__(img_files)
        self.path = source
        self.regex = regex
        self.capture_order = capture_order  #  or "CWTFZ"
        self._dimorder = dimorder or "TCZYX"
        if isinstance(source, str):  # The source is a wildcard
            self.image_filenames = sorted(
                x for x in glob(source) if re.match(self.regex, x)
            )
        else:  # The source is a list of images
            self.image_filenames = source

        self.image_id = calculate_checksum(
            self.image_filenames
        )  # checksum of all files

    @cached_property
    def meta(self):
        return {}

    def get_data_lazy(self) -> da.Array:
        """Return 5D dask array."""
        # Reshape first dimensions based on capture order
        lazy_arrays = [dask.delayed(imageio.imread)(fn) for fn in self.image_filenames]
        sample = lazy_arrays[0].compute()

        lazy_arrays = [
            da.from_delayed(x, shape=sample.shape, dtype=sample.dtype)
            for x in lazy_arrays
        ]

        order = tuple(self.dimorder_d[k] for k in self.dimorder if k in "TCZ")
        arr = np.zeros(
            order + (1, 1),
            dtype=object,
        )
        nd_order = np.array(
            np.where(
                np.arange(np.prod(tuple(self.dimorder_d.values()))).reshape(order) + 1
            )
        ).transpose()

        for i, (d1, d2, d3) in enumerate(nd_order):
            if i < len(lazy_arrays):
                arr[d1, d2, d3, 0, 0] = lazy_arrays[i]
            else:
                arr[d1, d2, d3, 0, 0] = da.from_delayed(
                    dask.delayed(np.zeros)(sample.shape),
                    shape=sample.shape,
                    dtype=sample.dtype,
                )

        a = da.block(arr.tolist())

        # rechunk to the last 3 dimensions. Leave time and channel unchunked
        pixels = da.rechunk(a, (1, 1, self.dimorder_d["Z"], *sample.shape))

        return pixels

    @property
    def name(self):
        """Return name of image directory."""
        return self.path.stem

    @property
    def dimorder(self):
        # sorted dimorder
        return self._dimorder

    @cached_property
    def dimorder_d(self):
        return get_dims_from_names(self.image_filenames, self.regex, self.capture_order)


def get_dims_from_names(
    image_filenames: list[str],
    regex: str,
    capture_order: str,
) -> dict[str, int]:
    """
    Capture order in this context means only the order in which matches occur in a regex.
    """
    regex_ = re.compile(regex)
    sorted_files = sorted(image_filenames)
    matches = [regex_.match(x).groups() for x in sorted_files]
    dim_size = {
        dim: len(set([y[i] for y in matches])) for i, dim in enumerate(capture_order)
    }

    # Check that the dimensions match the file
    if len(image_filenames) != np.prod(dim_size.values()):
        print(
            "Warning: The number of available images does not match the expected one given the dimensions and their maximum values. Will pad to match."
        )

    return dim_size


def calculate_checksum(filenames: list[str]) -> str:
    """
    Calculate the checksum for a list of image files.

    Helps to check that images composed of multiple other
    images are the same by calculating a checksum from their contents.

    Parameters
    ----------
    filenames : list[str]
        A list of file paths for the images.

    Returns
    -------
    str
        The hexadecimal representation of the MD5 checksum.
    """
    hash = hashlib.md5()
    for fn in filenames:
        hash.update(Path(fn).read_bytes())
    return hash.hexdigest()


def adjust_dimensions(lazy: da.array, capture_order: str, dimorder: str) -> da.array:
    """
    Adjusts the dimensions of a dask array to match a specified order.

    Parameters
    ----------
    lazy : da.array
        The input dask array.
    capture_order : str
        The current order of dimensions in the array.
    dimorder : str
        The desired order of dimensions.

    Returns
    -------
    da.array
        The input array with its dimensions adjusted to match `dimorder`.

    Notes
    -----
    This function first removes any unused dimensions from the array, then adds
    any missing dimensions. Finally, it rearranges the dimensions to match `dimorder`.
    """
    missing_dims = set(dimorder).difference(capture_order)

    # drop unused dimensions
    for i, dim in enumerate(reversed(capture_order), 1):
        if dim not in dimorder:
            assert lazy.shape[-i] == 1, f"Missing dimension {dim} not in dimorder"
            lazy = da.squeeze(lazy, -i)
            capture_order = capture_order.replace(dim, "")

    # Add missing dimensions
    for dim in sorted(missing_dims):
        lazy = lazy[..., np.newaxis]
        capture_order += dim

    assert len(capture_order) == len(dimorder), (
        "Post-adjustment captureorder and dimorder do not match."
    )
    # sort capture_order to match dimorder
    lazy = da.moveaxis(
        lazy, [capture_order.index(i) for i in dimorder], range(len(dimorder))
    )
    return lazy
