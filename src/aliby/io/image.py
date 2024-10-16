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

import hashlib
import re
import typing as t
from abc import ABC, abstractmethod, abstractproperty
from datetime import datetime
from functools import cached_property
from glob import glob
from pathlib import Path

import dask
import dask.array as da
import imageio
import numpy as np
import xmltodict
import zarr
from agora.io.metadata import parse_metadata
from dask.array.image import imread
from tifffile import TiffFile


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
    if isinstance(source, int):
        from aliby.io.omero import Image

        img_type = Image
    else:  # Local files
        match Path(source):
            case s if s.is_dir() and s.exists():
                img_type = ImageDir
            case s if s.suffix(".zarr"):
                img_type = ImageZarr
            case s if s.is_file() and s.exists():
                img_type = ImageLocalOME
            case _:
                raise ("Invalid data type")

    return img_type

    # if isinstance(source, (int, np.int64)):
    #     # requires omero module
    #     from aliby.io.omero import Image

    #     instantiator = Image
    # elif isinstance(source, dict):
    #         instantiator = ImageDir
    # elif isinstance(source, ):
    #     or (
    #     (source_path := Path(source)).is_dir() and source_path.exists()
    # ):
    #     # zarr files are considered directories
    #     if source_path.suffix == ".zarr":
    #         instantiator = ImageZarr
    #     else:
    # elif isinstance(source, (str, Path)) and Path(source).is_file():
    #     instantiator = ImageLocalOME
    # elif isinstance(source, str) and not Path(source).is_file():
    #     instantiator = ImageIndFiles
    # else:
    #     raise Exception(f"Invalid data source at {source}")
    # return instantiator


def files_to_image_sizes(path: Path, suffix="tiff"):
    """Deduce image sizes from the naming convention of tiff files."""
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
        print("Warning: files_to_image_sizes failed." f"\nError: {e}")
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
        parsed_meta = parse_metadata(self.path)
        if parsed_meta is None:
            # try to deduce metadata
            parsed_meta = files_to_image_sizes(self.path)
        self.meta = parsed_meta

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

    This is a derivative Image class. It fetches an image from OMEXML data format,
    in which a multidimensional tiff image contains the metadata.
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
                    *[i for i, d in enumerate(self.base) if d not in self.dimorder],
                )
                reshaped = da.reshape(
                    img,
                    shape=(
                        *img.shape,
                        *[1 for _ in range(5 - len(self.dimorder))],
                    ),
                )
                img = da.moveaxis(reshaped, range(len(reshaped.shape)), target_order)
        return self.rechunk_data(img)


class ImageDirLegacy(BaseLocalImage):
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
        self.meta.update(
            {f"size_{dim}": shape for dim, shape in zip(self.dimorder, self._img.shape)}
        )

    @property
    def name(self):
        """Return name of zarr directory."""
        return self.path.stem

    @property
    def dimorder(self):
        """Impose a hard-coded order of dimensions based on the zarr compression script."""
        return "TCZYX"


class ImageDir(BaseLocalImage):
    """
    Standard image class for tiff files.

    Image class for the case in which all images are split in one or
    multiple folders with positions and time-points as independent files.
    It inherits from BaseLocalImage so we only override methods that are critical.
    The target use-case of this class is when we create virtual staining channels in
    independent folders.

    Assumptions:
    - One folder per channel.
    - Images are flat.
    - Position, Time, z-stack and the others are determined by filenames.
    - Provides Dimorder as it is set in the filenames, or expects order

    There are some peculiarities:
    path is a wildcard that encodes all the dimensional information.
    image_id is the hex md5sum for all the images involved.
    there is an `image_filenames` directory that encodes both
    """

    def __init__(
        self,
        wildcard: str,
        regex: str or None = None,
        image_filenames: list[str] or None = None,
        capture_order: str or None = None,
        dimorder: str or None = None,
        **kwargs,
    ):
        """Initialise using a directory and parse the files inside of it using a regex."""
        # super().__init__(img_files)
        self.path = wildcard
        # self.meta = filename_to_meta_gsk(self.path)
        self.regex = regex or self.path.replace("*", "(.*)")
        self.image_filenames = sorted(
            image_filenames
            or tuple(x for x in glob(wildcard) if re.match(self.regex, x))
        )
        self.image_id = calculate_checksum(
            self.image_filenames
        )  # checksum of all files
        self.capture_order = capture_order or "CWTFZ"
        self._dimorder = dimorder or "TCZYX"

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

        # This has to be done manually because if we load everything
        # reshaping fails
        # FIXME: Temporary workaround, something is off when fetching
        # large chunks of the data
        order = tuple(v for k, v in self.dimorder_d.items() if k in self._dimorder)
        arr = np.empty(
            order + (1, 1),
            dtype=object,
        )
        nd_order = np.array(
            np.where(np.arange(len(lazy_arrays)).reshape(order) + 1)
        ).transpose()

        for i, (d1, d2, d3) in enumerate(nd_order):
            arr[d1, d2, d3, 0, 0] = lazy_arrays[i]

        # convert to TCZ
        arr = arr.swapaxes(0, 1)
        a = da.block(arr.tolist())

        # rechunk to the last 3 dimensions
        pixels = da.rechunk(a, (1, 1, 9, 1000, 1000))

        return pixels

    @property
    def name(self):
        """Return name of image directory."""
        return self.path.stem

    @property
    def dimorder(self):
        # sorted dimorder
        return [self.dimorder_d[dim] for dim in self._dimorder]

    @cached_property
    def dimorder_d(self):
        return get_dims_from_names(self.image_filenames, self.regex, self.capture_order)

    @cached_property
    def image_filenames(self):
        return tuple(glob(self.path))


def get_dims_from_names(
    image_filenames: list[str],
    regex: str,
    capture_order: str,
) -> dict[str, int]:
    regex_ = re.compile(regex)
    sorted_files = sorted(image_filenames)
    matches = [regex_.match(x).groups() for x in sorted_files]
    dim_size = {
        dim: len(set([y[i] for y in matches])) for i, dim in enumerate(capture_order)
    }

    # Check that the dimensions match the file
    n = 1
    for v in dim_size.values():
        n *= v
    assert len(image_filenames) == n

    return dim_size


def calculate_checksum(filenames: list[str]) -> bytes:
    """
    This helps to check that images composed of multiple other
    images are the same.
    """
    hash = hashlib.md5()
    for fn in filenames:
        hash.update(Path(fn).read_bytes())
    return hash.hexdigest()
