"""
Image: Loads images and registers them.

Image instances load images from a specified directory into an object that
also contains image properties such as name and metadata. Pixels from images
are stored in dask arrays; the standard way is to store them in 5-dimensional
arrays: T(ime point), C(channel), Z(-stack), Y, X.

The pixels are read by tiler's sources, which aliby shares with wela, bairn
and the curation GUI; this module adds what aliby knows and they do not, the
microscope's log files. BaseLocalImage is the base class. ImageLocalOME
handles a multi-dimensional OME-TIFF. ImageDir handles a directory with one
TIFF for each time point, channel and z slice. ImageZarr handles a zarr
store.
"""

from abc import ABC, abstractmethod, abstractproperty
from datetime import datetime
from pathlib import Path

import dask.array as da
import numpy as np
import xmltodict
from agora.io.metadata import parse_microscopy_logs
from tiler import TiffFolderSource, TiffSource, ZarrSource, as_dask


def instantiate_image(source: str | int | dict[str, str] | Path, **kwargs):
    """
    Instantiate the image.

    Parameters
    ----------
    source : str, int, dict or Path
        Image identifier

    Examples
    --------
    image_path = "path/to/image"
    with instantiate_image(image_path) as img:
        print(imz.data, img.metadata)
    """
    return dispatch_image(source)(source, **kwargs)


def dispatch_image(source: str | int | dict[str, str] | Path):
    """Pick the appropriate Image class for the source of data."""
    if isinstance(source, (int, np.int64)):
        # omero is optional, so import it only for OMERO data
        from aliby.io.omero import Image

        instantiator = Image
    elif isinstance(source, dict) or (
        isinstance(source, (str, Path)) and Path(source).is_dir()
    ):
        if Path(source).suffix == ".zarr":
            instantiator = ImageZarr
        else:
            instantiator = ImageDir
    elif isinstance(source, (str, Path)) and Path(source).is_file():
        instantiator = ImageLocalOME
    else:
        raise ValueError(f"Invalid data source at {source}.")
    return instantiator


def shape_to_meta(shape: tuple[int, ...]) -> dict[str, int]:
    """Describe an image by the sizes of its dimensions, TCZYX."""
    return {f"size_{dim}": int(size) for dim, size in zip("tczyx", shape)}


class BaseLocalImage(ABC):
    """
    Set path and provide method for context management.

    A subclass sets ``source``, the tiler source that reads its pixels.
    Leaving a with block closes nothing, so the pixels stay readable, as a
    Tiler built inside the block expects.
    """

    # default image order
    default_dimorder = "tczyx"

    def __init__(self, path: str | Path):
        """Initiate with data directory."""
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

    @property
    def data(self):
        """Get data."""
        return self.get_data_lazy()

    @property
    def metadata(self):
        """Get metadata."""
        return self.meta

    @property
    def pixel_size_um(self) -> float | None:
        """Return the pixel size the image records, or None."""
        return self.source.pixel_size_um

    def set_meta(self):
        """
        Load metadata from microscopy logs.

        With no log, describe the image by the sizes of its dimensions.
        """
        self.meta = parse_microscopy_logs(self.path)
        if self.meta is None:
            self.meta = shape_to_meta(self.source.shape)

    @abstractmethod
    def get_data_lazy(self):
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

    Fetch an image from OMEXML data format, in which a multidimensional
    tiff image contains the metadata.
    """

    def __init__(self, path: str, dimorder=None, **kwargs):
        """
        Initialise using file name.

        Parameters
        ----------
        path : str
            The OME-TIFF.
        dimorder : str, optional
            The image's axes, such as "TCYX", overriding the file's.
        """
        super().__init__(path)
        self._id = str(path)
        self.source = TiffSource(self.path, axes=dimorder)
        self.set_meta()

    def set_meta(self):
        """Get metadata from the tiff file itself."""
        ome_metadata = self.source.tif.ome_metadata
        self.ome = (
            xmltodict.parse(ome_metadata)["OME"] if ome_metadata else {}
        )
        self.meta = shape_to_meta(self.source.shape)
        self.meta["channels"] = self.source.channels
        self.meta["name"] = self.ome_image_name() or self.source.name
        self.meta["type"] = str(self.source.dtype)

    def ome_image_name(self) -> str | None:
        """Return the name the OME-XML gives the first image, if any."""
        image = self.ome.get("Image")
        if isinstance(image, list):
            image = image[0] if image else None
        if not isinstance(image, dict):
            return None
        return image.get("@Name")

    @property
    def name(self):
        """Get name of experiment."""
        return self.meta["name"]

    @property
    def date(self):
        """Get date of experiment."""
        date_str = [
            x
            for x in self.ome["StructuredAnnotations"]["TagAnnotation"]
            if x["Description"] == "Date"
        ][0]["Value"]
        return datetime.strptime(date_str, "%d-%b-%Y")

    @property
    def dimorder(self):
        """Return the order of dimensions in the data."""
        return "TCZYX"

    def get_data_lazy(self) -> da.Array:
        """Return 5D dask array, reading a page when it is computed."""
        return as_dask(self.source)


class ImageDir(BaseLocalImage):
    """
    Read tiff files.

    Each position should has a separate directory and the files must
    be named following the convention:
       position_t0001_channel_z01.tiff

    We assume that the images are shaped Y times X.
    The data is put in the order of TCZYX. The channels follow the
    microscope's log, if there is one, and each channel is read from the
    files that name it.
    """

    def __init__(self, path: str | Path, **kwargs):
        """Initialise and define metadata."""
        super().__init__(path)
        log_meta = parse_microscopy_logs(self.path)
        channels = None if log_meta is None else log_meta.get("channels")
        self.source = TiffFolderSource(self.path, channels=channels)
        self.meta = log_meta
        if self.meta is None:
            self.meta = shape_to_meta(self.source.shape)

    def get_data_lazy(self) -> da.Array:
        """Return 5D dask array."""
        self.meta["size_y"], self.meta["size_x"] = self.source.shape[-2:]
        return as_dask(self.source)

    @property
    def name(self):
        """Return the file name without its suffix."""
        return self.path.stem

    @property
    def dimorder(self):
        """Assume the default order for tiff files."""
        return "TCZYX"


class ImageZarr(BaseLocalImage):
    """Read zarr compressed files."""

    def __init__(self, path: str | Path, **kwargs):
        """Initialise using file name."""
        super().__init__(path)
        self.source = ZarrSource(self.path)
        self.set_meta()
        self.add_size_to_meta()

    def get_data_lazy(self):
        """Return the zarr array, which reads lazily."""
        return self.source.array

    def add_size_to_meta(self):
        """Add shape of image array to metadata."""
        self.meta.update(
            {
                f"size_{dim}": shape
                for dim, shape in zip(self.dimorder, self.source.array.shape)
            }
        )

    @property
    def name(self):
        """Return name of zarr directory."""
        return self.path.stem

    @property
    def dimorder(self):
        """Return the order of dimensions the zarr script writes."""
        return "TCZYX"
