"""Tools to manage I/O using a remote OMERO server."""

import re
import typing as t
from collections.abc import Iterable
from pathlib import Path

import dask.array as da
import numpy as np
from agora.io.bridge import BridgeH5
from agora.utils.indexing import wrap_int
from dask import delayed
from yaml import safe_load

import omero
from omero.gateway import BlitzGateway, ImageWrapper
from omero.model import enums as omero_enums

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


class BridgeOmero:
    """
    Interact with OMERO.

    See
    https://docs.openmicroscopy.org/omero/5.6.0/developers/Python.html
    """

    def __init__(
        self,
        host: str = None,
        username: str = None,
        password: str = None,
        ome_id: int = None,
    ):
        """
        Initialise with OMERO login details.

        Parameters
        ----------
        host : string
            web address of OMERO host
        username: string
        password : string
        ome_id: Optional int
            Unique identifier on Omero database.
            Used to fetch specific objects.
        """
        if host is None or username is None or password is None:
            raise ValueError(
                f"Invalid credentials. host: {host}, user: {username},"
                f" pwd: {password}"
            )
        self.conn = None
        self.host = host
        self.username = username
        self.password = password
        self.ome_id = ome_id

    def __enter__(self):
        """For Python's with statement."""
        self.create_gate()
        return self

    def __exit__(self, *exc) -> bool:
        """For Python's with statement."""
        for e in exc:
            if e is not None:
                print(e)
        self.conn.close()
        return False

    @property
    def ome_class(self):
        """Initialise OMERO Object Wrapper for instances when applicable."""
        if not hasattr(self, "_ome_class"):
            if self.conn.isConnected() and self.ome_id is not None:
                ome_type = [
                    valid_name
                    for valid_name in ("Dataset", "Image")
                    if re.match(
                        f".*{ valid_name }.*",
                        self.__class__.__name__,
                        re.IGNORECASE,
                    )
                ][0]
                # load data
                self._ome_class = self.conn.getObject(ome_type, self.ome_id)
                assert self._ome_class, f"{ome_type} {self.ome_id} not found."
            else:
                raise ConnectionError("No Blitz connection or valid OMERO ID.")
        return self._ome_class

    def create_gate(self) -> bool:
        """Connect to OMERO."""
        self.conn = BlitzGateway(
            host=self.host, username=self.username, passwd=self.password
        )
        self.conn.connect()
        self.conn.c.enableKeepAlive(60)
        return self.conn.isConnected()

    def destroy_gate(self) -> bool:
        """Disconnect from OMERO."""
        self.conn.close()
        return self.conn.isConnected()

    @classmethod
    def server_info_from_h5(cls, filepath: t.Union[str, Path]):
        """
        Return server info from hdf5 file.

        Parameters
        ----------
        cls : BridgeOmero
            BridgeOmero class
        filepath : t.Union[str, Path]
            Location of hdf5 file.
        """
        bridge = BridgeH5(filepath)
        meta = safe_load(bridge.meta_h5["parameters"])["general"]
        server_info = {k: meta[k] for k in ("host", "username", "password")}
        return server_info

    def set_id(self, ome_id: int):
        """Set ome_id attribute."""
        self.ome_id = ome_id

    @property
    def file_annotations(self):
        """Get file annotations."""
        valid_annotations = [
            ann.getFileName()
            for ann in self.ome_class.listAnnotations()
            if hasattr(ann, "getFileName")
        ]
        return valid_annotations

    def add_file_as_annotation(
        self, file_to_upload: t.Union[str, Path], **kwargs
    ):
        """
        Upload annotation to object on OMERO server.

        Only valid in subclasses.

        Parameters
        ----------
        file_to_upload: File to upload
        **kwargs: Additional keyword arguments passed on
            to BlitzGateway.createFileAnnfromLocalFile
        """
        file_annotation = self.conn.createFileAnnfromLocalFile(
            file_to_upload,
            mimetype="text/plain",
            **kwargs,
        )
        self.ome_class.linkAnnotation(file_annotation)


class Dataset(BridgeOmero):
    """
    Interact with Omero Datasets remotely.

    Access their metadata and associated files and images.

    Parameters
    ----------
    expt_id: int
        Dataset id on server
    server_info: dict
        Host, username and password
    """

    def __init__(self, expt_id: int, **server_info):
        """Initialise with experiment OMERO ID and server details."""
        super().__init__(ome_id=expt_id, **server_info)

    @property
    def name(self):
        """Get name of experiment."""
        return self.ome_class.getName()

    @property
    def date(self):
        """Get date of experiment."""
        return self.ome_class.getDate()

    @property
    def unique_name(self):
        """Get full name of experiment including its date."""
        return "_".join(
            (
                str(self.ome_id),
                self.date.strftime("%Y_%m_%d").replace("/", "_"),
                self.name,
            )
        )

    def get_position_ids(self):
        """Get dict of image names and IDs from OMERO."""
        return {
            im.getName(): im.getId() for im in self.ome_class.listChildren()
        }

    def get_channels(self):
        """
        Get list of channels from OMERO.

        Assume all positions have the same channels.
        """
        for im in self.ome_class.listChildren():
            channels = [ch.getLabel() for ch in im.getChannels()]
            break
        return channels

    @property
    def files(self):
        """Get a dict of FileAnnotationWrappers, typically for log files."""
        if not hasattr(self, "_files"):
            self._files = {
                x.getFileName(): x
                for x in self.ome_class.listAnnotations()
                if isinstance(x, omero.gateway.FileAnnotationWrapper)
            }
        if not self._files:
            raise FileNotFoundError(
                "exception:metadata: experiment has no annotation files."
            )
        if len(self.file_annotations) != len(self._files):
            raise FileNotFoundError(
                "Number of files and annotations do not match"
            )
        return self._files

    @property
    def tags(self):
        """Get a dict of TagAnnotationWrapper from OMERO for each position."""
        if not hasattr(self, "_tags"):
            self._tags = {
                x.getId(): x
                for x in self.ome_class.listAnnotations()
                if isinstance(x, omero.gateway.TagAnnotationWrapper)
            }
        return self._tags

    def cache_logs(self, root_dir):
        """Save the log files for an experiment."""
        valid_suffixes = ("txt", "log")
        for _, annotation in self.files.items():
            filepath = root_dir / annotation.getFileName().replace("/", "_")
            if (
                any([str(filepath).endswith(suff) for suff in valid_suffixes])
                and not filepath.exists()
            ):
                # save only the text files
                with open(str(filepath), "wb") as fd:
                    for chunk in annotation.getFileInChunks():
                        fd.write(chunk)
        return True

    @classmethod
    def from_h5(
        cls,
        filepath: t.Union[str, Path],
    ):
        """
        Instantiate data set from a h5 file.

        Parameters
        ----------
        cls : Image
            Image class
        filepath : t.Union[str, Path]
            Location of hdf5 file.
        """
        bridge = BridgeH5(filepath)
        dataset_keys = ("omero_id", "omero_id,", "dataset_id")
        for k in dataset_keys:
            if k in bridge.meta_h5:
                return cls(
                    bridge.meta_h5[k], **cls.server_info_from_h5(filepath)
                )


class Image(BridgeOmero):
    """Load images from OMERO and their data and metadata."""

    def __init__(self, image_id: int, **server_info):
        """
        Connect to the OMERO server.

        Parameters
        ----------
        image_id: integer
        server_info: dictionary
            Specifies the host, username, and password as strings
        """
        super().__init__(ome_id=image_id, **server_info)

    @classmethod
    def from_h5(
        cls,
        filepath: t.Union[str, Path],
    ):
        """
        Instantiate Image from a h5 file.

        Parameters
        ----------
        cls : Image
            Image class
        filepath : t.Union[str, Path]
            Location of h5 file.
        """
        bridge = BridgeH5(filepath)
        image_id = bridge.meta_h5["image_id"]
        return cls(image_id, **cls.server_info_from_h5(filepath))

    @property
    def name(self):
        """Get name."""
        return self.ome_class.getName()

    @property
    def data(self):
        """Load image data as a 5D dask array - TCXYZ."""
        return load_data_lazy(self.ome_class)

    @property
    def metadata(self):
        """
        Get image metadata from OMERO as a dict.

        Get image size, number of time points, labels of channels,
        and image name.
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


class MinimalImage(Image):
    """Load images from OMERO."""

    def __init__(self, image_id, **server_info):
        """
        Connect to the OMERO server.

        Parameters
        ----------
        image_id: integer
        server_info: dictionary
            Specifies the host, username, and password as strings
        """
        super().__init__(image_id, **server_info)
        success = self.create_gate()
        if success:
            print("Connected to OMERO.")
        else:
            raise ConnectionError("Failed to connect to OMERO.")

    @property
    def data(self):
        """Get image data as a 5D dask array - TCXYZ."""
        try:
            return load_data_lazy(image=self.ome_class)
        except ConnectionError as e:
            print(f"Failed to fetch image from server: {e}")
            # disconnect from OMERO
            self.conn.connect(False)
            raise e

    def tiles(self, tile_slices, tps, channel_indices, zs):
        """Get tiles as dask arrays."""
        tps, channel_indices, zs = (
            wrap_int(tps),
            wrap_int(channel_indices),
            wrap_int(zs),
        )
        return load_tiles_lazy(
            image=self.ome_class,
            tile_slices=tile_slices,
            tps=tps,
            channel_indices=channel_indices,
            zs=zs,
        )


@delayed
def load_plane(pixels, z, c, tp):
    """Load a single plane lazily."""
    return pixels.getPlane(z, c, tp)


def load_data_lazy(image: ImageWrapper) -> da.Array:
    """Load a 5D dask array (T, C, Z, Y, X) from OMERO image."""
    nt, nc, nz, ny, nx = (
        image.getSizeT(),
        image.getSizeC(),
        image.getSizeZ(),
        image.getSizeY(),
        image.getSizeX(),
    )
    # get dtype
    pixels = image.getPrimaryPixels()
    omero_type = pixels.getPixelsType().getValue()
    dtype = PIXEL_TYPES.get(omero_type, np.uint16)
    # create delayed objects for each plane
    delayed_planes = []
    for tp in range(nt):
        for c in range(nc):
            for z in range(nz):
                delayed_planes.append(load_plane(pixels, z, c, tp))
    arrays = [
        da.from_delayed(delayed_plane, shape=(ny, nx), dtype=dtype)
        for delayed_plane in delayed_planes
    ]
    dask_array = da.stack(arrays).reshape(nt, nc, nz, ny, nx)
    return dask_array


def load_tiles_lazy(
    image: ImageWrapper,
    tile_slices: Iterable[tuple[slice, slice]],
    tps: Iterable[int],
    channel_indices: Iterable[int],
    zs: Iterable[int],
) -> list[da.Array]:
    """
    Load tiles from OMERO image as dask arrays.

    Tiles are arranged as z stacks for each channel for each time point.
    """
    if len(tile_slices) != len(tps):
        raise ValueError("For each time point, you need a tile location.")
    # shape: (T, C, Z, Y, X)
    data = load_data_lazy(image)
    tiles = []
    for tp, tile_slice in zip(tps, tile_slices):
        for c in channel_indices:
            for z in zs:
                plane = data[tp, c, z]
                tiles.append(plane[tile_slice])
    return tiles
