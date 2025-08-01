"""Tools to manage I/O using a remote OMERO server."""

import re
import typing as t
from pathlib import Path

import dask.array as da
import numpy as np
import omero
from dask import delayed
from omero.gateway import BlitzGateway
from omero.model import enums as omero_enums
from yaml import safe_load

from agora.io.bridge import BridgeH5

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
            raise Exception(
                f"Invalid credentials. host: {host}, user: {username}, pwd: {password}"
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
        """Initialise Omero Object Wrapper for instances when applicable."""
        if not hasattr(self, "_ome_class"):
            if self.conn.isConnected() and self.ome_id is not None:
                ome_type = [
                    valid_name
                    for valid_name in ("Dataset", "Image")
                    if re.match(
                        f".*{valid_name}.*",
                        self.__class__.__name__,
                        re.IGNORECASE,
                    )
                ][0]
                # load data
                self._ome_class = self.conn.getObject(ome_type, self.ome_id)
                assert self._ome_class, f"{ome_type} {self.ome_id} not found."
            else:
                raise Exception("No Blitz connection or valid omero id.")
        return self._ome_class

    def create_gate(self) -> bool:
        """Connect to OMERO."""
        self.conn = BlitzGateway(
            host=self.host, username=self.username, passwd=self.password
        )
        self.conn.connect()
        self.conn.c.enableKeepAlive(60)
        self.conn.isConnected()

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

    def add_file_as_annotation(self, file_to_upload: t.Union[str, Path], **kwargs):
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
        return {im.getName(): im.getId() for im in self.ome_class.listChildren()}

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
        if not len(self._files):
            raise Exception("exception:metadata: experiment has no annotation files.")
        elif len(self.file_annotations) != len(self._files):
            raise Exception("Number of files and annotations do not match")
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
                return cls(bridge.meta_h5[k], **cls.server_info_from_h5(filepath))


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
        """Get image data as a 5D dask array - TCXYZ."""
        return get_data_lazy(self.ome_class)

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


class UnsafeImage(Image):
    """
    Load images from OMERO and gives access to the data and metadata.

    This class is a temporary solution while we find a way to use
    context managers inside napari. It risks resulting in zombie
    connections and producing freezes in an OMERO server.
    """

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
        self.create_gate()

    @property
    def data(self):
        """Get image data as a 5D dask array - TCXYZ."""
        try:
            return get_data_lazy(self.ome_class)
        except Exception as e:
            print(f"ERROR: Failed fetching image from server: {e}")
            self.conn.connect(False)


def get_data_lazy(image) -> da.Array:
    """
    Get 5D dask array - TCXYZ.

    Use delayed reading from OMERO image.
    """
    nt, nc, nz, ny, nx = [getattr(image, f"getSize{x}")() for x in "TCZYX"]
    pixels = image.getPrimaryPixels()
    dtype = PIXEL_TYPES.get(pixels.getPixelsType().value, None)
    # using dask
    get_plane = delayed(lambda idx: pixels.getPlane(*idx))
    # 5D stack: TCZXY
    t_stacks = []
    for tpt in range(nt):
        c_stacks = []
        for c in range(nc):
            z_stack = []
            for z in range(nz):
                plane = da.from_delayed(
                    get_plane((z, c, tpt)), shape=(ny, nx), dtype=dtype
                )
                z_stack.append(plane)
            c_stacks.append(da.stack(z_stack))
        t_stacks.append(da.stack(c_stacks))
    return da.stack(t_stacks)
