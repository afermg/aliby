"""
Tools to manage I/O using a remote OMERO server.
"""

import re
import typing as t
from abc import abstractmethod
from pathlib import PosixPath

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
    Core to interact with OMERO, using credentials or fetching them from h5 file (temporary trick).
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
        Parameters
        ----------
        host : string
            web address of OMERO host
        username: string
        password : string
        ome_id: Optional int
            Unique identifier on Omero database. Used to fetch specific objects.
        """
        # assert all((host, username, password)), str(f"Invalid credentials host:{host}, user:{username}, pass:{pass}")
        assert all(
            (host, username, password)
        ), f"Invalid credentials. host: {host}, user: {username}, pwd: {password}"

        self.conn = None
        self.host = host
        self.username = username
        self.password = password
        self.ome_id = ome_id

    # standard method required for Python's with statement
    def __enter__(self):
        self.create_gate()

        return self

    @property
    def ome_class(self):
        # Initialise Omero Object Wrapper for instances when applicable.
        if not hasattr(self, "_ome_class"):
            assert (
                self.conn.isConnected() and self.ome_id is not None
            ), "No Blitz connection or valid omero id"

            ome_type = [
                valid_name
                for valid_name in ("Dataset", "Image")
                if re.match(
                    f".*{ valid_name }.*",
                    self.__class__.__name__,
                    re.IGNORECASE,
                )
            ][0]
            self._ome_class = self.conn.getObject(ome_type, self.ome_id)

            assert self._ome_class, f"{ome_type} {self.ome_id} not found."

        return self._ome_class

    def create_gate(self) -> bool:
        self.conn = BlitzGateway(
            host=self.host, username=self.username, passwd=self.password
        )
        self.conn.connect()
        self.conn.c.enableKeepAlive(60)

        self.conn.isConnected()

    # standard method required for Python's with statement
    def __exit__(self, *exc) -> bool:
        for e in exc:
            if e is not None:
                print(e)

        self.conn.close()
        return False

    @classmethod
    def server_info_from_h5(
        cls,
        filepath: t.Union[str, PosixPath],
    ):
        """Return server info from hdf5 file.

        Parameters
        ----------
        cls : BridgeOmero
            BridgeOmero class
        filepath : t.Union[str, PosixPath]
            Location of hdf5 file.

        Examples
        --------
        FIXME: Add docs.

        """
        # metadata = load_attributes(filepath)
        bridge = BridgeH5(filepath)
        server_info = safe_load(bridge.meta_h5["parameters"])["general"][
            "server_info"
        ]
        return server_info

    def set_id(self, ome_id: int):
        self.ome_id = ome_id

    @property
    def file_annotations(self):
        valid_annotations = [
            ann.getFileName()
            for ann in self.ome_class.listAnnotations()
            if hasattr(ann, "getFileName")
        ]
        return valid_annotations

    def add_file_as_annotation(
        self, file_to_upload: t.Union[str, PosixPath], **kwargs
    ):
        """Upload annotation to object on OMERO server. Only valid in subclasses.

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
    def __init__(self, expt_id: str or int, **server_info):
        super().__init__(ome_id=expt_id, **server_info)

    @property
    def name(self):
        return self.ome_class.getName()

    @property
    def date(self):
        return self.ome_class.getDate()

    @property
    def unique_name(self):
        return "_".join(
            (
                str(self.ome_id),
                self.date.strftime("%Y_%m_%d").replace("/", "_"),
                self.name,
            )
        )

    def get_images(self):
        return {
            im.getName(): im.getId() for im in self.ome_class.listChildren()
        }

    @property
    def files(self):
        if not hasattr(self, "_files"):
            self._files = {
                x.getFileName(): x
                for x in self.ome_class.listAnnotations()
                if isinstance(x, omero.gateway.FileAnnotationWrapper)
            }
        if not len(self._files):
            raise Exception(
                "exception:metadata: experiment has no annotation files."
            )
        elif len(self.file_annotations) != len(self._files):
            raise Exception("Number of files and annotations do not match")

        return self._files

    @property
    def tags(self):
        if self._tags is None:
            self._tags = {
                x.getname(): x
                for x in self.ome_class.listAnnotations()
                if isinstance(x, omero.gateway.TagAnnotationWrapper)
            }
        return self._tags

    def cache_logs(self, root_dir):
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
        filepath: t.Union[str, PosixPath],
    ):
        """Instatiate Dataset from a hdf5 file.

        Parameters
        ----------
        cls : Image
            Image class
        filepath : t.Union[str, PosixPath]
            Location of hdf5 file.

        Examples
        --------
        FIXME: Add docs.

        """
        # metadata = load_attributes(filepath)
        bridge = BridgeH5(filepath)
        dataset_keys = ("omero_id", "omero_id,", "dataset_id")
        for k in dataset_keys:
            if k in bridge.meta_h5:
                return cls(
                    bridge.meta_h5[k], **cls.server_info_from_h5(filepath)
                )


class Image(BridgeOmero):
    """
    Loads images from OMERO and gives access to the data and metadata.
    """

    def __init__(self, image_id: int, **server_info):
        """
        Establishes the connection to the OMERO server via the Argo
        base class.

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
        filepath: t.Union[str, PosixPath],
    ):
        """Instatiate Image from a hdf5 file.

        Parameters
        ----------
        cls : Image
            Image class
        filepath : t.Union[str, PosixPath]
            Location of hdf5 file.

        Examples
        --------
        FIXME: Add docs.

        """
        # metadata = load_attributes(filepath)
        bridge = BridgeH5(filepath)
        image_id = bridge.meta_h5["image_id"]
        return cls(image_id, **cls.server_info_from_h5(filepath))

    @property
    def name(self):
        return self.ome_class.getName()

    @property
    def data(self):
        return get_data_lazy(self.ome_class)

    @property
    def metadata(self):
        """
        Store metadata saved in OMERO: image size, number of time points,
        labels of channels, and image name.
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
    Loads images from OMERO and gives access to the data and metadata.
    This class is a temporary solution while we find a way to use
    context managers inside napari. It risks resulting in zombie connections
    and producing freezes in an OMERO server.

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
        super().__init__(image_id, **server_info)
        self.create_gate()

    @property
    def data(self):
        try:
            return get_data_lazy(self.ome_class)
        except Exception as e:
            print(f"ERROR: Failed fetching image from server: {e}")
            self.conn.connect(False)


def get_data_lazy(image) -> da.Array:
    """
    Get 5D dask array, with delayed reading from OMERO image.
    """
    nt, nc, nz, ny, nx = [getattr(image, f"getSize{x}")() for x in "TCZYX"]
    pixels = image.getPrimaryPixels()
    dtype = PIXEL_TYPES.get(pixels.getPixelsType().value, None)
    # using dask
    get_plane = delayed(lambda idx: pixels.getPlane(*idx))

    def get_lazy_plane(zct):
        return da.from_delayed(get_plane(zct), shape=(ny, nx), dtype=dtype)

    # 5D stack: TCZXY
    t_stacks = []
    for t in range(nt):
        c_stacks = []
        for c in range(nc):
            z_stack = []
            for z in range(nz):
                z_stack.append(get_lazy_plane((z, c, t)))
            c_stacks.append(da.stack(z_stack))
        t_stacks.append(da.stack(c_stacks))

    return da.stack(t_stacks)
