import re
import typing as t
from abc import abstractmethod
from pathlib import PosixPath

import omero
from omero.gateway import BlitzGateway
from yaml import safe_load

from agora.io.bridge import BridgeH5


class BridgeOmero:
    """
    Core to interact with OMERO, using credentials or fetching them from h5 file (temporary trick).
    See
    https://docs.openmicroscopy.org/omero/5.6.0/developers/Python.html
    """

    def __init__(
        self,
        host="islay.bio.ed.ac.uk",
        username="upload",
        password="***REMOVED***",
    ):
        """
        Parameters
        ----------
        host : string
            web address of OMERO host
        username: string
        password : string
        """
        self.conn = None
        self.host = host
        self.username = username
        self.password = password

    # standard method required for Python's with statement
    def __enter__(self):
        self.create_gate()
        self.init_wrapper()

        return self

    def init_wrapper(self):
        # Initialise Omero Object Wrapper for instances when applicable.
        if hasattr(self, "ome_id"):
            ome_type = [
                valid_name
                for valid_name in ("Dataset", "Image")
                if re.match(
                    f".*{ valid_name }.*",
                    self.__class__.__name__,
                    re.IGNORECASE,
                )
            ][0]
            self.ome_class = self.conn.getObject(ome_type, self.ome_id)

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

    @abstractmethod
    def init_interface(self):
        ...

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
    def __init__(self, expt_id, **server_info):
        self.ome_id = expt_id

        super().__init__(**server_info)

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
        for name, annotation in self.files.items():
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
