#!/usr/bin/env python3
import shutil
import typing as t
from pathlib import Path, PosixPath
from typing import Union

import omero
from agora.io.bridge import BridgeH5

from aliby.io.image import ImageLocal
from aliby.io.omero import BridgeOmero


class DatasetLocal:
    """Load a dataset from a folder

    We use a given image of a dataset to obtain the metadata, for we cannot expect folders to contain it straight away.

    """

    def __init__(self, dpath: Union[str, PosixPath], *args, **kwargs):
        self.fpath = Path(dpath)
        assert len(self.get_images()), "No tif files found"

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    @property
    def dataset(self):
        return self.fpath

    @property
    def name(self):
        return self.fpath.name

    @property
    def unique_name(self):
        return self.fpath.name

    @property
    def date(self):
        return ImageLocal(list(self.get_images().values())[0]).date

    def get_images(self):
        return {f.name: str(f) for f in self.fpath.glob("*.tif")}

    @property
    def files(self):
        if not hasattr(self, "_files"):
            self._files = {
                f: f for f in self.fpath.rglob("*") if str(f).endswith(".txt")
            }
        return self._files

    def cache_logs(self, root_dir):
        for name, annotation in self.files.items():
            shutil.copy(annotation, root_dir / name.name)
        return True


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
        for name, annotation in self.files.items():
            filepath = root_dir / annotation.getFileName().replace("/", "_")
            if str(filepath).endswith("txt") and not filepath.exists():
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
