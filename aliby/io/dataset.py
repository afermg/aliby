#!/usr/bin/env python3
import shutil
from pathlib import Path, PosixPath
from typing import Union

import omero

from aliby.io.image import ImageLocal
from aliby.io.omero import Argo


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


class Dataset(Argo):
    def __init__(self, expt_id, **server_info):
        super().__init__(**server_info)
        self.expt_id = expt_id
        self._files = None

    @property
    def dataset(self):
        return self.conn.getObject("Dataset", self.expt_id)

    @property
    def name(self):
        return self.dataset.getName()

    @property
    def date(self):
        return self.dataset.getDate()

    @property
    def unique_name(self):
        return "_".join(
            (
                str(self.expt_id),
                self.date.strftime("%Y_%m_%d").replace("/", "_"),
                self.name,
            )
        )

    def get_images(self):
        return {im.getName(): im.getId() for im in self.dataset.listChildren()}

    @property
    def files(self):
        if self._files is None:
            self._files = {
                x.getFileName(): x
                for x in self.dataset.listAnnotations()
                if isinstance(x, omero.gateway.FileAnnotationWrapper)
            }
        if not len(self._files):
            raise Exception(
                "exception:metadata: experiment has no annotation files."
            )
        return self._files

    @property
    def tags(self):
        if self._tags is None:
            self._tags = {
                x.getname(): x
                for x in self.dataset.listAnnotations()
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
