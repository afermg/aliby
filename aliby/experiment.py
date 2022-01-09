"""Core classes for the pipeline"""
import atexit
import itertools
import os
import abc
import glob
import json
import warnings
from getpass import getpass
from pathlib import Path
import re
import logging
from typing import Union

import h5py
from tqdm import tqdm
import pandas as pd

import omero
from omero.gateway import BlitzGateway
from logfile_parser import Parser

from agora.io.utils import accumulate
from aliby.timelapse import TimelapseOMERO, TimelapseLocal

from agora.io.metadata_parser import parse_logfiles
from agora.io.writer import Writer

logger = logging.getLogger(__name__)

########################### Dask objects ###################################
##################### ENVIRONMENT INITIALISATION ################
import omero
from omero.gateway import BlitzGateway, PixelsWrapper
from omero.model import enums as omero_enums
import numpy as np

# Set up the pixels so that we can reuse them across sessions (?)
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


class NonCachedPixelsWrapper(PixelsWrapper):
    """Extend gateway.PixelWrapper to override _prepareRawPixelsStore."""

    def _prepareRawPixelsStore(self):
        """
        Creates RawPixelsStore and sets the id etc
        This overrides the superclass behaviour to make sure that
        we don't re-use RawPixelStore in multiple processes since
        the Store may be closed in 1 process while still needed elsewhere.
        This is needed when napari requests may planes simultaneously,
        e.g. when switching to 3D view.
        """
        ps = self._conn.c.sf.createRawPixelsStore()
        ps.setPixelsId(self._obj.id.val, True, self._conn.SERVICE_OPTS)
        return ps


omero.gateway.PixelsWrapper = NonCachedPixelsWrapper
# Update the BlitzGateway to use our NonCachedPixelsWrapper
omero.gateway.refreshWrappers()


######################  DATA ACCESS ###################
import dask.array as da
from dask import delayed


def get_data_lazy(image) -> da.Array:
    """Get 5D dask array, with delayed reading from OMERO image."""
    nt, nc, nz, ny, nx = [getattr(image, f"getSize{x}")() for x in "TCZYX"]
    pixels = image.getPrimaryPixels()
    dtype = PIXEL_TYPES.get(pixels.getPixelsType().value, None)
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


class MetaData:
    """Small metadata Process that loads log."""

    def __init__(self, log_dir, store):
        self.log_dir = log_dir
        self.store = store

    def load_logs(self):
        parsed_flattened = parse_logfiles(self.log_dir)
        return parsed_flattened

    def run(self):
        metadata_writer = Writer(self.store)
        metadata_dict = self.load_logs()
        metadata_writer.write(path="/", meta=metadata_dict, overwrite=False)


########################### Old Objects ####################################


class Experiment(abc.ABC):
    """
    Abstract base class for experiments.
    Gives all the functions that need to be implemented in both the local
    version and the Omero version of the Experiment class.

    As this is an abstract class, experiments can not be directly instantiated
    through the usual `__init__` function, but must be instantiated from a
    source.
    >>> expt = Experiment.from_source(root_directory)
    Data from the current timelapse can be obtained from the experiment using
    colon and comma separated slicing.
    The order of data is C, T, X, Y, Z
    C, T and Z can have any slice
    X and Y will only consider the beginning and end as we want the images
    to be continuous
    >>> bf_1 = expt[0, 0, :, :, :] # First channel, first timepoint, all x,y,z
    """

    __metaclass__ = abc.ABCMeta

    # metadata_parser = AcqMetadataParser()

    def __init__(self):
        self.exptID = ""
        self._current_position = None
        self.position_to_process = 0

    def __getitem__(self, item):
        return self.current_position[item]

    @property
    def shape(self):
        return self.current_position.shape

    @staticmethod
    def from_source(*args, **kwargs):
        """
        Factory method to construct an instance of an Experiment subclass (
        either ExperimentOMERO or ExperimentLocal).

        :param source: Where the data is stored (OMERO server or directory
        name)
        :param kwargs: If OMERO server, `user` and `password` keyword
        arguments are required. If the data is stored locally keyword
        arguments are ignored.
        """
        if len(args) > 1:
            logger.debug("ExperimentOMERO: {}".format(args, kwargs))
            return ExperimentOMERO(*args, **kwargs)
        else:
            logger.debug("ExperimentLocal: {}".format(args, kwargs))
            return ExperimentLocal(*args, **kwargs)

    @property
    @abc.abstractmethod
    def positions(self):
        """Returns list of available position names"""
        return

    @abc.abstractmethod
    def get_position(self, position):
        return

    @property
    def current_position(self):
        return self._current_position

    @property
    def channels(self):
        return self._current_position.channels

    @current_position.setter
    def current_position(self, position):
        self._current_position = self.get_position(position)

    def get_hypercube(self, x, y, z_positions, channels, timepoints):
        return self.current_position.get_hypercube(
            x, y, z_positions, channels, timepoints
        )


# Todo: cache images like in ExperimentLocal
class ExperimentOMERO(Experiment):
    """
    Experiment class to organise different timelapses.
    Connected to a Dataset object which handles database I/O.
    """

    def __init__(self, omero_id, host, port=4064, **kwargs):
        super(ExperimentOMERO, self).__init__()
        self.exptID = omero_id
        # Get annotations
        self.use_annotations = kwargs.get("use_annotations", True)
        self._files = None
        self._tags = None

        # Create a connection
        self.connection = BlitzGateway(
            kwargs.get("username") or input("Username: "),
            kwargs.get("password") or getpass("Password: "),
            host=host,
            port=port,
        )
        connected = self.connection.connect()
        try:
            assert connected is True, "Could not connect to server."
        except AssertionError as e:
            self.connection.close()
            raise (e)
        try:  # Run everything that could cause the initialisation to fail
            self.dataset = self.connection.getObject("Dataset", self.exptID)
            self.name = self.dataset.getName()
            # Create positions objects
            self._positions = {
                img.getName(): img.getId()
                for img in sorted(
                    self.dataset.listChildren(), key=lambda x: x.getName()
                )
            }
            # Set up local cache
            self.root_dir = Path(kwargs.get("save_dir", "./")) / self.name
            if not self.root_dir.exists():
                self.root_dir.mkdir(parents=True)
            self.compression = kwargs.get("compression", None)
            self.image_cache = h5py.File(self.root_dir / "images.h5", "a")

            # Set up the current position as the first in the list
            self._current_position = self.get_position(self.positions[0])
            self.running_tp = 0
        except Exception as e:
            # Close the connection!
            print("Error in initialisation, closing connection.")
            self.connection.close()
            print(self.connection.isConnected())
            raise e
        atexit.register(self.close)  # Close everything if program ends

    def close(self):
        print("Clean-up on exit.")
        self.image_cache.close()
        self.connection.close()

    @property
    def files(self):
        if self._files is None:
            self._files = {
                x.getFileName(): x
                for x in self.dataset.listAnnotations()
                if isinstance(x, omero.gateway.FileAnnotationWrapper)
            }
        return self._files

    @property
    def tags(self):
        if self._tags is None:
            self._tags = {
                x.getName(): x
                for x in self.dataset.listAnnotations()
                if isinstance(x, omero.gateway.TagAnnotationWrapper)
            }
        return self._tags

    @property
    def positions(self):
        return list(self._positions.keys())

    def _get_position_annotation(self, position):
        # Get file annotations filtered by position name and ordered by
        # creation date
        r = re.compile(position)
        wrappers = sorted(
            [self.files[key] for key in filter(r.match, self.files)],
            key=lambda x: x.creationEventDate(),
            reverse=True,
        )
        # Choose newest file
        if len(wrappers) < 1:
            return None
        else:
            # Choose the newest annotation and cache it
            annotation = wrappers[0]
            filepath = self.root_dir / annotation.getFileName().replace("/", "_")
            if not filepath.exists():
                with open(str(filepath), "wb") as fd:
                    for chunk in annotation.getFileInChunks():
                        fd.write(chunk)
            return filepath

    def get_position(self, position):
        """Get a Timelapse object for a given position by name"""
        # assert position in self.positions, "Position not available."
        img = self.connection.getObject("Image", self._positions[position])
        if self.use_annotations:
            annotation = self._get_position_annotation(position)
        else:
            annotation = None
        return TimelapseOMERO(img, annotation, self.image_cache)

    def cache_locally(
        self,
        root_dir="./",
        positions=None,
        channels=None,
        timepoints=None,
        z_positions=None,
    ):
        """
        Save the experiment locally.

        :param root_dir: The directory in which the experiment will be
        saved. The experiment will be a subdirectory of "root_directory"
        and will be named by its id.
        """
        logger.warning("Saving experiment {}; may take some time.".format(self.name))

        if positions is None:
            positions = self.positions
        if channels is None:
            channels = self.current_position.channels
        if timepoints is None:
            timepoints = range(self.current_position.size_t)
        if z_positions is None:
            z_positions = range(self.current_position.size_z)

        save_dir = Path(root_dir) / self.name
        if not save_dir.exists():
            save_dir.mkdir()
        # Save the images
        for pos_name in tqdm(positions):
            pos = self.get_position(pos_name)
            pos_dir = save_dir / pos_name
            if not pos_dir.exists():
                pos_dir.mkdir()
            self.cache_set(pos, range(pos.size_t))

        self.cache_logs(save_dir)
        # Save the file annotations
        cache_config = dict(
            positions=positions,
            channels=channels,
            timepoints=timepoints,
            z_positions=z_positions,
        )
        with open(str(save_dir / "cache.config"), "w") as fd:
            json.dump(cache_config, fd)
        logger.info("Downloaded experiment {}".format(self.exptID))

    def cache_logs(self, **kwargs):
        # Save the file annotations
        tags = dict()  # and the tag annotations
        for annotation in self.dataset.listAnnotations():
            if isinstance(annotation, omero.gateway.FileAnnotationWrapper):
                filepath = self.root_dir / annotation.getFileName().replace("/", "_")
                if str(filepath).endswith("txt") and not filepath.exists():
                    # Save only the text files
                    with open(str(filepath), "wb") as fd:
                        for chunk in annotation.getFileInChunks():
                            fd.write(chunk)
            if isinstance(annotation, omero.gateway.TagAnnotationWrapper):
                key = annotation.getDescription()
                if key == "":
                    key = "misc. tags"
                if key in tags:
                    if not isinstance(tags[key], list):
                        tags[key] = [tags[key]]
                    tags[key].append(annotation.getValue())
                else:
                    tags[key] = annotation.getValue()
        with open(str(self.root_dir / "omero_tags.json"), "w") as fd:
            json.dump(tags, fd)
        return

    def run(self, keys: Union[list, int], store, **kwargs):
        if self.running_tp == 0:
            self.cache_logs(**kwargs)
            self.running_tp = 1  # Todo rename based on annotations
        run_tps = dict()
        for pos, tps in accumulate(keys):
            position = self.get_position(pos)
            run_tps[pos] = position.run(tps, store, save_dir=self.root_dir)
        # Update the keys to match what was actually run
        keys = [(pos, tp) for pos in run_tps for tp in run_tps[pos]]
        return keys


class ExperimentLocal(Experiment):
    def __init__(self, root_dir, finished=True):
        super(ExperimentLocal, self).__init__()
        self.root_dir = Path(root_dir)
        self.exptID = self.root_dir.name
        self._pos_mapper = dict()
        # Fixme: Made the assumption that the Acq file gets saved before the
        #  experiment is run and that the information in that file is
        #  trustworthy.
        acq_file = self._find_acq_file()
        acq_parser = Parser("multiDGUI_acq_format")
        with open(acq_file, "r") as fd:
            metadata = acq_parser.parse(fd)
        self.metadata = metadata
        self.metadata["finished"] = finished
        self.files = [f for f in self.root_dir.iterdir() if f.is_file()]
        self.image_cache = h5py.File(self.root_dir / "images.h5", "a")
        if self.finished:
            cache = self._find_cache()
            # log = self._find_log() # Todo: add log metadata
            if cache is not None:
                with open(cache, "r") as fd:
                    cache_config = json.load(fd)
                self.metadata.update(**cache_config)
        self._current_position = self.get_position(self.positions[0])

    def _find_file(self, regex):
        file = glob.glob(os.path.join(str(self.root_dir), regex))
        if len(file) != 1:
            return None
        else:
            return file[0]

    def _find_acq_file(self):
        file = self._find_file("*[Aa]cq.txt")
        if file is None:
            raise ValueError(
                "Cannot load this experiment. There are either "
                "too many or too few acq files."
            )
        return file

    def _find_cache(self):
        return self._find_file("cache.config")

    @property
    def finished(self):
        return self.metadata["finished"]

    @property
    def running(self):
        return not self.metadata["finished"]

    @property
    def positions(self):
        return self.metadata["positions"]["posname"]

    def _get_position_annotation(self, position):
        r = re.compile(position)
        files = list(filter(lambda x: r.match(x.stem), self.files))
        if len(files) == 0:
            return None
        files = sorted(files, key=lambda x: x.lstat().st_ctime, reverse=True)
        # Get the newest and return as string
        return files[0]

    def get_position(self, position):
        if position not in self._pos_mapper:
            annotation = self._get_position_annotation(position)
            self._pos_mapper[position] = TimelapseLocal(
                position,
                self.root_dir,
                finished=self.finished,
                annotation=annotation,
                cache=self.image_cache,
            )
        return self._pos_mapper[position]

    def run(self, keys, store, **kwargs):
        """

        :param keys: List of (position, time point) tuples to process.
        :return:
        """
        run_tps = dict()
        for pos, tps in accumulate(keys):
            run_tps[pos] = self.get_position(pos).run(tps, store)
        # Update the keys to match what was actually run
        keys = [(pos, tp) for pos in run_tps for tp in run_tps[pos]]
        return keys
