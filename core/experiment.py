"""Core classes for the pipeline"""
import itertools
import os
import abc
import re
import glob
import json
from pathlib import Path
import logging
from typing import Iterable, Union

import cv2
import imageio
from tqdm import tqdm
import numpy as np

import omero
from omero.gateway import BlitzGateway
from logfile_parser import Parser

from core.timelapse import TimelapseOMERO, TimelapseLocal
from core.utils import accumulate
from database.records import Position

logger = logging.getLogger(__name__)


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
        self.exptID = ''
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
            logger.debug('ExperimentOMERO: {}'.format(args, kwargs))
            return ExperimentOMERO(*args, **kwargs)
        else:
            logger.debug('ExperimentLocal: {}'.format(args, kwargs))
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

    def get_hypercube(self, x, y, z_positions, channels,
                      timepoints):
        return self.current_position.get_hypercube(x, y,
                                                   z_positions, channels,
                                                   timepoints)


# Todo: cache images like in ExperimentLocal
class ExperimentOMERO(Experiment):
    """
    Experiment class to organise different timelapses.
    Connected to a Dataset object which handles database I/O.
    """

    def __init__(self, exptID, username, password, host, port=4064, **kwargs):
        super(ExperimentOMERO, self).__init__()
        self.exptID = exptID
        self.connection = BlitzGateway(username, password, host=host,
                                       port=port)

        connected = self.connection.connect()
        assert connected is True, "Could not connect to server."
        self.dataset = self.connection.getObject("Dataset", self.exptID)
        self.name = self.dataset.getName()

        self._positions = {img.getName(): img.getId() for img in
                           sorted(self.dataset.listChildren(),
                                  key=lambda x: x.getName())}

        # Set up the current position as the first in the list
        self._current_position = self.get_position(self.positions[0])

        self.save_dir = Path(kwargs.get('save_dir', './')) / self.name
        if not self.save_dir.exists():
            self.save_dir.mkdir(parents=True)
        self.running_tp = 0

    @property
    def positions(self):
        return list(self._positions.keys())

    def get_position(self, position):
        """Get a Timelapse object for a given position by name"""
        # assert position in self.positions, "Position not available."
        img = self.connection.getObject("Image", self._positions[position])
        return TimelapseOMERO(img)

    def cache_locally(self, root_dir='./', positions=None, channels=None,
                      timepoints=None, z_positions=None):
        """
        Save the experiment locally.

        :param root_dir: The directory in which the experiment will be
        saved. The experiment will be a subdirectory of "root_directory"
        and will be named by its id.
        """
        logger.warning('Saving experiment {}; may take some time.'.format(
            self.name))

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
        print(save_dir)
        # Save the images
        for pos_name in tqdm(positions):
            pos = self.get_position(pos_name)
            pos_dir = save_dir / pos_name
            if not pos_dir.exists():
                pos_dir.mkdir()
            self.cache_set(pos, range(pos.size_t))

        self.cache_annotations(save_dir)
        # Save the file annotations
        cache_config = dict(positions=positions, channels=channels,
                            timepoints=timepoints, z_positions=z_positions)
        with open(str(save_dir / 'cache.config'), 'w') as fd:
            json.dump(cache_config, fd)
        logger.info('Downloaded experiment {}'.format(self.exptID))

    # Todo: turn this static
    def cache_annotations(self, save_dir, **kwargs):
        # Save the file annotations
        save_mat = kwargs.get('save_mat', False)
        tags = dict()# and the tag annotations
        for annotation in self.dataset.listAnnotations():
            if isinstance(annotation, omero.gateway.FileAnnotationWrapper):
                filepath = save_dir / annotation.getFileName()
                if save_mat or not str(filepath).endswith('mat') and not filepath.exists():
                    with open(str(filepath), 'wb') as fd:
                        for chunk in annotation.getFileInChunks():
                            fd.write(chunk)
            if isinstance(annotation, omero.gateway.TagAnnotationWrapper):
                # TODO save TagAnnotations in tags dictionary
                key = annotation.getDescription()
                if key == '':
                    key = 'misc. tags'
                if key in tags:
                    if not isinstance(tags[key], list):
                        tags[key] = [tags[key]]
                    tags[key].append(annotation.getValue())
                else:
                    tags[key] = annotation.getValue()
        with open(str(save_dir / 'omero_tags.json'), 'w') as fd:
            json.dump(tags, fd)
        return

    # Todo: turn this static
    def cache_set(self, save_dir, position: TimelapseOMERO,
                  timepoints: Iterable[int], db_pos, **kwargs):
        # Todo: save one time point to file
        #       save it under self.save_dir / self.exptID / self.position
        #       save each channel, z_position separately
        pos_dir = save_dir / position.name
        if not pos_dir.exists():
            pos_dir.mkdir()
        for tp in tqdm(timepoints):
            for channel in tqdm(position.channels):
                for z_pos in tqdm(range(position.size_z)):
                    ch_id = position.get_channel_index(channel)
                    image = position.get_hypercube(x=None, y=None,
                                                   channels=[ch_id],
                                                   z_positions=[z_pos],
                                                   timepoints=[tp])

                    im_name = "{}_{:06d}_{}_{:03d}.png".format(self.name,
                                                               tp + 1,
                                                               channel,
                                                               z_pos + 1)
                    cv2.imwrite(str(pos_dir / im_name), np.squeeze(
                        image))
            db_pos.n_timepoints = tp
        return list(itertools.product([position.name], timepoints))

    def run(self, keys: Union[list, int], session, **kwargs):
        if self.running_tp == 0:
            self.cache_annotations(self.save_dir)
        if isinstance(keys, list):
            # tells you how many time points to do at once
            keys = len(keys)
        # Locally save `keys` images at a time for each position
        cached = []
        for pos_name in self.positions:
            db_pos = session.query(Position).filter_by(name=pos_name).first()
            if db_pos is None:
                db_pos = Position(name=pos_name, n_timepoints=0)
                session.add(db_pos)
            position = self.get_position(pos_name)
            timepoints = list(range(db_pos.n_timepoints,
                                    min(db_pos.n_timepoints + keys,
                                        position.size_t)))
            if len(timepoints) > 0 and db_pos.n_timepoints < max(timepoints):
                try:
                    cached += self.cache_set(self.save_dir, position,
                                         timepoints, db_pos, **kwargs)
                finally:
                    # Add position to storage
                    session.commit()
        self.running_tp += keys  # increase by number of processed time points
        return cached


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
        acq_parser = Parser('multiDGUI_acq_format')
        with open(acq_file, 'r') as fd:
            metadata = acq_parser.parse(fd)
        self.metadata = metadata
        self.metadata['finished'] = finished
        if self.finished:
            cache = self._find_cache()
            # log = self._find_log() # Todo: add log metadata
            if cache is not None:
                with open(cache, 'r') as fd:
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
        file = self._find_file('*[Aa]cq.txt')
        if file is None:
            raise ValueError('Cannot load this experiment. There are either '
                             'too many or too few acq files.')
        return file

    def _find_cache(self):
        return self._find_file('cache.config')

    @property
    def finished(self):
        return self.metadata['finished']

    @property
    def running(self):
        return not self.metadata['finished']

    @property
    def positions(self):
        return self.metadata['positions']['posname']

    def get_position(self, position):
        if position not in self._pos_mapper:
            self._pos_mapper[position] = TimelapseLocal(position,
                                                        self.root_dir,
                                                        finished=self.finished)
        return self._pos_mapper[position]

    def run(self, keys, session=None, **kwargs):
        """

        :param keys: List of (position, time point) tuples to process.
        :return:
        """
        for pos, tps in accumulate(keys):
            self.get_position(pos).run(tps, session=session)
        session.commit()
        # Todo: if keys is none, get the positions table from the session
        #  and save it to the store, then return None
        return keys
