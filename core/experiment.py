"""Core classes for the pipeline"""
import os
import abc
import re
import glob
from pathlib import Path
import logging
import imageio
from tqdm import tqdm
import numpy as np
import json

import omero
from omero.gateway import BlitzGateway
from omero_metadata_parser.extract_acq_metadata import AcqMetadataParser

from timelapse import TimelapseOMERO, TimelapseLocal

logger = logging.getLogger(__name__)


class Experiment:
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
    metadata_parser = AcqMetadataParser()

    def __init__(self):
        self._current_position = None

    def __getitem__(self, item):
        """
        # TODO : Slicing also for the position?

        """
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

    @staticmethod
    def parse_metadata(filename):
        return Experiment.metadata_parser.extract_metadata(filename)

    @abc.abstractproperty
    def positions(self):
        """Returns list of available position names"""
        return

    @abc.abstractmethod
    def get_position(self, position):
        return

    @property
    def current_position(self):
        return self._current_position

    @current_position.setter
    def current_position(self, position):
        self._current_position = self.get_position(position)

    def get_hypercube(self, x, y, z_positions, channels,
                      timepoints):
        return self.current_position.get_hypercube(x, y,
                                                   z_positions, channels,
                                                   timepoints)


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
                           self.dataset.listChildren()}

        # Get annotation Acq file
        try:
            acq_annotation = [item for item in self.dataset.listAnnotations()
                              if (isinstance(
                                    item, omero.gateway.FileAnnotationWrapper)
                                and item.getFileName().endswith('Acq.txt'))][0]
        except IndexError as e:
            raise (e, "No acquisition file found for this experiment")

        with open('acq_file.txt', 'w') as acq_fd:
            # Download and cache the file
            for chunk in acq_annotation.getFileInChunks():
                acq_fd.write(chunk)

        self.metadata = Experiment.parse_metadata('acq_file.txt')
        # TODO use a tempfile?
        os.remove('acq_file.txt')

        # Set up the current position as the first in the list
        self._current_position = self.get_position(self.positions[0])

    @property
    def positions(self):
        return list(self._positions.keys())

    def get_position(self, position):
        """Get a Timelapse object for a given position by name"""
        assert position in self.positions, "Position not available."
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
            for channel in tqdm(channels):
                for tp in tqdm(timepoints):
                    for z_pos in tqdm(z_positions):
                        ch_id = pos.get_channel_index(channel)
                        image = pos.get_hypercube(x=None, y=None,
                                                  width=None,
                                                  height=None,
                                                  channels=[ch_id],
                                                  z_positions=[z_pos],
                                                  timepoints=[tp])

                        im_name = "{}_{:06d}_{}_{:03d}.png".format(self.name,
                                                           tp + 1,
                                                           channel,
                                                           z_pos + 1)
                        imageio.imwrite(str(pos_dir / im_name), np.squeeze(
                            image))

        # Save the file annotations
        # Get annotation Acq file
        try:
            acq_annotation = [item for item in self.dataset.listAnnotations()
                              if (isinstance(
                                    item, omero.gateway.FileAnnotationWrapper)
                                and item.getFileName().endswith('Acq.txt'))][0]
        except IndexError as e:
            raise (e, "No acquisition file found for this experiment")

        filepath = save_dir / acq_annotation.getFileName()
        with open(str(filepath), 'w') as acq_fd:
            # Download the file
            for chunk in acq_annotation.getFileInChunks():
                acq_fd.write(chunk)

        # Create caching log
        cache_config = dict(positions=positions, channels=channels,
                            timepoints=timepoints, z_positions=z_positions)

        with open(str(save_dir / 'cache.config'), 'w') as fd:
            json.dump(cache_config, fd)
        logger.info('Downloaded experiment {}'.format(self.exptID))

class ExperimentLocal(Experiment):
    """
    Experiment class connected to a local file structure.
    """
    def __init__(self, root_dir):
        self.root_dir = Path(root_dir)
        self.exptID = self.root_dir.name
        pos, acq_file, cache = ExperimentLocal.parse_dir_structure(self.root_dir)
        self._positions = pos
        self.metadata = Experiment.parse_metadata(acq_file)
        if cache is not None:
            with open(cache, 'r') as fd:
                cache_config = json.load(fd)
            self.cache_config(cache_config)
        self._current_position = self.get_position(self.positions[0])


    @staticmethod
    def parse_dir_structure(root_dir):
        """
        The images are stored as follows:
        ```
        root_directory
        - {exptID}Acq.txt
        - {exptID}log.txt
        - pos001
        -- exptID_{timepointID}_{ChannelID}_{z_position_id}.png
        ```

        :param root_dir: The experiment's root directory, organised as
        described above.
        :return:
        """
        positions = [f.name for f in root_dir.iterdir()
                     if (re.search(r'pos[0-9]+$', f.name) and
                         f.is_dir())]
        acq_file = glob.glob(os.path.join(str(root_dir), '*[Aa]cq.txt'))[0]

        cache_file = glob.glob(os.path.join(str(root_dir), 'cache.config'))
        if len(cache_file) == 1:
            cache_file = cache_file[0]
        else:
            cache_file = None
        return positions, acq_file, cache_file

    def cache_config(self, cache):
        self.metadata.positions = self.metadata.positions[
                    self.metadata.positions.name.isin(cache['positions'])]
        self.metadata.channels = self.metadata.channels[
                    self.metadata.channels.names.isin(cache['channels'])]
        ntimepoints = len(cache['timepoints'])
        totalduration = ntimepoints*self.metadata.times['interval']
        self.metadata.times.update(dict(ntimepoints=ntimepoints,
                                        totalduration=totalduration))

        diffs = np.unique([cache['z_positions'][i+1] - cache['z_positions'][i]
                            for i in range(len(cache['z_positions']) - 1)])
        if len(diffs) != 1:
            self.metadata.zsections.spacing = np.nan
        else:
            self.metadata.zsections.spacing = \
                self.metadata.zsections.spacing* diffs[0]

        self.metadata.zsections.sections = len(cache['z_positions'])

    @property
    def positions(self):
        return self._positions

    def get_position(self, position):
        assert position in self.positions, "Position {} not available in {" \
                                           "}.".format(position, self.positions)
        return TimelapseLocal(position, self.root_dir, self.metadata)


Experiment.register(ExperimentOMERO)
Experiment.register(ExperimentLocal)

