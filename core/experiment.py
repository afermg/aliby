"""Core classes for the pipeline"""
import os
import abc
import re
import glob
import json
from pathlib import Path
import logging

import imageio
from tqdm import tqdm
import numpy as np

import omero
from omero.gateway import BlitzGateway

from core.timelapse import TimelapseOMERO, TimelapseLocal

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
    #metadata_parser = AcqMetadataParser()

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

    # Pipelining
    def run(self, keys, store):
        try:
            self.current_position = self.positions[self.position_to_process]
            # Todo: check if we should use the position's id or name
            return ['/'.join(['', self.exptID, self.current_position.name])]
            # Todo: write to store
        except IndexError:
            return None



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

    @property
    def positions(self):
        return list(self._positions.keys())

    def get_position(self, position):
        """Get a Timelapse object for a given position by name"""
        #assert position in self.positions, "Position not available."
        img = self.connection.getObject("Image", self.positions[position])
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
        for annotation in self.dataset.listAnnotations():
            if isinstance(annotation, omero.gateway.FileAnnotationWrapper):
                filepath = save_dir / annotation.getFileName()
                if filepath.stem.endswith('.mat'):
                    mode = 'wb'
                else:
                    mode = 'w'
                with open(str(filepath), mode) as fd:
                    for chunk in annotation:
                        fd.write(chunk)
        # Create caching log
        cache_config = dict(positions=positions, channels=channels,
                            timepoints=timepoints, z_positions=z_positions)
        # TODO save TagAnnotations in Cache config

        with open(str(save_dir / 'cache.config'), 'w') as fd:
            json.dump(cache_config, fd)
        logger.info('Downloaded experiment {}'.format(self.exptID))


class ExperimentLocal(Experiment):
    """
    Experiment class connected to a local file structure.
    It relies on the file structure and file names being organised as follows:

    root_directory
    - {exptID}Acq.txt
    - {exptID}log.txt
    - {posID}
    -- exptID_{timepointID}_{ChannelID}_{ZStackID}.png
    -- exptID_{timepointID}_{ChannelID}_{ZStackID}.png
    -- exptID_{timepointID}_{ChannelID}_{ZStackID}.png
    -- ...
    - {posID}
    -- exptID_{timepointID}_{ChannelID}_{ZStackID}.png
    -- ...
    """
    def __init__(self, root_dir):
        super(ExperimentLocal, self).__init__()
        self.root_dir = Path(root_dir)
        self.exptID = self.root_dir.name
        self.metadata = dict()
        pos, acq_file, log_file, cache = ExperimentLocal.parse_dir_structure(
            self.root_dir)
        self._positions = pos
        self.metadata['acq_file'] = acq_file
        self.metadata['log_file'] = log_file
        if cache is not None:
            with open(cache, 'r') as fd:
                cache_config = json.load(fd)
            self.metadata.update(**cache_config)
        self._current_position = self.get_position(self.positions[0])

    @staticmethod
    def parse_dir_structure(root_dir):
        """
        The images are stored as follows:
        ```
        root_directory
        - {exptID}Acq.txt
        - {exptID}log.txt
        - {posID}
        -- exptID_{timepointID}_{ChannelID}_{z_position_id}.png
        ```

        :param root_dir: The experiment's root directory, organised as
        described above.
        :return:
        """
        positions = sorted([f.name for f in root_dir.iterdir()
                     if (re.search(r'pos[0-9]+$', f.name) and
                         f.is_dir())])
        acq_file = glob.glob(os.path.join(str(root_dir), '*[Aa]cq.txt'))
        log_file = glob.glob(os.path.join(str(root_dir), '*[Ll]og.txt'))
        cache_file = glob.glob(os.path.join(str(root_dir), 'cache.config'))

        acq_file = acq_file[0] if len(acq_file) == 1 else None
        log_file = log_file[0] if len(log_file) == 1 else None
        cache_file = acq_file[0] if len(cache_file) == 1 else None
        return positions, acq_file, log_file, cache_file

    @property
    def positions(self):
        return self._positions

    def get_position(self, position):
        # assert position in self.positions, "Position {} not available in {" \
        #                                    "}.".format(position, self.positions)
        return TimelapseLocal(position, self.root_dir)


