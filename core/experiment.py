"""Core classes for the pipeline"""
import os
import abc
import re
import glob
from pathlib import Path

from omero_metadata_parser.extract_acq_metadata import AcqMetadataParser

import omero
from omero.gateway import BlitzGateway

from timelapse import TimelapseOMERO, TimelapseLocal


class Experiment:
    """
    Abstract base class for experiments.
    Gives all the functions that need to be implemented in both the local
    version and the Omero version of the Experiment class.
    """
    __metaclass__ = abc.ABCMeta
    metadata_parser = AcqMetadataParser()

    def __init__(self):
        self._current_position = None

    @staticmethod
    def from_source(source, **kwargs):
        """
        Factory method to construct an instance of an Experiment subclass (
        either ExperimentOMERO or ExperimentLocal).

        :param source: Where the data is stored (OMERO server or directory
        name)
        :param kwargs: If OMERO server, `user` and `password` keyword
        arguments are required. If the data is stored locally keyword
        arguments are ignored.
        """

        pass

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

    def get_hypercube(self, x, y, width, height, z_positions, channels,
                      timepoints):
        return self.current_position.get_hypercube(x, y, width, height,
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

        self._positions = {img.getName(): img.getId() for img in
                           self.dataset.listChildren()}

        # Get annotation Acq file (cached as a tmp file)
        try:
            acq_annotation = [item for item in self.dataset.listAnnotations()
                              if (isinstance(item,
                                             omero.gateway.FileAnnotationWrapper)
                                  and item.getFileName().endswith('Acq.txt'))][
                0]
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


class ExperimentLocal(Experiment):
    """
    Experiment class connected to a local file structure.
    """
    def __init__(self, root_dir):
        self.root_dir = Path(root_dir)
        self.exptID = self.root_dir.name
        pos, acq_file = ExperimentLocal.parse_dir_structure(self.root_dir)
        self._positions = pos
        self.metadata = Experiment.parse_metadata(acq_file)
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

        return positions, acq_file


    @property
    def positions(self):
        return self._positions

    def get_position(self, position):
        assert position in self.positions, "Position {} not available in {" \
                                           "}.".format(position, self.positions)
        return TimelapseLocal(position, self.root_dir, self.metadata)


Experiment.register(ExperimentOMERO)
Experiment.register(ExperimentLocal)

