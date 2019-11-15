"""
Objects describing an experiment.
"""


class Experiemnt:
    """
    Experiment object.

    Contains:
    * A set of timelapses
    * A metadata structure
    * Parameters for TODO: what are the parameters for
    * A Logger TODO: is logger a class, function, or method
    """

    def __init__(self, timelapses=None, metadata=None,
                 params=None, logger=None):
        self._timelapses = timelapses
        self._metadata = metadata
        self._params = params
        self._logger = logger
        pass

    @property
    def timelapses(self):
        return self._timelapses

    @timelapses.setter
    def timelapses(self, timelapses):
        self._timelapses = timelapses

    @property
    def metadata(self):
        return self._metadata

    @metadata.setter
    def metadata(self, metadata):
        self._metadata = metadata

    @property
    def params(self):
        return self._params

    @params.setter
    def params(self, params):
        self._params = params

    @property
    def logger(self):
        return self._logger

    @logger.setter
    def logger(self, logger):
        self._logger = logger


class Timelapses:
    """
    Timelapse object.

    Contains properties:
    * Trap locations
    * Cell IDs and outlines
    * Cell annotations
    And methods (?):
    * Image preprocess
    """

    def __init__(self):
        pass


class Stats:
    """
    Statistics for an experiment.
    Relies on an omero dataset.

    Contains properties:
    * tracking (?)
    * histograms
    * flatfield
    * darkfield
    * aperture
    """

    def __init__(self):
        pass
