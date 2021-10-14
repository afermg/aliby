"""Pipeline results classes and utilities"""


class SegmentationResults:
    """
    Object storing the data from the Segmentation pipeline.
    Everything is stored as an `AttributeDict`, which is a `defaultdict` where
    you can get elements as attributes.

    In addition, it implements:
     - IO functionality (read from file, write to file)
    """
    def __init__(self, raw_expt):
        pass




class CellResults:
    """
    Results on a set of cells TODO: what set of cells, how?

    Contains:
    * cellInf describing which cells are taken into account
    * annotations on the cell
    * segmentation maps of the cell TODO: how to define and save this?
    * trapLocations TODO: why is this not part of cellInf?
    """

    def __init__(self, cellInf=None, annotations=None, segmentation=None,
                 trapLocations=None):
        self._cellInf = cellInf
        self._annotations = annotations
        self._segmentation = segmentation
        self._trapLocations = trapLocations
