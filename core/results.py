"""Cell results class and utilities"""


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
