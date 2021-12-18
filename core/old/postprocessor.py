import pkg_resources
from pathlib import Path, PosixPath
from typing import Union, List, Dict
from datetime import datetime

from compress_pickle import load, dump

from core.experiment import Experiment
from core.cells import Cells
from core.segment import Tiler
from core.io.matlab import matObject
from core.experiment import Experiment


class PostProcessor:
    '''
    Base class to perform feature extraction.
    :param parameters: Parameters class with channels, reduction functions and
        extraction functions to use.
    :param source: Origin of experiment, if int it is assumed from Omero, if str
        or Path
    '''
    def __init__(self, parameters=None, source: Union[int, str, Path] = None):
        # self.params = parameters
        if source is not None:
            if type(source) is int:
                self.expt_id = source
                self.load_expt(source, omero=True)
            elif type(source) is str or PosixPath:
                self.load_expt(source, omero=False)

    @property
    def channels(self):
        if not hasattr(self, '_channels'):
            if type(self.params.tree) is dict:
                self._channels = tuple(self.params.tree.keys())

        return self._channels

    @property
    def current_position(self):
        assert hasattr(self, 'expt'), 'No experiment loaded.'

        if hasattr(self, 'tiler'):
            assert self.expt.current_position.name == self.tiler.current_position

        return self.expt.current_position

    def load_expt(self, source: Union[int, str], omero: bool = False) -> None:
        if omero:
            self.expt = Experiment.from_source(
                self.expt_id,  #Experiment ID on OMERO
                'upload',  #OMERO Username
                'gothamc1ty',  #OMERO Password
                'islay.bio.ed.ac.uk',  #OMERO host
                port=4064  #This is default
            )
        else:
            self.expt = Experiment.from_source(source)
            self.expt_id = self.expt.exptID

    def load_tiler_cells(self) -> None:
        self.tiler = Tiler(self.expt)
        self.cells = Cells()
        self.cells = self.cells.from_source(
            self.expt.current_position.annotation)

    def get_pos_mo_bud(self):
        annot = self.expt._get_position_annotation(
            self.expt.current_position.name)
        matob = matObject(annot)
        m = matob["timelapseTrapsOmero"].get("cellMothers", None)
        if m is not None:
            ids = np.nonzero(m.todense())
            d = {(self.expt.current_position.name, i, int(m[i, j])): []
                 for i, j in zip(*ids)}
            for i, j, k in zip(*ids, d.keys()):
                d[k].append((self.expt.current_position.name, i, j + 1))
        else:
            print("Pos {} has no mother matrix".format(
                self.expt.current_position.name))
            d = {}

        return d

    def get_exp_mo_bud(self):
        d = {}
        for pos in self.expt.positions:
            self.expt.current_position = pos
            d = {**d, **self.get_pos_mo_bud()}

        self.expt.current_position = self.expt.positions[0]

        return d

    def load_extraction(self, folder=None) -> None:
        if folder is None:
            folder = Path(self.expt.name + '/extraction')

        self.extraction = {}
        for pos in self.expt.positions:
            try:
                self.extraction[pos] = load(folder / Path(pos + '.gz'))

            except:
                print(pos, ' not found')
