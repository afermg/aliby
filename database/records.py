from functools import lru_cache
import numpy as np
from sqlalchemy import Column, Sequence, String, Integer, ForeignKey, Float

from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship

Base = declarative_base()


class Position(Base):
    """
    Describes a position by name and how many timepoints it has.
    Has a relationship with the traps and the drift of the position over time.
    """
    __tablename__ = "positions"
    id = Column(Integer, Sequence('position_id_sequence'), primary_key=True)
    name = Column(String(6))
    n_timepoints = Column(Integer)
    traps = relationship("Trap", back_populates="position")
    drifts = relationship("Drift", back_populates="position")

    def __repr__(self):
        return "<Position(name={}, n_timepoints={})>".format(self.name,
                                                             self.n_timepoints)

    @property
    def trap_locations(self):
        # Todo: get all traps + drifts and create TrapLocations class
        #   save to an attribute so only need to get once.
        pass


class Trap(Base):
    """
    Describes a trap by its initial location in space.
    Has a relationship with the position it is in.
    Has a relationship with the cells available in that trap.
    """
    __tablename__ = "traps"
    id = Column(Integer, Sequence('trap_id_sequence'), primary_key=True)
    number = Column(Integer)
    x = Column(Integer)
    y = Column(Integer)
    size = Column(Integer)

    pos_id = Column(Integer, ForeignKey('positions.id'))
    position = relationship("Position", back_populates="traps")

    cells = relationship("Cell", back_populates="trap")

    def __repr__(self):
        return "<Trap(id={}, position={}, x={}, y={}, size={})>".format(
            self.number, self.position.name, self.x, self.y, self.size)


class Drift(Base):
    """
    Describe the drift of a position over time.
    Has a relationship with the position it describes.
    """
    __tablename__ = "drifts"
    id = Column(Integer, Sequence('drift_id_sequence'), primary_key=True)
    x = Column(Integer)
    y = Column(Integer)
    t = Column(Integer)

    pos_id = Column(Integer, ForeignKey('positions.id'))
    position = relationship("Position", back_populates="drifts")

    def __repr__(self):
        return "<Drifts(x={}, y={}, t={}, position={})>"\
            .format(self.x, self.y, self.t, self.position.name)


class Cell(Base):
    """
    Describes the cells by number (within the cells of a trap).
    Has a relationship to the trap in which is is.
    Has a relationship to Cell Information for each time point.
    """
    __tablename__ = "cells"
    id = Column(Integer, Sequence('cell_id_sequence'), primary_key=True)
    number = Column(Integer)

    trap_id = Column(Integer, ForeignKey('traps.id'))
    trap = relationship("Trap", back_populates="cells")

    info = relationship("CellInfo", back_populates="cell")

    def __repr__(self):
        return "<Cell(id={}, trap={})>".format(self.number, self.trap.number)

    @property
    @lru_cache
    def cell_info(self):
        """
        Returns the cell info in ordered manner (by time)
        :return: self.info, sorted.
        """
        return sorted(self.info, key=lambda x: x.t)

    def edge(self, store):
        """
        The edge masks of this cell over time, in dimensions (time, x, y)
        :param store: The data store to use for extracting masks.
        :return: np.ndarray in dimensions (time, x, y) of cell outline masks.
        """
        return np.stack([t.edge(store) for t in self.cell_info], axis=0)

    def mask(self, store):
        """
        Extracts a mask of the full cell out of the store.
        If you need only the outline, use Cell.edge
        :param store: The data store from which to extract masks
        :return: np.ndarray in dimensions (time, x, y) of cell masks.
        """
        return np.stack([t.mask(store) for t in self.cell_info], axis=0)

    @property
    @lru_cache
    def timepoints(self):
        """
        The timepoints
        :return: A list of timepoints. The first dimension of all other
        computed attributes corresponds to these timepoints.
        """
        # Todo: save in an attribute so that only need to get it once
        # Todo: validate if contiguous to extract tracking errors.
        return [x.t for x in self.cell_info]


class CellInfo(Base):
    """
    Describes information about a given cell at a given timepoint: position
    (x,y) within the trap, timepoint, as well as a data field which points
    to the group in the HDF5 store where the cell's radial coordiantes,
    edge_mask, and other information can be found.
    Has a relationship with the cell it relates to.
    """
    __tablename__ = "cell_info"
    id = Column(Integer, Sequence('cell_id_sequence'), primary_key=True)
    number = Column(Integer)
    x = Column(Float)
    y = Column(Float)
    t = Column(Integer)
    data = Column(String(50))# The key to the data in the hdf5 storage file

    cell_id = Column(Integer, ForeignKey('cells.id'))
    cell = relationship("Cell", back_populates="info")

    def __repr__(self):
        return "<CellInfo(id={}, x={}, y={}, t={}, data={}, cell={})>"\
            .format(self.number, self.x, self.y, self.t, self.data,
                    self.cell.number)

    @lru_cache
    def edge(self, store):
        key = self.data
        try:
            edge = store[key + 'edge_mask']
        except KeyError as e:
            # Todo log e?
            radii = store[key + 'radii']
            angles = store[key + 'angles']
            # Todo Active contour using the radii and angles
            edge = None
        return edge

    @lru_cache
    def mask(self, store):
        edge = self.edge(store)
        # Todo fill the image using opencv
        filled = edge
        return filled