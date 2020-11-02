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
    data = Column(String(50)) # The key to the data in the hdf5 storage file

    cell_id = Column(Integer, ForeignKey('cells.id'))
    cell = relationship("Cell", back_populates="info")

    def __repr__(self):
        return "<CellInfo(id={}, x={}, y={}, t={}, data={}, cell={})>"\
            .format(self.number, self.x, self.y, self.t, self.data,
                    self.cell.number)

