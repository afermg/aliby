"""Define classes for a tile and all tiles."""

import typing as t

import numpy as np


class TileLocations:
    """Store each tile as an instance of Tile."""

    def __init__(
        self,
        initial_location: np.array,
        tile_size: int or list[int] = None,
        max_size: int or list[int] = 1200,
        drifts: np.array = None,
    ):
        """
        Initialise tiles as an array of Tile objects.

        Parameters
        ----------
        initial_location: array
            An array of tile centres.
        tile_size: int or list[int]
            Length of one (if square) or both sides of a tile.
        max_size: int or list[int], optional
            Maximum size of a tile. Default is 1200.
        drifts: array
            An array of translations to correct drift of the microscope.
        """
        if drifts is None:
            drifts = []
        if isinstance(tile_size, int):
            max_size = (tile_size, max_size)
        self.tile_size = tile_size
        if isinstance(max_size, int):
            max_size = (max_size, max_size)
        self.max_size = max_size
        self.initial_location = initial_location
        self.tiles = [
            Tile(centre, self, tile_size or max_size, max_size)
            for centre in initial_location
        ]
        self.drifts = drifts

    def __len__(self):
        """Find number of tiles."""
        return len(self.tiles)

    def __iter__(self):
        """Return the next tile from the list of tiles."""
        yield from self.tiles

    @property
    def shape(self):
        """Return the number of tiles and the number of drifts."""
        return len(self.tiles), len(self.drifts)

    def to_dict(self, tp: int):
        """
        Export initial locations, tile_size, max_size, and drifts as a dict.

        Parameters
        ----------
        tp: integer
            An index for a time point
        """
        res = dict()
        if tp == 0:
            res["trap_locations"] = self.initial_location
            res["attrs/tile_size"] = self.tile_size
            res["attrs/max_size"] = self.max_size
        res["drifts"] = np.expand_dims(self.drifts[tp], axis=0)
        return res

    def centres_at_time(self, tp: int) -> np.ndarray:
        """Return an array of tile centres (x- and y-coords)."""
        return np.array([tile.centre_at_time(tp) for tile in self.tiles])

    @classmethod
    def from_tiler_init(
        cls,
        initial_location,
        tile_size: int = None,
        max_size: int = 1200,
    ):
        """Instantiate from a Tiler."""
        return cls(initial_location, tile_size, max_size, drifts=[])


class Tile:
    """Define a tile."""

    def __init__(
        self,
        centre: tuple[int, int],
        parent_class: TileLocations,
        size: list[int],
        max_size: list[int],
    ):
        """Initialise using a parent class."""
        self.centre = centre
        self.parent_class = parent_class  # used to access drifts
        self.size = size
        self.half_size = [x // 2 for x in size]
        self.max_size = max_size

    def centre_at_time(self, tp: int) -> t.List[int]:
        """
        Return tile's centre by applying drifts.

        Parameters
        ----------
        tp: integer
            Index for the time point of interest.
        """
        drifts = self.parent_class.drifts
        tile_centre = self.centre - np.sum(drifts[: tp + 1], axis=0)
        return list(tile_centre.astype(int))

    def as_tile(self, tp: int):
        """
        Return tile in the OMERO tile format of x, y, w, h.

        Here x, y are at the bottom left corner of the tile
        and w and h are the tile width and height.

        Parameters
        ----------
        tp: integer
            Index for the time point of interest.

        Returns
        -------
        x: int
            x-coordinate of bottom left corner of tile.
        y: int
            y-coordinate of bottom left corner of tile.
        w: int
            Width of tile.
        h: int
            Height of tile.
        """
        x, y = self.centre_at_time(tp)
        # tile bottom corner
        x = int(x - self.half_size[0])
        y = int(y - self.half_size[1])
        return (x, y, *self.size)

    def as_range(self, tp: int):
        """
        Return a horizontal and a vertical slice of a tile.

        Parameters
        ----------
        tp: integer
            Index for a time point

        Returns
        -------
        A slice of x coordinates from left to right
        A slice of y coordinates from top to bottom
        """
        x, y, w, h = self.as_tile(tp)
        return slice(x, x + w), slice(y, y + h)
