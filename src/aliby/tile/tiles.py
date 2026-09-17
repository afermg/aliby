"""Define classes for a tile and all tiles."""

import typing as t

import h5py
import numpy as np


class Tile:
    """Define a tile."""

    def __init__(self, centre, parent_class, size, max_size):
        """Initialise using a parent class."""
        self.centre = centre
        self.parent_class = parent_class  # used to access drifts
        self.size = size
        self.half_size = size // 2
        self.max_size = max_size

    def centre_at_time(self, tp: int) -> t.List[int]:
        """
        Return tile's centre by applying drifts.

        The centre is (y, x) - row first - because trap locations come from
        skimage's peak_local_max and regionprops.centroid and the drifts from
        phase_cross_correlation, all of which are (row, column). The result is
        truncated to integers, so a tile's origin is always an integer.

        Parameters
        ----------
        tp: integer
            Index for the time point of interest.

        Returns
        -------
        A list of the y- and x-coordinates of the tile's centre.
        """
        drifts = self.parent_class.drifts
        tile_centre = self.centre - np.sum(drifts[: tp + 1], axis=0)
        return list(tile_centre.astype(int))

    def as_tile(self, tp: int):
        """
        Return tile as y, x, h, w.

        Here y, x are the tile's top left corner - its [0, 0] pixel - and h
        and w are the tile's height and width.

        Parameters
        ----------
        tp: integer
            Index for the time point of interest.

        Returns
        -------
        y: int
            y-coordinate, the row, of the tile's top left corner.
        x: int
            x-coordinate, the column, of the tile's top left corner.
        h: int
            Height of tile.
        w: int
            Width of tile.
        """
        y, x = self.centre_at_time(tp)
        # tile top left corner
        y = int(y - self.half_size)
        x = int(x - self.half_size)
        return y, x, self.size, self.size

    def as_range(self, tp: int):
        """
        Return a vertical and a horizontal slice of a tile.

        Rows first: the first slice indexes the y-axis of an image, as in
        image_array[:, y, x].

        Parameters
        ----------
        tp: integer
            Index for a time point

        Returns
        -------
        A slice of y coordinates from top to bottom
        A slice of x coordinates from left to right
        """
        y, x, h, w = self.as_tile(tp)
        return slice(y, y + h), slice(x, x + w)


class TileLocations:
    """Store each tile as an instance of Tile."""

    def __init__(
        self,
        initial_location: np.array,
        tile_size: int = None,
        max_size: int = 1200,
        drifts: np.array = None,
    ):
        """
        Initialise tiles as an array of Tile objects.

        Parameters
        ----------
        initial_location: array
            An array of tile centres.
        tile_size: int
            Length of one side of a square tile.
        max_size: int, optional
            Default is 1200.
        drifts: array
            An array of translations to correct drift of the microscope.
        """
        if drifts is None:
            drifts = []
        self.tile_size = tile_size
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

    def to_dict(self, tp: int, first_tp: int = 0):
        """
        Export initial locations, tile_size, max_size, and drifts as a dict.

        At the first time point processed, export the drifts of every time
        point up to it, so that drifts are indexed by time point.

        Parameters
        ----------
        tp: integer
            An index for a time point
        first_tp: integer
            The first time point processed.
        """
        res = dict()
        if tp == first_tp:
            res["trap_locations"] = self.initial_location
            res["attrs/tile_size"] = self.tile_size
            res["attrs/max_size"] = self.max_size
            res["drifts"] = np.asarray(self.drifts[: tp + 1])
        else:
            res["drifts"] = np.expand_dims(self.drifts[tp], axis=0)
        return res

    def centres_at_time(self, tp: int) -> np.ndarray:
        """Return an array of tile centres (y- and x-coords, rows first)."""
        return np.array([tile.centre_at_time(tp) for tile in self.tiles])

    @classmethod
    def from_tiler(
        cls,
        initial_location,
        tile_size: int = None,
        max_size: int = 1200,
    ):
        """Instantiate from a Tiler."""
        return cls(initial_location, tile_size, max_size, drifts=[])

    @classmethod
    def from_h5(cls, file):
        """Instantiate from a h5 file."""
        with h5py.File(file, "r") as hfile:
            tile_info = hfile["trap_info"]
            initial_locations = tile_info["trap_locations"][()]
            drifts = tile_info["drifts"][()].tolist()
            max_size = tile_info.attrs["max_size"]
            tile_size = tile_info.attrs["tile_size"]
        tile_loc_cls = cls(initial_locations, tile_size, max_size=max_size)
        tile_loc_cls.drifts = drifts
        return tile_loc_cls


def tile_in_image(
    slices: tuple[slice, slice], shape_yx: tuple[int, int]
) -> tuple[tuple[slice, slice], np.ndarray]:
    """
    Find the part of a tile inside an image and the padding the tile lacks.

    A drift-corrected tile can extend past the image's edge. This is the one
    rule for where it is cut, shared by the tiler, which cuts tiles from
    whole planes, and by OMERO's tile loader, which asks the server for only
    the part inside the image.

    Parameters
    ----------
    slices: tuple of two slices
        The tile's y- and x-ranges, rows first.
    shape_yx: tuple of two ints
        The image's height and width.

    Returns
    -------
    inside: tuple of two slices
        The tile's y- and x-ranges clipped to the image.
    padding: array
        The rows and columns the clipped tile lacks, as
        ``[[top, bottom], [left, right]]``.
    """
    inside = tuple(
        slice(max(0, s.start), min(size, s.stop))
        for s, size in zip(slices, shape_yx)
    )
    padding = np.array(
        [
            (-min(0, s.start), -min(0, size - s.stop))
            for s, size in zip(slices, shape_yx)
        ]
    )
    return inside, padding


def too_far_outside(padding: np.ndarray, tile_size: int) -> bool:
    """
    Return whether too much of a tile lies outside the image to pad it.

    Such a tile is filled with NaN rather than padded with edge values.

    Parameters
    ----------
    padding: array
        The padding the tile lacks, as ``tile_in_image`` returns it.
    tile_size: int
        Length of one side of the square tile.
    """
    return bool((padding > tile_size / 4).any())
