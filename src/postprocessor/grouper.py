#!/usr/bin/env python3

import typing as t
from abc import ABC
from collections import Counter
from functools import cached_property as property
from pathlib import Path
from typing import Union

import h5py
import numpy as np
import pandas as pd
from pathos.multiprocessing import Pool

from agora.io.signal import Signal
from postprocessor.chainer import Chainer


class Grouper(ABC):
    """Base grouper class."""

    def __init__(self, dir: Union[str, Path], name_inds=(0, -4)):
        """Find h5 files and load a chain for each one."""
        path = Path(dir)
        assert path.exists(), f"{str(dir)} does not exist"
        self.name = path.name
        self.files = list(path.glob("*.h5"))
        assert len(self.files), "No valid h5 files in dir"
        self.load_positions()
        self.positions_groups = {
            name: name[name_inds[0] : name_inds[1]]
            for name in self.chainers.keys()
        }

    def load_positions(self) -> None:
        """Load a chain for each position, or h5 file."""
        self.chainers = {f.name[:-3]: Signal(f) for f in self.files}

    @property
    def first_signal(self) -> Signal:
        """Get Signal for the first position."""
        return list(self.chainers.values())[0]

    @property
    def ntimepoints(self) -> int:
        """Find number of time points."""
        return max([s.ntimepoints for s in self.chainers.values()])

    @property
    def max_tinterval(self) -> float:
        """Find the maximum time interval for all chains."""
        tintervals = set([s.tinterval / 60 for s in self.chainers.values()])
        assert (
            len(tintervals) == 1
        ), "Not all chains have the same time interval"
        return max(tintervals)

    @property
    def available(self) -> t.Collection[str]:
        """Generate list of available signals from the first position."""
        return self.first_signal.available

    @property
    def available_grouped(self) -> None:
        """Display available signals and the number of positions with these signals."""
        if not hasattr(self, "available_grouped"):
            self._available_grouped = Counter(
                [x for s in self.chainers.values() for x in s.available]
            )
        for s, n in self._available_grouped.items():
            print(f"{s} - {n}")

    def concat_signal(
        self,
        path: str,
        pool: t.Optional[int] = None,
        mode: str = "retained",
        **kwargs,
    ):
        """
        Concatenate data for one signal from different h5 files into a data frame.

        Each h5 files corresponds to a different position.

        Parameters
        ----------
        path : str
           Signal location within h5 file.
        pool : int
           Number of threads used; if 0 or None only one core is used.
        mode: str
        standard: boolean
        **kwargs : key, value pairings
           Named arguments for concat_ind_function

        Examples
        --------
        >>> record = grouper.concat_signal("extraction/GFP/max/median")
        """
        if path.startswith("/"):
            path = path.strip("/")
        good_positions = self.filter_positions(path)
        if good_positions:
            fn_pos = concat_one_signal
            kwargs["mode"] = mode
            records = self.pool_function(
                path=path,
                f=fn_pos,
                pool=pool,
                chainers=good_positions,
                **kwargs,
            )
            # check for errors
            errors = [
                k
                for kymo, k in zip(records, self.chainers.keys())
                if kymo is None
            ]
            records = [record for record in records if record is not None]
            if len(errors):
                print("Warning: Positions contain errors {errors}")
            assert len(records), "All data sets contain errors"
            # combine into one dataframe
            concat = pd.concat(records, axis=0)
            if len(concat.index.names) > 4:
                # reorder levels in the multi-index dataframe when mother_label is present
                concat = concat.reorder_levels(
                    ("group", "position", "trap", "cell_label", "mother_label")
                )
            concat_sorted = concat.sort_index()
            return concat_sorted

    def filter_positions(self, path: str) -> t.Dict[str, Chainer]:
        """Filter chains to those whose data is available in the h5 file."""
        good_positions = {
            k: v for k, v in self.chainers.items() if path in [*v.available]
        }
        no_positions_dif = len(self.chainers) - len(good_positions)
        if no_positions_dif:
            print(
                f"Grouper: Warning: {no_positions_dif} positions do not contain"
                f" {path}."
            )
        return good_positions

    def pool_function(
        self,
        path: str,
        f: t.Callable,
        pool: t.Optional[int] = None,
        chainers: t.Dict[str, Chainer] = None,
        **kwargs,
    ):
        """
        Enable different threads for independent chains.

        Particularly useful when aggregating multiple elements.
        """
        chainers = chainers or self.chainers
        if pool:
            with Pool(pool) as p:
                records = p.map(
                    lambda x: f(
                        path=path,
                        chainer=x[1],
                        group=self.positions_groups[x[0]],
                        position=x[0],
                        **kwargs,
                    ),
                    chainers.items(),
                )
        else:
            records = [
                f(
                    path=path,
                    chainer=chainer,
                    group=self.positions_groups[name],
                    position=name,
                    **kwargs,
                )
                for name, chainer in self.chainers.items()
            ]
        return records

    @property
    def no_tiles(self):
        """Get total number of tiles per position (h5 file)."""
        for pos, s in self.chainers.items():
            with h5py.File(s.filename, "r") as f:
                print(pos, f["/trap_info/trap_locations"].shape[0])

    @property
    def tilelocs(self) -> t.Dict[str, np.ndarray]:
        """Get the locations of the tiles for each position as a dictionary."""
        d = {}
        for pos, s in self.chainers.items():
            with h5py.File(s.filename, "r") as f:
                d[pos] = f["/trap_info/trap_locations"][()]
        return d

    def no_cells(
        self,
        path="extraction/general/None/area",
        mode="retained",
        **kwargs,
    ) -> t.Dict[str, int]:
        """Get number of cells retained per position in base channel as a dictionary."""
        return (
            self.concat_signal(path=path, mode=mode, **kwargs)
            .groupby("group")
            .apply(len)
            .to_dict()
        )

    @property
    def no_retained(self) -> t.Dict[str, int]:
        """Get number of cells retained per position in base channel as a dictionary."""
        return self.no_cells()

    @property
    def channels(self):
        """Get channels available over all positions as a set."""
        return set(
            [
                channel
                for position in self.chainers.values()
                for channel in position.channels
            ]
        )

    @property
    def tinterval(self):
        """Get interval between time points in seconds."""
        return self.first_signal.tinterval

    @property
    def no_members(self) -> t.Dict[str, int]:
        """Get the number of positions belonging to each group."""
        return Counter(self.positions_groups.values())

    @property
    def no_tiles_by_group(self) -> t.Dict[str, int]:
        """Get total number of tiles per group."""
        no_tiles = {}
        for pos, s in self.chainers.items():
            with h5py.File(s.filename, "r") as f:
                no_tiles[pos] = f["/trap_info/trap_locations"].shape[0]
        no_tiles_by_group = {k: 0 for k in self.groups}
        for posname, vals in no_tiles.items():
            no_tiles_by_group[self.positions_groups[posname]] += vals
        return no_tiles_by_group

    @property
    def groups(self) -> t.Tuple[str]:
        """Get groups, sorted alphabetically, as a tuple."""
        return tuple(sorted(set(self.positions_groups.values())))

    @property
    def positions(self) -> t.Tuple[str]:
        """Get positions, sorted alphabetically, as a tuple."""
        return tuple(sorted(set(self.positions_groups.keys())))


class NameGrouper(Grouper):
    """Group a set of positions with a shorter version of the group's name."""

    def __init__(self, dir, name_inds=(0, -4)):
        """Define the indices to slice names."""
        super().__init__(dir=dir)
        self.name_inds = name_inds


def concat_one_signal(
    path: str,
    chainer: Chainer,
    group: str,
    mode: str = "retained",
    position=None,
    **kwargs,
) -> pd.DataFrame:
    """
    Retrieve an individual signal.

    Applies filtering if requested and adjusts indices.
    """
    if position is None:
        # name of h5 file
        position = chainer.stem
    if mode == "retained":
        combined = chainer.retained(path, **kwargs)
    elif mode == "raw":
        combined = chainer.get_raw(path, **kwargs)
    elif mode == "daughters":
        combined = chainer.get_raw(path, **kwargs)
        combined = combined.loc[
            combined.index.get_level_values("mother_label") > 0
        ]
    elif mode == "families":
        combined = chainer[path]
    else:
        raise Exception(f"{mode} not recognised.")
    if combined is not None:
        # adjust indices
        combined["position"] = position
        combined["group"] = group
        combined.set_index(["group", "position"], inplace=True, append=True)
        combined.index = combined.index.swaplevel(-2, 0).swaplevel(-1, 1)
    # should there be an error message if None is returned?
    return combined
