#!/usr/bin/env python3

import typing as t
from abc import ABC
from collections import Counter
from functools import cached_property as property
from pathlib import Path
from typing import Union
from tqdm import tqdm
import h5py
import numpy as np
import pandas as pd
from pathos.multiprocessing import Pool

from agora.io.signal import Signal


class Grouper(ABC):
    """Base grouper class."""

    def __init__(self, dir: Union[str, Path], name_inds=(0, -4)):
        """Find h5 files and load each one."""
        path = Path(dir)
        assert path.exists(), f"{str(dir)} does not exist"
        self.name = path.name
        self.files = list(path.glob("*.h5"))
        assert len(self.files), "No valid h5 files in dir"
        self.load_positions()
        self.positions_groups = {
            name: name[name_inds[0] : name_inds[1]]
            for name in self.positions.keys()
        }

    def load_positions(self) -> None:
        """Load a Signal for each position, or h5 file."""
        self.positions = {f.name[:-3]: Signal(f) for f in sorted(self.files)}

    @property
    def first_signal(self) -> Signal:
        """Get Signal for the first position."""
        return list(self.positions.values())[0]

    @property
    def ntimepoints(self) -> int:
        """Find number of time points."""
        return max([s.ntimepoints for s in self.positions.values()])

    @property
    def tinterval_minutes(self) -> float:
        """Find the time interval for all positions."""
        tintervals = list(
            np.unique([s.tinterval / 60 for s in self.positions.values()])
        )
        if len(tintervals) > 1:
            raise Exception(
                "Grouper: Not all positions have the same time interval."
            )
        return tintervals[0]

    @property
    def all_available(self) -> t.Collection[str]:
        """Generate list of available signals from all positions."""
        all_available = [
            x for s in tqdm(self.positions.values()) for x in s.available
        ]
        return sorted(set(all_available))

    @property
    def available(self) -> t.Collection[str]:
        """Generate list of available signals from all positions."""
        available = self.first_signal.available
        return available

    @property
    def available_grouped(self) -> None:
        """Display available signals and the number of positions with these signals."""
        if not hasattr(self, "_available_grouped"):
            self._available_grouped = Counter(
                [x for s in self.positions.values() for x in s.available]
            )
        for s, n in self._available_grouped.items():
            print(f"{s} - {n}")

    def concat_signal(
        self,
        path: str,
        pool: t.Optional[int] = None,
        mode: str = "retained",
        selected_positions: t.List[str] = None,
        tmax_in_mins_dict: dict = None,
        **kwargs,
    ):
        """
        Concatenate data for one signal from different h5 files.

        Each h5 files corresponds to a different position.

        Parameters
        ----------
        path : str
           Signal location within h5 file.
        pool : int (optional)
           Number of threads used; if 0 or None only one core is used.
        mode: str
           If "retained" (default), return Signal with merging, picking, and lineage
            information applied but only for cells present for at least some
            cutoff fraction of the movie.
           If "raw", return Signal without merging, picking, lineage information,
            or a cutoff applied. Each of the first three options can be
            re-selected. A raw Signal with all three selected is the same as a
            retained Signal with a 0 cutoff.
           If "daughters", return Signal with only daughters - cells with an
            identified mother.
           If "families", get Signal with merging, picking, and lineage
            information applied.
        selected_positions: list[str] (optional)
            If defined, get signals for only these positions.
        tmax_in_mins_dict: dict (optional)
            A dictionary with positions as keys and maximum times in minutes as
            values. For example: { "PDR5_GFP_001": 6 * 60}.
            Data will only be include up to this time point, which is a way to
            avoid errors in assigning lineages because of clogging.
        **kwargs : key, value pairings
           Named arguments for concat_ind_function

        Examples
        --------
        >>> record = grouper.concat_signal("extraction/GFP/max/median")
        """
        if path.startswith("/"):
            path = path.strip("/")
        good_positions = self.filter_positions(path)
        if selected_positions is not None:
            good_positions = {
                key: value
                for key, value in good_positions.items()
                if key in selected_positions
            }
        if good_positions:
            kwargs["mode"] = mode
            records = self.pool_function(
                path=path,
                f=concat_one_signal,
                pool=pool,
                positions=good_positions,
                tmax_in_mins_dict=tmax_in_mins_dict,
                **kwargs,
            )
            # check for errors
            errors = [
                position
                for record, position in zip(records, good_positions.keys())
                if record is None
            ]
            records = [record for record in records if record is not None]
            if len(errors):
                print(f"Warning: Positions ({errors}) contain errors.")
            assert len(records), "All data sets contain errors"
            # combine into one data frame
            concat = pd.concat(records, axis=0)
            if len(concat.index.names) > 4:
                # reorder levels in the multi-index data frame
                # when mother_label is present
                concat = concat.reorder_levels(
                    ("group", "position", "trap", "cell_label", "mother_label")
                )
            concat_sorted = concat.sort_index()
            return concat_sorted
        else:
            print("No data found.")

    def filter_positions(self, path: str) -> t.Dict[str, Signal]:
        """Filter positions to those whose data is available in the h5 file."""
        good_positions = {
            k: v for k, v in self.positions.items() if path in [*v.available]
        }
        no_positions_dif = len(self.positions) - len(good_positions)
        if no_positions_dif:
            print(
                f"Grouper:Warning: some positions ({no_positions_dif}) do not"
                f" contain {path}."
            )
        return good_positions

    def pool_function(
        self,
        path: str,
        f: t.Callable,
        pool: t.Optional[int] = None,
        positions: t.Dict[str, Signal] = None,
        tmax_in_mins_dict: dict = None,
        **kwargs,
    ):
        """
        Enable different threads for different positions.

        Particularly useful when aggregating multiple elements.
        """
        positions = positions or self.positions
        if pool:
            with Pool(pool) as p:
                records = p.map(
                    lambda x: f(
                        path=path,
                        position=x[1],
                        group=self.positions_groups[x[0]],
                        position_name=x[0],
                        tmax_in_mins_dict=tmax_in_mins_dict,
                        **kwargs,
                    ),
                    positions.items(),
                )
        else:
            records = [
                f(
                    path=path,
                    position=position,
                    group=self.positions_groups[name],
                    position_name=name,
                    tmax_in_mins_dict=tmax_in_mins_dict,
                    **kwargs,
                )
                for name, position in positions.items()
            ]
        return records

    @property
    def no_tiles(self):
        """Get total number of tiles per position (h5 file)."""
        for pos, s in self.positions.items():
            with h5py.File(s.filename, "r") as f:
                print(pos, f["/trap_info/trap_locations"].shape[0])

    @property
    def tile_locs(self) -> t.Dict[str, np.ndarray]:
        """Get the locations of the tiles for each position as a dictionary."""
        d = {}
        for pos, s in self.positions.items():
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
                for position in self.positions.values()
                for channel in position.channels
            ]
        )

    @property
    def no_members(self) -> t.Dict[str, int]:
        """Get the number of positions belonging to each group."""
        return Counter(self.positions_groups.values())

    @property
    def no_tiles_by_group(self) -> t.Dict[str, int]:
        """Get total number of tiles per group."""
        no_tiles = {}
        for pos, s in self.positions.items():
            with h5py.File(s.filename, "r") as f:
                no_tiles[pos] = f["/trap_info/trap_locations"].shape[0]
        no_tiles_by_group = {k: 0 for k in self.groups}
        for posname, vals in no_tiles.items():
            no_tiles_by_group[self.positions_groups[posname]] += vals
        return no_tiles_by_group

    @property
    def groups(self) -> t.Tuple[str]:
        """Get groups, sorted alphabetically, as a list."""
        return list(sorted(set(self.positions_groups.values())))


def concat_one_signal(
    path: str,
    position: Signal,
    group: str,
    mode: str = "retained",
    position_name=None,
    tmax_in_mins_dict=None,
    cutoff: float = 0,
    **kwargs,
) -> pd.DataFrame:
    """
    Retrieve a signal for one position.

    kwargs passed to signal.get_raw.
    """
    if tmax_in_mins_dict and position_name in tmax_in_mins_dict:
        tmax_in_mins = tmax_in_mins_dict[position_name]
    else:
        tmax_in_mins = None
    if position_name is None:
        # name of h5 file
        position_name = position.stem
    if tmax_in_mins:
        print(
            f" Loading {path} for {position_name} up to time {tmax_in_mins}."
        )
    else:
        print(f" Loading {path} for {position_name}.")
    if mode == "retained":
        # applies picking and merging via Signal.get
        combined = position.retained(
            path, tmax_in_mins=tmax_in_mins, cutoff=cutoff
        )
    elif mode == "raw":
        # no picking and merging
        combined = position.get_raw(path, tmax_in_mins=tmax_in_mins, **kwargs)
    elif mode == "raw_daughters":
        combined = position.get_raw(
            path, lineage=True, tmax_in_mins=tmax_in_mins, **kwargs
        )
        if combined is not None:
            combined = combined.loc[
                combined.index.get_level_values("mother_label") > 0
            ]
    elif mode == "raw_mothers":
        combined = position.get_raw(
            path, lineage=True, tmax_in_mins=tmax_in_mins, **kwargs
        )
        if combined is not None:
            combined = combined.loc[
                combined.index.get_level_values("mother_label") == 0
            ]
            combined = combined.droplevel("mother_label")
    elif mode == "families":
        # applies picking and merging
        combined = position.get(path, tmax_in_mins=tmax_in_mins)
    else:
        raise Exception(f"concat_one_signal: {mode} not recognised.")
    if combined is not None:
        # add position and group as indices
        combined["position"] = position_name
        combined["group"] = group
        combined.set_index(["group", "position"], inplace=True, append=True)
        combined.index = combined.index.swaplevel(-2, 0).swaplevel(-1, 1)
    return combined
