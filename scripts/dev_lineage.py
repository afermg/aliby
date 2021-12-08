#! Script to test the mother assignment arrays in the results h5py files.
# /usr/bin/env python3

from itertools import groupby
from pathlib import Path
import numpy as np
import h5py
from pcore.cells import CellsHDF
from utils_find_1st import find_1st, cmp_equal

# fpath = Path(
#     "/home/alan/Documents/sync_docs/data/data/2021_03_20_sugarShift_pal_glu_pal_Myo1Whi5_Sfp1Nhp6a__00/2021_03_20_sugarShift_pal_glu_pal_Myo1Whi5_Sfp1Nhp6a__00/myo1_whi5_002.h5"
# )

dpath = Path(
    "/home/alan/Documents/sync_docs/data/data/2021_03_20_sugarShift_pal_glu_pal_Myo1Whi5_Sfp1Nhp6a__00/2021_03_20_sugarShift_pal_glu_pal_Myo1Whi5_Sfp1Nhp6a__00"
)


def compare_ma_methods(fpath):

    with h5py.File(fpath, "r") as f:
        ma = f["cell_info/mother_assign"][()]
        mad = f["cell_info/mother_assign_dynamic"][()]
        trap = f["cell_info/trap"][()]
        timepoint = f["cell_info/timepoint"][()]
        cell_label = f["cell_info/cell_label"][()]

    cells = Cells().from_source(fpath)

    def mother_assign_from_dynamic(ma, cell_label, trap, ntraps: int):
        """
        Interpolate the list of lists containing the associated mothers from the mother_assign_dynamic feature
        """
        idlist = list(zip(trap, label))
        cell_gid = np.unique(idlist, axis=0)

        last_lin_preds = [
            find_1st(((label[::-1] == lbl) & (trap[::-1] == tr)), True, cmp_equal)
            for tr, lbl in cell_gid
        ]
        mother_assign_sorted = ma[::-1][last_lin_preds]

        traps = cell_gid[:, 0]
        iterator = groupby(zip(traps, mother_assign_sorted), lambda x: x[0])
        d = {key: [x[1] for x in group] for key, group in iterator}
        nested_massign = [d.get(i, []) for i in range(ntraps)]

        return nested_massign

    mad_fixed = mother_assign_from_dynamic(mad, cell_label, trap, len(np.unique(trap)))

    dyn = sum([np.array(i, dtype=bool).sum() for i in mad_fixed])
    nondyn = sum([x.astype(bool).sum() for x in ma])
    return dyn, nondyn


for fpath in dpath.glob("*.h5"):
    print(compare_ma_methods(fpath))
