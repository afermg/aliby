filepath = (
    "/shared_libs/pydask/pipeline-core/data/2021_05_27_PDR5Fluc_00/YST_1490_001.h5"
    # "/shared_libs/pydask/pipeline-core/data/2021_08_21_KCl_pH_00/YST_247_013.h5"
)
import h5py

import gnuplotlib as gp

with h5py.File(filepath, "r") as f:
    tlocs = f["trap_info/trap_locations"][()]


gp.plot(
    *[i for i in tlocs.T],
    title="Trap map {}".format(filepath.split["/"][-1]),
    unset="grid",
    cmds="set view map",
    terminal="dumb 100, 50",
    _with="points"
)
