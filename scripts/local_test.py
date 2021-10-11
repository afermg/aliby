#!/usr/bin/env python3


from core.segment import Tiler

from core.io.writer import Writer, load_attributes

local_file = "/shared_libs/pydask/pipeline-core/data/2021_06_27_downUpShift_2_0_2_glu_dual_phl_ura8__00/phl_ura8_001.h5"

tiler = Tiler.from_hdf5(None, local_file)
