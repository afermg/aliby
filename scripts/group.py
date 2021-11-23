#!/usr/bin/env python3
from pathlib import Path

from pcore.grouper import NameGrouper

path = Path(
    "/home/alan/Documents/dev/stoa_libs/pipeline-core/data/2021_11_04_doseResponse_raf_1_15_2_glu_01_2_dual_phluorin_whi5_constantMedia_00/2021_11_04_doseResponse_raf_1_15_2_glu_01_2_dual_phluorin_whi5_constantMedia_00"
)
ng = NameGrouper(path)
# agg = ng.aggregate_multisignals(["extraction/general/None/area"], pool=0)
agg = ng.concat_signal("extraction/general/None/area")
