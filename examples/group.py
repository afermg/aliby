from pathlib import Path

from postprocessor.core.group import GroupParameters, Group

poses = [
    x.name.split("store")[0]
    for x in Path(
        "/shared_libs/pipeline-core/scripts/data/ph_calibration_dual_phl_ura8_5_04_5_83_7_69_7_13_6_59__01"
    ).rglob("*")
    if x.name != "images.h5"
]

gr = Group(
    GroupParameters(
        signals=["/extraction/general/None/area", "/extraction/mCherry/np_max/median"]
    )
)
gr.run(
    central_store="/shared_libs/pipeline-core/scripts/data/ph_calibration_dual_phl_ura8_5_04_5_83_7_69_7_13_6_59__01",
    poses=poses,
)
