from postprocessor.core.processor import PostProcessor, PostProcessorParameters

params = PostProcessorParameters.default()
pp = PostProcessor(
    "/shared_libs/pipeline-core/scripts/pH_calibration_dual_phl__ura8__by4741__01/ph_5_29_025store.h5",
    params,
)
tmp = pp.run()

import h5py

# f = h5py.File(
#     "/shared_libs/pipeline-core/scripts/pH_calibration_dual_phl__ura8__by4741__01/ph_5_29_025store.h5",
#     "a",
# )
