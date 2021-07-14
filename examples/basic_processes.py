from postprocessor.core.processor import PostProcessor, PostProcessorParameters

params = PostProcessorParameters.default()
pp = PostProcessor(
    "/shared_libs/pipeline-core/scripts/data/ph_calibration_dual_phl_ura8_5_04_5_83_7_69_7_13_6_59__01/ph_5_04_005store.h5",
    params,
)
tmp = pp.run()
