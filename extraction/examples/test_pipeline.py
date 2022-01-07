import numpy as np
from pathlib import Path
from extraction.core.extractor import Extractor
from extraction.core.parameters import Parameters
from extraction.core.functions.defaults import get_params

params = Parameters(**get_params("batman_ph_dual_fast"))
# ext = Extractor(params, source=19918)  # 19831
ext = Extractor(params, source=19831)
ext.load_tiler()
self = ext
# s=self.extract_exp(tree={'general':{None:['area']}, 'GFPFast':{np.maximum:['median','mean']}},poses=self.expt.positions[:2], tps=[0,1], stg='df')
s = self.extract_exp()
# # import cProfile
# # profile = cProfile.Profile()
# # profile.enable()
# # ext.change_position(ext.expt.positions[1])
# # tracks = self.extract_pos(
# #     tree={('general'):{None:  # Other metrics can be used
# #                        [tidy_metric]}})#['general',None,'area']

# # profile.disable()
# # import pstats
# # ps = pstats.Stats(profile)
# # ps.sort_stats('cumulative')
# # ps.print_stats()
