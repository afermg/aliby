# Example of argo experiment explorer
from agora.argo import Argo
from extraction.core.extractor import Extractor
from extraction.core.parameters import Parameters
from extraction.core.functions.defaults import get_params

argo = Argo()
argo.load()
# argo.channels("GFP")
argo.tags(["Alan"])
argo.complete()
# argo.cExperiment()
# argo.tiler_cells()

# params = Parameters(**get_params("batman_ph_dual_fast"))


# def try_extract(d):
#     try:
#         params = Parameters(**get_params("batman_ph_dual_fast"))
#         ext = Extractor(params, source=d.getId())
#         ext.load_tiler_cells()
#         ext.process_experiment()
#         print(d.getId(), d.getName(), "Experiment processed")
#         return True
#     except:
#         print(d.getId(), d.getName(), "Experiment not processed")

#         return False


# from multiprocessing.dummy import Pool as ThreadPool

# pool = ThreadPool(4)
# results = pool.map(try_extract, argo.dsets)
# import pickle

# with open("results.pkl", "wb") as f:
#     pickle.dump(results, f)
