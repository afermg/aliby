from pathos.multiprocessing import Pool

from aliby.pipeline import PipelineParameters, Pipeline


class MultiExp:
    """
    Manages cases when you need to segment several different experiments with a single
    position (e.g. pH calibration).
    """

    def __init__(self, expt_ids, npools=8, *args, **kwargs):

        self.expt_ids = expt_ids

    def run(self):
        run_expt = lambda expt: Pipeline(
            PipelineParameters.default(general={"expt_id": expt, "distributed": 0})
        ).run()
        with Pool(npools) as p:
            results = p.map(lambda x: self.create_pipeline(x), self.exp_ids)

    @classmethod
    def default(self):
        return cls(expt_ids=list(range(20448, 20467 + 1)))
