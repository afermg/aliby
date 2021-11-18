#!/usr/bin/env python3

from pcore.pipeline import PipelineParameters, Pipeline

params = PipelineParameters.default(general={"expt_id": 19995, "distributed": 5})
p = Pipeline(params)
p.run()
