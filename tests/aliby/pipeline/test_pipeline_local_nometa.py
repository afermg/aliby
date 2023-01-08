#!/usr/bin/env jupyter

from aliby.pipeline import PipelineParameters, Pipeline


def test_local_pipeline(file: str):
    params = PipelineParameters.default(
        general={
            "expt_id": file,
            "distributed": 0,
            "directory": "test_output/",
            "overwrite": True,
        },
        tiler={"ref_channel": 0},
    )
    p = Pipeline(params)

    p.run()
