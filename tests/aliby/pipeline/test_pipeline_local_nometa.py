#!/usr/bin/env jupyter

from pathlib import Path

import pytest

from aliby.pipeline import Pipeline, PipelineParameters


def test_local_pipeline(file: str):
    if Path(file).exists():
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
    else:
        print("Test dataset not downloaded")
