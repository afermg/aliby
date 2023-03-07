#!/usr/bin/env jupyter
import argparse

from agora.utils.cast import _str_to_int
from aliby.pipeline import Pipeline, PipelineParameters


def run():
    """
    Run a default microscopy analysis pipeline.

    Parse command-line arguments and set default parameter values for running a pipeline, then
    construct and execute the pipeline with the parameters obtained. Command-line arguments can
    override default parameter values. If a command-line argument is a string representation of
    an integer, convert it to an integer.

    Returns
    -------
    None

    Examples
    --------
    FIXME: Add docs.
    """
    parser = argparse.ArgumentParser(
        prog="aliby-run",
        description="Run a default microscopy analysis pipeline",
    )

    param_values = {
        "expt_id": None,
        "distributed": 2,
        "tps": 2,
        "directory": "./data",
        "filter": 0,
        "host": None,
        "username": None,
        "password": None,
    }

    for k in param_values:
        parser.add_argument(f"--{k}", action="store")

    args = parser.parse_args()

    for k in param_values:
        if passed_value := _str_to_int(getattr(args, k)):

            param_values[k] = passed_value

    params = PipelineParameters.default(general=param_values)
    p = Pipeline(params)

    p.run()
