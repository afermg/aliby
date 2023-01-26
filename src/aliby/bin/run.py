#!/usr/bin/env jupyter


def run():
    import argparse

    from aliby.pipeline import Pipeline, PipelineParameters

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

    def _cast_str(x: str or None):
        """
        Cast string as int if possible. If Nonetype return None.
        """
        if x:
            try:
                return int(x)
            except:
                return x

    for k in param_values:
        parser.add_argument(f"--{k}", action="store")

    args = parser.parse_args()

    for k in param_values:
        if passed_value := _cast_str(getattr(args, k)):

            param_values[k] = passed_value

    params = PipelineParameters.default(general=param_values)
    p = Pipeline(params)

    p.run()
