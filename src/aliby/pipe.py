#!/usr/bin/env jupyter

"""
New and simpler pipeline that uses dictionaries as parameters and to define variable and between-step method execution.
"""

from copy import copy
from itertools import cycle
from pathlib import Path

import pandas as pd
import polars as pl

from aliby.io.image import dispatch_image
from aliby.segment.dispatch import dispatch_segmenter
from aliby.tile.tiler import Tiler, TilerParameters
from aliby.track.dispatch import dispatch_tracker
from extraction.core.extractor import Extractor, ExtractorParameters


def init_step(
    step_name: str,
    parameters: dict[str, str or callable or int or dict],
    other_steps: callable,
) -> callable:
    """
    Set up the parameters for any step. This mostly includes dispatching a specific step subtype.
    """
    match step_name:
        case "tile":
            image_kwargs = parameters["image_kwargs"]
            tiler_kwargs = {k: v for k, v in parameters.items() if k != "image_kwargs"}
            image_type = dispatch_image(source=image_kwargs["source"])
            image = image_type(**image_kwargs)

            step = Tiler.from_image(image, TilerParameters(**tiler_kwargs))
        case "segment":
            if (
                parameters["segmenter_kwargs"]["kind"] == "baby"
            ):  # Baby needs a tiler inside
                parameters["segmenter_kwargs"]["tiler"] = other_steps["tile"]
            step = dispatch_segmenter(**{
                **parameters["segmenter_kwargs"],
            })
        case "track":
            if (
                parameters["kind"] == "baby"
            ):  # Tracker needs to pull info from baby crawler
                parameters["crawler"] = other_steps["segment"].crawler
            step = dispatch_tracker(**parameters)
        case "extract":
            tiler = other_steps["tile"]
            step = Extractor(ExtractorParameters(parameters["tree"]), tiler=tiler)
        case _:
            raise ("Invalid step name")

    return step


def run_step(step, *args, **kwargs):
    if hasattr(step, "run_tp"):  # in case of older OO-style
        result = step.run_tp(*args, **kwargs)
    else:  # Functional version, all relevant kwargs are provided but no more
        if "tp" in kwargs:
            del kwargs["tp"]
        result = step(*args, **kwargs)

    return result


def pipeline_step(
    pipeline: dict,
    state: dict = {},
) -> dict:
    """ """
    steps = pipeline["steps"]
    passed_data = pipeline["passed_data"]
    passed_methods = pipeline["passed_methods"]
    tp = list(state.get("tps", {None: 0}).values())[0]
    if not tp:  # Initialise steps
        state = {"tps": dict(zip(steps, cycle([0]))), "data": {}, "fn": {}}

    for step_name, parameters in steps.items():
        # Get or initialise step

        if step_name not in state["data"]:
            state["data"][step_name] = []
        step = state["fn"].get(step_name, init_step(step_name, parameters, state["fn"]))

        # Pass input data if available
        this_step_receives = pipeline["passed_data"].get(step_name, {})

        passed_data = {}
        for kwd, from_step in this_step_receives:
            passed_data[kwd] = state["data"].get(from_step, [])

            if not (
                step_name == "track" and kwd == "masks"
            ):  # Only tracking segmentation masks require multiple time points
                if len(passed_data[kwd]):
                    passed_data[kwd] = passed_data[kwd][-1]

        # Run step
        args = []
        if (
            step_name == "segment" and parameters["segmenter_kwargs"]["kind"] != "baby"
        ):  # Pass correct images from tiler
            source_step, method, param_name = passed_methods.get("segment")
            args = getattr(state["fn"][source_step], method)(tp, parameters[param_name])

        step_result = run_step(step, *args, tp=tp, **passed_data)

        state["data"][step_name].append(
            step_result  # TODO replace this with a variable to adjust ntps in memory
        )

        # Update state
        if not (step_name in state["fn"]):
            state["fn"][step_name] = step  # the steps should not require update
        state["tps"][step_name] = tp + 1

    return state


# TODO pass sorted images instead of wildcard
def run_pipeline(pipeline: dict, img_source: str or list[str], ntps: int):
    pipeline = copy(pipeline)
    pipeline["steps"]["tile"]["image_kwargs"]["source"] = img_source
    data = []
    state = {}

    for i in range(ntps):
        state = pipeline_step(pipeline, state)
        new_data = format_extraction(state["data"]["extract"][-1])
        data.append(new_data)

    extracted_fov = pl.concat(data)
    return extracted_fov


def format_extraction(extracted_tp: dict[str, pd.DataFrame]) -> pl.DataFrame:
    if not len(list(extracted_tp.values())[0]):
        return pl.DataFrame()

    # If DataFrame contains multiple items (e.g., CellProfiler measurements)
    # Split them into multiple dict entries
    to_delete = []
    new_entries = {}
    for k, df in extracted_tp.items():
        if isinstance(df.iloc[:, 0].values[0], dict):
            to_delete.append(k)
            exploded = pd.json_normalize(df.iloc[:, 0])

            for col in exploded.columns:
                new_entry = exploded.loc(axis=1)[[col]]
                new_entry.columns = df.columns
                new_entry.index = df.index

                new_name = f"{k}_{col}"

                new_entries[new_name] = new_entry

    for k in to_delete:
        del extracted_tp[k]

    extracted_tp = {**extracted_tp, **new_entries}

    renamed_columns = []
    for k, v in extracted_tp.items():
        if (tp := str(v.columns[0])) is not None:
            renamed_columns.append(
                pl.DataFrame(v.reset_index())
                .with_columns(
                    pl.col(tp).cast(pl.Float32).alias("value"),
                    pl.lit(k).alias("Feature"),
                    pl.lit(tp).alias("tp"),
                    pl.col("trap").cast(pl.UInt16),
                )
                .select(pl.exclude(tp))
            )
    concat = pl.concat(renamed_columns)
    return concat


def get_well_fov(wildcard: str) -> tuple[str, str]:
    """
    Extract the well and field-of-view from a wildcard-like string.
    """
    split_fname = wildcard.split("/")[-1].split("_")
    well = split_fname[2]
    fov = split_fname[-1][3:6]
    return (well, fov)


def label_and_concat_extraction(
    results: list[pl.DataFrame], wildcards: list[str]
) -> pl.DataFrame:
    datasets = []
    for wc, df in zip(wildcards, results):
        well, fov = get_well_fov(wc)
        datasets.append(
            df.with_columns(
                pl.lit(well).alias("well"),
                pl.lit(fov).alias("fov"),
            )
        )

    extracted_dataset = pl.concat(datasets)
    # TODO check if we want to remove the trap column
    return extracted_dataset.select()


def run_pipeline_save(out_file: Path, overwrite: bool = False, **kwargs) -> None:
    """
    Runs a pipeline and saves the result to a parquet file.

    Parameters
    ----------
    base_pipeline : dict
        The base pipeline configuration.
    img_source : str or list[str]
        Input files for the pipeline. It can be a list of files
    or an expression with a wildcard.
    out_file : str or Path
        Output file path for the result.
    ntps : int, optional
        Number of threads to use (default is 1).

    Returns
    -------
    result
        The result of running the pipeline.
    """
    print(f"Running {out_file}\n")
    result = None
    if overwrite or not Path(out_file).exists():
        result = run_pipeline(**kwargs)
        out_dir = Path(out_file).parent
        if not out_dir.exists():  # Only create a dir after we have files to save
            out_dir.mkdir(parents=True, exist_ok=True)
        result.write_parquet(out_file)
    return result
