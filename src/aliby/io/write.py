from pathlib import Path

import numpy as np
import pyarrow
from pyarrow.parquet import write_table


def dispatch_write_fn(
    step_name: str,
):
    match step_name:
        case s if s.startswith("segment"):
            return write_ndarray

        case s if s.startswith("tile"):
            return write_ndarray

        case s if s.startswith("nahual_trackastra"):
            return write_parquet

        case _:
            raise Exception(f"Writing {step_name} is not supported yet")


def write_ndarray(result, steps_dir: Path, subpath: str or int, tp: int) -> None:
    """Write a numpy array into an npy file.

    Creates parent directories if needed."""
    this_step_path = Path(steps_dir) / subpath
    this_step_path.mkdir(exist_ok=True, parents=True)
    if subpath == "tile":
        subpath = "pixels"

    out_file = this_step_path / f"{tp:04d}.npz"
    np.savez(out_file, np.array(result))


def write_parquet(
    result: pyarrow.Table, out_dir: Path, subpath: str, filename: str
) -> None:
    """
    Write a PyArrow table into a parquet file. Create the directory if it does not exist.

    Parameters
    ----------
    output_dir: Root directory of all outputs for a given experiment.
    subpath: Sublocaton of multiple parquets (e.g., name of objects being tracked).
    filename: Name of the parquet file (wihout suffix), usually it is the well identifier (e.g., A01).

    Returns
    -------
    None
    """
    this_outdir = out_dir / subpath
    this_outdir.mkdir(exist_ok=True, parents=True)
    out_filepath = this_outdir / f"{filename}.parquet"

    write_table(result, out_filepath)
