#!/usr/bin/env python3
"""Extract profiles from yeast microscopy time-series using BABY segmentation.

Processes all experiments in a data directory through the aliby pipeline:
  Tiling -> BABY segmentation (via Nahual) -> Feature extraction

Requires a running BABY server:
  cd ~/projects/nahual_models/baby && LOKY_MAX_CPU_COUNT=1 nix run .
"""

import argparse
import datetime
import time
from pathlib import Path

import numpy as np
import zarr

from aliby.global_settings import detect_brightfield_channels
from aliby.io.dataset import DatasetZarr
from aliby.pipe import run_pipeline_and_post
from aliby.pipe_builder import build_pipeline_steps


# ---------------------------------------------------------------------------
# Progress logging
# ---------------------------------------------------------------------------


class ProgressLogger:
    """Write PROGRESS.org and PLAN.md to the output directory."""

    def __init__(self, output_dir: Path, plan_text: str = ""):
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.progress_file = output_dir / "PROGRESS.org"
        self.plan_file = output_dir / "PLAN.md"
        if plan_text:
            self.plan_file.write_text(plan_text)

    def _write(self, line: str):
        with open(self.progress_file, "a") as f:
            f.write(line + "\n")

    def log_start(self, config: dict):
        ts = datetime.datetime.now().isoformat(timespec="seconds")
        self._write("#+TITLE: Extraction Progress")
        self._write(f"#+DATE: {ts}")
        self._write("")
        self._write("* Configuration")
        for k, v in config.items():
            self._write(f"  - {k}: {v}")
        self._write("")
        self._write("* Experiments")

    def log_experiment_start(self, experiment: str, n_positions: int):
        ts = datetime.datetime.now().isoformat(timespec="seconds")
        self._write(f"** TODO {experiment} [{ts}]")
        self._write("   :PROPERTIES:")
        self._write(f"   :POSITIONS: {n_positions}")
        self._write("   :END:")

    def log_position(self, key: str, status: str, elapsed: float = 0, error: str = ""):
        ts = datetime.datetime.now().isoformat(timespec="seconds")
        tag = "DONE" if status == "done" else "FAIL"
        line = f"*** {tag} {key} ({elapsed:.1f}s) [{ts}]"
        self._write(line)
        if error:
            self._write(f"    {error}")

    def log_experiment_done(self, experiment: str):
        ts = datetime.datetime.now().isoformat(timespec="seconds")
        self._write(f"   Completed: {ts}")


# ---------------------------------------------------------------------------
# Pipeline construction
# ---------------------------------------------------------------------------


def get_channel_info(
    zarr_path: Path, position_key: str
) -> tuple[list[int], list[int], int]:
    """Load first timepoint and detect BF vs fluorescence channels.

    Returns (bf_channels, fluo_channels, n_timepoints).
    """
    if hasattr(zarr.storage, "LocalStore"):
        store = zarr.storage.LocalStore(zarr_path)
    else:
        store = zarr.storage.DirectoryStore(zarr_path)
    root = zarr.group(store)
    arr = root[position_key]
    # Shape is (T, C, Z, Y, X)
    n_tps = arr.shape[0]
    first_tp = np.array(arr[0])  # (C, Z, Y, X)
    bf, fluo = detect_brightfield_channels(first_tp)
    return bf, fluo, n_tps


def build_baby_pipeline(
    zarr_path: Path,
    position_key: str,
    bf_channel: int,
    fluo_channels: list[int],
    baby_address: str,
    baby_modelset: str,
    ntps: int,
    tile_size: int = 117,
) -> dict:
    """Build a complete pipeline dict for BABY segmentation."""
    pipeline = build_pipeline_steps(
        channels_to_segment={"cells": bf_channel},
        channels_to_extract=fluo_channels,
        segmenter_kind="nahual_baby",
        baby_address=baby_address,
        baby_modelset=baby_modelset,
    )

    # Configure tile step
    pipeline["steps"]["tile"].update(
        tile_size=tile_size,
        image_kwargs={
            "source": {"path": str(zarr_path), "key": position_key},
            "capture_order": "TCZYX",
        },
    )

    pipeline["ntps"] = ntps

    return pipeline


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def find_experiments(data_dir: Path) -> list[Path]:
    """Find experiment wrapper zarrs (named <experiment>.zarr inside each experiment dir)."""
    experiments = []
    for exp_dir in sorted(data_dir.iterdir()):
        if not exp_dir.is_dir() or exp_dir.name.startswith("."):
            continue
        wrapper = exp_dir / f"{exp_dir.name}.zarr"
        if wrapper.exists():
            experiments.append(wrapper)
    return experiments


def process_experiment(
    wrapper_zarr: Path,
    output_dir: Path,
    baby_address: str,
    baby_modelset: str,
    max_positions: int | None,
    max_tps: int | None,
    tile_size: int,
    logger: ProgressLogger,
):
    """Process all positions in a single experiment."""
    exp_name = wrapper_zarr.parent.name
    exp_output = output_dir / exp_name

    ds = DatasetZarr(wrapper_zarr)
    positions = ds.get_position_ids()
    if max_positions:
        positions = positions[:max_positions]

    logger.log_experiment_start(exp_name, len(positions))
    print(f"\n{'=' * 60}")
    print(f"Experiment: {exp_name} ({len(positions)} positions)")
    print(f"{'=' * 60}")

    for pos in positions:
        key = pos["key"]
        t0 = time.time()
        try:
            bf, fluo, n_tps = get_channel_info(wrapper_zarr, key)
            if not bf:
                print(f"  [{key}] WARNING: No brightfield channel detected, using ch 0")
                bf = [0]
                fluo = [ch for ch in range(3) if ch != 0][:2]
            if not fluo:
                print(
                    f"  [{key}] WARNING: No fluorescence channels, extracting from all non-BF"
                )
                # Still need at least one channel for extraction
                fluo = [ch for ch in range(3) if ch not in bf]

            ntps = min(n_tps, max_tps) if max_tps else n_tps
            print(f"  [{key}] BF={bf}, fluo={fluo}, tps={ntps}")

            pipeline = build_baby_pipeline(
                zarr_path=wrapper_zarr,
                position_key=key,
                bf_channel=bf[0],
                fluo_channels=fluo,
                baby_address=baby_address,
                baby_modelset=baby_modelset,
                ntps=ntps,
                tile_size=tile_size,
            )

            profiles, post = run_pipeline_and_post(
                pipeline=pipeline,
                pipeline_name=key,
                output_path=exp_output,
            )

            elapsed = time.time() - t0
            n_rows = profiles.num_rows if profiles is not None else 0
            print(f"  [{key}] Done ({elapsed:.1f}s, {n_rows} profile rows)")
            logger.log_position(key, "done", elapsed)

        except Exception as e:
            elapsed = time.time() - t0
            print(f"  [{key}] FAILED ({elapsed:.1f}s): {e}")
            logger.log_position(key, "fail", elapsed, str(e))

    logger.log_experiment_done(exp_name)


def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("/home/amunoz/projects/microscopy_backup/data"),
        help="Root directory containing experiment subdirectories",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("/datastore/alan/aliby_baby_profiles"),
        help="Root output directory for results",
    )
    parser.add_argument(
        "--baby-address",
        default="http://0.0.0.0:5101",
        help="BABY Nahual server address",
    )
    parser.add_argument(
        "--baby-modelset",
        default="yeast-alcatras-brightfield-sCMOS-60x-5z",
        help="BABY model set identifier",
    )
    parser.add_argument(
        "--tile-size",
        type=int,
        default=117,
        help="Tile size in pixels",
    )
    parser.add_argument(
        "--max-positions",
        type=int,
        default=None,
        help="Limit number of positions per experiment (for testing)",
    )
    parser.add_argument(
        "--max-tps",
        type=int,
        default=None,
        help="Limit number of timepoints per position (for testing)",
    )
    parser.add_argument(
        "--experiment",
        type=str,
        default=None,
        help="Process only this experiment (substring match on directory name)",
    )
    args = parser.parse_args()

    # Find experiments
    experiments = find_experiments(args.data_dir)
    if args.experiment:
        experiments = [e for e in experiments if args.experiment in e.parent.name]

    if not experiments:
        print(f"No experiments found in {args.data_dir}")
        return

    print(f"Found {len(experiments)} experiment(s)")

    # Read plan.org if available
    plan_text = ""
    plan_org = Path(__file__).parent.parent / "plan.org"
    if plan_org.exists():
        plan_text = plan_org.read_text()

    logger = ProgressLogger(args.output_dir, plan_text=plan_text)
    logger.log_start(
        {
            "data_dir": str(args.data_dir),
            "output_dir": str(args.output_dir),
            "baby_address": args.baby_address,
            "baby_modelset": args.baby_modelset,
            "tile_size": args.tile_size,
            "max_positions": args.max_positions,
            "max_tps": args.max_tps,
            "n_experiments": len(experiments),
        }
    )

    for wrapper_zarr in experiments:
        process_experiment(
            wrapper_zarr=wrapper_zarr,
            output_dir=args.output_dir,
            baby_address=args.baby_address,
            baby_modelset=args.baby_modelset,
            max_positions=args.max_positions,
            max_tps=args.max_tps,
            tile_size=args.tile_size,
            logger=logger,
        )

    print(f"\nAll done. Results in {args.output_dir}")


if __name__ == "__main__":
    main()
