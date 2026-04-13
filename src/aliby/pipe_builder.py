#!/usr/bin/env python

from itertools import combinations, product
from typing import Sequence


def _create_extract_multich_tree(
    channels: Sequence[int], extract_ncores: int | None
) -> dict:
    """
    Generate the extract_multich_tree dictionary for colocalization.

    Parameters
    ----------
    channels : Sequence of int
        Sequence of channel indices to consider for colocalization.
    extract_ncores : int or None
        Number of cores to use for extraction.

    Returns
    -------
    dict
        Dictionary containing the multich tree.
    """
    return {
        "tree": {
            pair: {
                "None": {
                    "max": ["pearson", "costes", "manders_fold", "rwc"],
                },
            }
            for pair in combinations(channels, r=2)
        },
        "kwargs": {
            "ncores": extract_ncores,
        },
    }


def build_pipeline_steps(
    channels_to_segment: dict[str, int] | None = None,
    channels_to_extract: Sequence[int] | None = None,
    features_to_extract: Sequence[str] = (
        "radial_zernikes",
        "intensity",
        "feret",
        "texture",
        "radial_distribution",
        "zernike",
        # "granularity", # Too time-consuming, deactivated for now
    ),
    extract_ncores: int | None = None,
    nahual_addresses: str | Sequence[str] | None = None,
    devices: Sequence[int] | None = None,
    steps_to_write: Sequence[str] | None = None,
    segmenter_kind: str | None = None,
    baby_modelset: str | None = None,
    baby_address: str | None = None,
) -> dict:
    """
    Convenience function to build a pipeline definition, does not fill in IO.

    Parameters
    ----------
    channels_to_segment : dict of {str: int} or None, default=None
        Dictionary mapping object names to their respective channel indices.
        If None, defaults to `{"nuclei": 1, "cell": 0}`.
    channels_to_extract : Sequence of int or None, default=None
        Sequence of channel indices from which to extract features. If None, it
        uses all values from `channels_to_segment`.
    features_to_extract : Sequence of str, default=("radial_zernikes", ...)
        Sequence of feature types to extract for each extracted channel.
    extract_ncores : int or None, default=None
        Number of CPU cores to use for feature extraction.
    nahual_addresses : str or Sequence of str or None, default=None
        Address(es) of the Nahual servers for remote segmentation.
    devices : Sequence of int or None, default=None
        Sequence of device IDs to distribute Nahual segmentation across.
    steps_to_write : Sequence of str or None, default=None
        Sequence of steps whose outputs should be saved (e.g., written to disk).
        If None, defaults to all keys in `channels_to_segment`.

    Returns
    -------
    dict
        A dictionary defining the pipeline configuration. Expected keys include:

        - ``steps`` (dict): Dictionary mapping step names to their parameter dictionaries.
          Used to instantiate pipeline steps via `init_step`.
        - ``passed_data`` (dict): Specifies which outputs from previous steps should be
          passed as arguments to a given step. Format is
          `{step_name: [(parameter_key, from_step, *optional_varname), ...]}`.
        - ``passed_methods`` (dict): Specifies methods to call on previous step objects to
          retrieve data. Format is `{step_name: (source_step, method_name)}`.
        - ``save`` (list of str): List of step names whose outputs should be saved to disk.
        - ``save_interval`` (int): Interval (in time points) at which to save outputs.
    """

    if channels_to_segment is None:
        channels_to_segment = {"nuclei": 1, "cell": 0}

    use_nahual = nahual_addresses is not None
    distribute_across_devices = devices is not None

    if segmenter_kind is None:
        segmenter_kind = "cellpose"
        if use_nahual:
            segmenter_kind = "nahual_cellpose"

    use_baby = segmenter_kind == "nahual_baby"

    if distribute_across_devices:  # Randomly assign Nahual_Cellpose instances
        assert isinstance(nahual_addresses, list), (
            "`nahual_addresses must` be specified distributing across devices"
        )
        n_devices = len(devices)
        n_addresses = len(addresses)

        hashed_input = hash(str(input_path))
        device_id = hashed_input % n_devices

    if channels_to_extract is None:
        channels_to_extract = list(channels_to_segment.values())

    # Build parameter names
    seg_params = {}
    for i, (obj, ch_id) in enumerate(channels_to_segment.items()):
        step_name = f"segment_{obj}"
        seg_params[step_name] = dict(
            segmenter_kwargs=dict(
                kind=segmenter_kind,
            ),
            channel_to_segment=ch_id,
        )
        if use_baby:
            assert baby_address is not None, "baby_address required for nahual_baby"
            assert baby_modelset is not None, "baby_modelset required for nahual_baby"
            seg_params[step_name]["segmenter_kwargs"]["address"] = baby_address
            seg_params[step_name]["segmenter_kwargs"]["modelset"] = baby_modelset
        if distribute_across_devices:
            seg_params[step_name]["segmenter_kwargs"]["address"] = addresses[
                hashed_input % n_addresses
            ]
            seg_params[step_name]["segmenter_kwargs"]["setup_params"] = (
                hashed_input % n_devices
            )

    extract_base = dict(
        tree={"None": {"None": ("sizeshape",)}},
        kwargs=dict(ncores=extract_ncores),
    )
    if use_baby:
        extract_base["overlap"] = True
    for i in channels_to_extract:
        extract_base["tree"][i] = {
            "max": features_to_extract,
        }
    extract_multich_base = _create_extract_multich_tree(
        channels_to_extract, extract_ncores
    )

    # Build extraction parameters using segmentation
    # Multi-channel extraction doesn't support baby's overlapping masks
    extract_variants = [("", extract_base)]
    if not use_baby:
        extract_variants.append(("multi", extract_multich_base))
    ext_params = {
        f"extract{name}_{obj}": var
        for (name, var), obj in product(
            extract_variants,
            channels_to_segment,
        )
        if len(var)
    }

    base_pipeline = {
        "steps": dict(
            tile=dict(
                # These should be provided outside
                # image_kwargs=dict(
                #     source=input_path,
                #     # regex=regex,
                #     # capture_order=fluo_base_config["capture_order"],
                #     # dimorder=fluo_base_config["dimorder"],
                # ),
                tile_size=None,  # default value
                # ref_channel=0,
                # ref_z=0,
                # calculate_drift=False,
            ),
            **seg_params,
            **ext_params,
        ),
        "passed_data": dict(
            **{
                f"extract{multi}_{obj}": [
                    ("masks", f"segment_{obj}"),
                    ("pixels", "tile"),
                ]
                for obj in channels_to_segment
                for multi in ([n for n, _ in extract_variants])
            },
        ),
        "passed_methods": {
            f"segment_{obj}": ("tile", "get_fczyx") for obj in channels_to_segment
        },
        "save": [f"segment_{obj}" for obj in channels_to_segment.keys()],
        "save_interval": 1,
    }

    if steps_to_write is not None:
        base_pipeline["save"] = list(steps_to_write)

    return base_pipeline

    # try:
    # fov = input_path["key"]  # TODO Homogeneize how to name experiments
    # return fov, base_pipeline

    # result, _ = run_pipeline_and_post(
    #     pipeline=base_pipeline,
    #     img_source=input_path,
    #     output_path=output_path,
    #     fov=fov,
    #     overwrite=False,
    # )
    # except Exception as e:
    #     print(f"Error: {e}")
