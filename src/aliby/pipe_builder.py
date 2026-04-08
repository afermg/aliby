#!/usr/bin/env python

from itertools import combinations, product


def _create_extract_multich_tree(
    channels: list[int], extract_ncores: int | None
) -> dict:
    """Generate the extract_multich_tree dictionary for colocalization."""
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
    channels_to_segment: dict[str, int] = {"nuclei": 1, "cell": 0},
    channels_to_extract: list[int] | None = None,
    extract_ncores: int | None = None,
    nahual_addresses: str | list[str] | None = None,
    devices: list[int] | None = None,
    steps_to_write: list[str] = [],
) -> dict:
    """Convenience function to build a pipeline definition, does not fill in IO."""

    use_nahual = nahual_addresses is not None
    distribute_across_devices = devices is not None

    segmenter_kind = "cellpose"
    if use_nahual:
        segmenter_kind = "nahual_cellpose"

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
        if distribute_across_devices:
            seg_params[step_name]["segmenter_kwargs"]["address"] = addresses[
                hashed_input % n_addresses
            ]
            seg_params[step_name]["segmenter_kwargs"]["setup_params"] = (
                hashed_input % n_devices
            )

    extract_base = dict(
        tree={None: {None: "sizeshape"}},
        kwargs=dict(ncores=extract_ncores),
    )
    for i in channels_to_extract:
        extract_base["tree"][i] = {
            "max": [
                "radial_zernikes",
                "intensity",
                "feret",
                "texture",
                "radial_distribution",
                "zernike",
                # "granularity", # Too time-consuming, deactivated for now
            ]
        }

    # Add sizeshape with no channels TODO adjust extraction code
    extract_base["tree"][None] = {None: "sizeshape"}

    extract_multich_base = _create_extract_multich_tree(
        channels_to_extract, extract_ncores
    )

    # Build extraction parameters using segmentation
    ext_params = {
        f"extract{name}_{obj}": var
        for (name, var), obj in product(
            (("", extract_base), ("multich", extract_multich_base)),
            channels_to_extract,
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
                f"extract_{obj}": [
                    ("masks", f"segment_{obj}"),
                    ("pixels", "tile"),
                ]
                for obj in channels_to_segment
            },
        ),
        "passed_methods": {
            f"segment_{obj}": ("tile", "get_fczyx") for obj in channels_to_segment
        },
        "write": [],
        "write_interval": 1,
    }

    if steps_to_write is not None:
        base_pipeline["write"] = steps_to_write

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
