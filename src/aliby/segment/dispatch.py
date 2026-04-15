#!/usr/bin/env jupyter
"""

See all the available models at
https://cellpose.readthedocs.io/en/latest/models.html#full-built-in-models
"""

from functools import partial

import numpy as np
from skimage.segmentation import relabel_sequential


def dispatch_segmenter(
    kind: str, channel_to_segment: int, address: str = None, **kwargs
) -> callable:
    match kind:
        case "nahual_baby":
            # TODO update this with setup_process_dispatch
            from nahual.client.baby import load_model, process_data

            # Have a sensible set of defaults
            extra_args = {
                "refine_outlines": ("", "true"),
                "with_edgemasks": ("", "true"),
                "with_masks": ("", "true"),
            }

            modelset = kwargs.pop("modelset")
            assert modelset is not None, f"Missing modelset on {kind} segmentation"
            session_id = load_model(address, modelset)

            if "extra_args" in kwargs:
                for k, v in kwargs["extra_args"]:
                    extra_args[k] = v

            _process = partial(
                process_data,
                address=address,
                session_id=session_id,
                channel_to_segment=channel_to_segment,
                extra_args=tuple(extra_args.items()),
                return_metadata=True,
            )

            def segment(pixels):
                # Baby returns list of (n_layers, Y, X) per tile.
                # Collapse layers into a single 2D label mask (Y, X) per tile
                # using max-projection (safe: DSatur ensures no pixel overlap
                # across layers, so max gives the unique cell label at each pixel).
                # Return as list so process_tree_masks sees a per-tile structure.
                tile_shape = pixels.shape[-2:]  # (Y, X) from input
                per_tile = _process(pixels)
                return [
                    nyx.max(axis=0)
                    if nyx.shape[0] > 0
                    else np.zeros(tile_shape, dtype=int)
                    for nyx in per_tile
                ]

            return segment
        case "nahual_cellpose":
            # Examples over at https://github.com/afermg/nahual/blob/master/examples/
            # Cellpose via a nahual running server
            from nahual.process import dispatch_setup_process

            assert address is not None, "You must provide an address if using Nahual."

            tool = kind.removeprefix("nahual_")

            setup, process = dispatch_setup_process(tool)

            setup_params = kwargs.get("setup_params", {})
            # eval_params = kwargs.get("eval_params", {})

            info = setup(setup_params, address=address)
            print(f"Cellpose via nahual set up. Remote returned {info}")

            return partial(process, address=address)
        case "cellpose":  # One of the cellpose models
            # cellpose does without all the ABC stuff
            # It returns a function to segment
            from cellpose.models import CellposeModel

            # Meta parameters
            setup_params = kwargs.get("setup_params", {})
            gpu = setup_params.pop("gpu", True)
            device = setup_params.pop("device", None)

            # use custom models if fullpath is provided
            pretrained = {}

            model = CellposeModel(
                **pretrained,
                gpu=gpu,
                device=device,
            )

            # ensure it returns only masks
            # TODO generalise so it does not assume a 1-tile file
            def segment(
                pixels: np.ndarray,
                do_3D: bool = False,
                stitch_threshold: int | None = None,
                **kwargs,
            ) -> list[np.ndarray]:
                """Preprocess the input numpy array to feed into Cellpose.
                Assumes FCZYX pixels shape."""
                z_size = pixels.shape[2]

                if pixels.ndim > 5:  # If time dim was passed remove it
                    pixels = pixels[0]  # FCZYX

                pixels = pixels[:, channel_to_segment]  # FZYX
                z_axis = None
                # Cellpose gets annoying if we keep the z-dimension with one stack
                if do_3D and z_size > 1:
                    z_axis = 1
                    stitch_threshold = 0.01
                    normalize = dict(norm3D=False)
                else:
                    z_axis = None
                    stitch_threshold = 0.0
                    normalize = True
                    if z_size > 1:  # FYX
                        pixels = pixels.max(axis=1)
                    else:
                        pixels = pixels[:, 0]

                result = model.eval(
                    pixels,
                    do_3D=do_3D,
                    stitch_threshold=stitch_threshold,
                    normalize=normalize,
                    z_axis=z_axis,
                    **kwargs,
                )
                labels = result[0]
                ndim = labels.ndim
                if ndim == 3:  # Cellpose squeezes dims!
                    # TODO Check that this is the best way to project 3-D labels into 2D
                    labels = labels.max(axis=0)

                    # Cover case where the reduction on z removes an entire item
                    labels = relabel_sequential(labels)[0]

                elif not 1 < ndim < 4:
                    raise Exception(
                        f"Segmentation yielded {result.ndim} dimensions instead of 3"
                    )

                return labels

        case _:
            raise Exception(f"Invalid segmentation method {kind}")

    return segment
