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
            }

            modelset = kwargs.pop("modelset")
            assert modelset is not None, f"Missing modelset on {kind} segmentation"
            session_id = load_model(address, modelset)

            if "extra_args" in kwargs:
                for k, v in kwargs["extra_args"]:
                    extra_args[k] = v

            return partial(
                process_data,
                address=address,
                session_id=session_id,
                channel_to_segment=channel_to_segment,
                extra_args=tuple(extra_args.items()),
            )
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
            model_type = kind
            setup_params = kwargs.get("setup_params", {})
            gpu = setup_params.pop("gpu", True)
            device = setup_params.pop("device", None)

            # use custom models if fullpath is provided
            pretrained = {}
            if model_type.startswith("/"):
                pretrained["pretrained_model"] = model_type

            model = CellposeModel(
                **pretrained,
                gpu=gpu,
                device=device,
            )

            # ensure it returns only masks
            # TODO generalise so it does not assume a 1-tile file
            def segment(*args) -> list[np.ndarray]:
                result = model.eval(
                    *args,
                    z_axis=0,
                    normalize=dict(norm3D=False),
                    stitch_threshold=0.1,
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

                return labels[np.newaxis]  # Add "tile" dimension

        case _:
            raise Exception(f"Invalid segmentation method {kind}")

    return segment


def initialise_tensorflow(version=2):
    """Initialise tensorflow."""
    import tensorflow as tf

    if version == 2:
        gpus = tf.config.experimental.list_physical_devices("GPU")
        if gpus:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            logical_gpus = tf.config.experimental.list_logical_devices("GPU")
            print(len(gpus), "physical GPUs,", len(logical_gpus), "logical GPUs")
