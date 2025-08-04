#!/usr/bin/env jupyter
"""

See all the available models at
https://cellpose.readthedocs.io/en/latest/models.html#full-built-in-models
"""

import numpy as np
from skimage.segmentation import relabel_sequential


def dispatch_segmenter(kind: str, **kwargs) -> callable:
    match kind:
        case "baby":
            import itertools
            import logging
            import os

            from scipy.ndimage import binary_fill_holes

            from aliby.baby_client import BabyParameters, BabyRunner

            # stop warnings from TensorFlow
            os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
            logging.getLogger("tensorflow").setLevel(logging.ERROR)

            initialise_tensorflow()
            segmenter_cls, segmenter_params = BabyRunner, BabyParameters

            assert "tiler" in kwargs, "A Tiler must be passed to baby seg"
            tiler = kwargs["tiler"]  # Assume tiler is passed
            baby_kwargs = {k: v for k, v in kwargs.items() if k != "tiler"}
            segment = segmenter_cls.from_tiler(
                segmenter_params.default(**baby_kwargs),
                tiler=tiler,
            )

            # Monkey patch segmentation class so this returns masks only
            # https://github.com/julianpietsch/baby/issues/5

            def wrap_segmentation(tp, *args, **kwargs):
                segmentation = segment._run_tp(tp, *args, **kwargs)
                # refill edgemasks
                masks = [binary_fill_holes(x) for x in segmentation["edgemasks"]]
                # group by tile to match extractor input
                masks_by_tile = {
                    x[0]: np.stack([y[1] for y in x[1]])
                    for x in itertools.groupby(
                        zip(segmentation["trap"], masks), lambda x: x[0]
                    )
                }
                return [
                    masks_by_tile.get(i, np.array([]))
                    for i in range(len(segment.tiler.tile_locs))
                ]

            segment.run_tp = wrap_segmentation

            return segment

        case _:  # One of the cellpose models
            # cellpose does without all the ABC stuff
            # It returns a function to segment
            from cellpose.models import CellposeModel

            # Meta parameters
            model = kind
            gpu = kwargs.pop("gpu", False)
            device = kwargs.pop("device", None)

            argname = "model_type"
            # use custom models if fullpath is provided
            if model.startswith("/"):
                argname = "pretrained_model"
            model = CellposeModel(
                **{
                    argname: model,
                },
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

                return labels

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
