#!/usr/bin/env jupyter
"""

See all the available models at
https://cellpose.readthedocs.io/en/latest/models.html#full-built-in-models
"""

import numpy as np


def dispatch_segmenter(kind, **kwargs) -> callable:
    match kind:
        case "baby":
            import logging
            import os

            import tensorflow as tf
            from aliby.baby_client import BabyParameters, BabyRunner

            # stop warnings from TensorFlow
            os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
            logging.getLogger("tensorflow").setLevel(logging.ERROR)

            initialise_tensorflow()
            segmenter_cls, segmenter_params = BabyRunner, BabyParameters
            # TODO fix this
            segment = segmenter_cls.from_tiler(
                segmenter_params.from_dict(config["segmenter"]),
                tiler=kwargs["tiler"],
            )
        case _:  # One of the cellpose models
            # cellpose does without all the ABC stuff
            # It returns a function to segment
            from cellpose.models import CellposeModel

            model = kind
            argname = "model_type"
            # use custom models if fullpath is provided
            if model.startswith("/"):
                argname = "pretrained_model"
            model = CellposeModel(**{argname: model})

            # ensure it returns only masks
            # TODO generalise so it does not assume a 1-tile file
            def segment(*args) -> list[np.ndarray]:
                return [model.eval(*args, **kwargs)[0]]

            return segment

    return segment


def initialise_tensorflow(version=2):
    """Initialise tensorflow."""
    if version == 2:
        gpus = tf.config.experimental.list_physical_devices("GPU")
        if gpus:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            logical_gpus = tf.config.experimental.list_logical_devices("GPU")
            print(len(gpus), "physical GPUs,", len(logical_gpus), "logical GPUs")
