#!/usr/bin/env jupyter
"""

See all the available models at
https://cellpose.readthedocs.io/en/latest/models.html#full-built-in-models
"""

from agora.abc import StepABC


def dispatch_segmenter(config, **kwargs) -> callable:
    if config["model"] == "baby":
        import tensorflow as tf

        # stop warnings from TensorFlow
        os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
        logging.getLogger("tensorflow").setLevel(logging.ERROR)

        from aliby.baby_client import BabyParameters, BabyRunner

        initialise_tensorflow()
        segmenter_cls, segmenter_params = BabyRunner, BabyParameters
        segment = segmenter_cls.from_tiler(
            segmenter_params.from_dict(config["segmenter"]),
            tiler=kwargs["tiler"],
        )
    else:  # One of the cellpose models
        # cellpose does without all the ABC stuff
        # It returns a function to segment
        from cellpose.models import CellposeModel

        model_type = config.get("model", "cyto_3")
        argname = "model_type"
        # use custom models if fullpath is provided
        if model_type.startswith("/"):
            argname = "pretrained_model"
        model = CellPoseModel(**{argname: model_type})
        # ensure it returns only masks
        segment = lambda x: model.eval(x)[0]

    return segment


def initialise_tensorflow(version=2):
    """Initialise tensorflow."""
    if version == 2:
        gpus = tf.config.experimental.list_physical_devices("GPU")
        if gpus:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            logical_gpus = tf.config.experimental.list_logical_devices("GPU")
            print(
                len(gpus), "physical GPUs,", len(logical_gpus), "logical GPUs"
            )
