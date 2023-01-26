"""
Neural network initialisation.
"""
from pathlib import Path
from time import perf_counter

import numpy as np
import tensorflow as tf
from agora.io.writer import DynamicWriter


def initialise_tf(version):
    # Initialise tensorflow
    if version == 1:
        core_config = tf.ConfigProto()
        core_config.gpu_options.allow_growth = True
        session = tf.Session(config=core_config)
        return session
    # TODO this only works for TF2
    if version == 2:
        gpus = tf.config.experimental.list_physical_devices("GPU")
        if gpus:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            logical_gpus = tf.config.experimental.list_logical_devices("GPU")
            print(
                len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs"
            )
        return None


def timer(func, *args, **kwargs):
    start = perf_counter()
    result = func(*args, **kwargs)
    print(f"Function {func.__name__}: {perf_counter() - start}s")
    return result


################## CUSTOM OBJECTS ##################################
class ModelPredictor:
    """Generic object that takes a NN and returns the prediction.

    Use for predicting fluorescence/other from bright field.
    This does not do instance segmentations of anything.
    """

    def __init__(self, tiler, model, name):
        self.tiler = tiler
        self.model = model
        self.name = name

    def get_data(self, tp):
        # Change axes to X,Y,Z rather than Z,Y,X
        return (
            self.tiler.get_tp_data(tp, self.bf_channel)
            .swapaxes(1, 3)
            .swapaxes(1, 2)
        )

    def format_result(self, result, tp):
        return {self.name: result, "timepoints": [tp] * len(result)}

    def run_tp(self, tp):
        """Simulating processing time with sleep"""
        # Access the image
        segmentation = self.model.predict(self.get_data(tp))
        return self._format_result(segmentation, tp)


class ModelPredictorWriter(DynamicWriter):
    def __init__(self, file, name, shape, dtype):
        super.__init__(file)
        self.datatypes = {
            name: (shape, dtype),
            "timepoint": ((None,), np.uint16),
        }
        self.group = f"{self.name}_info"
