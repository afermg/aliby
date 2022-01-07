import numpy as np
import pandas as pd
from agora.abc import ParametersABC, ProcessABC


class bud_metricParameters(ParametersABC):
    """
    Parameters
    """

    def __init__(self, mode="longest"):
        super().__init__()
        self.mode = mode

    @classmethod
    def default(cls):
        return cls.from_dict({"mode": "longest"})


class bud_metric(ProcessABC):
    """
    Obtain the volume of daughter cells
    if 'longest' assumes a single mother per trap.
    """

    def __init__(self, parameters: bud_metricParameters):
        super().__init__(parameters)

    def run(self, signal: pd.DataFrame):
        if self.parameters.mode is "longest":
            result = self.get_bud_metric_wrap(signal)

        return result

    @staticmethod
    def get_bud_metric(signal):
        mother_id = signal.index[signal.notna().sum(axis=1).argmax()]

        nomother = signal.drop(mother_id)

        if not len(nomother):
            bud_metric = [np.nan for i in range(len(signal.columns))]

        else:
            starts = nomother.apply(pd.Series.first_valid_index, axis=1).sort_values()

            ranges = [np.arange(i, j) for i, j in zip(starts[:-1], starts[1:])]
            ranges.append(np.arange(starts.iloc[-1], signal.columns[-1]))

            bud_metric = pd.concat(
                [signal.loc[i, rng] for i, rng in zip(starts.index, ranges)]
            )
        srs = pd.Series(bud_metric, index=signal.columns, name=mother_id)

        return srs

    def get_bud_metric_wrap(self, signals):
        srs = [
            self.get_bud_metric(signals.loc[trap])
            for trap in signals.index.unique(level="trap")
        ]
        index = [
            (trap, mother.name)
            for trap, mother in zip(signals.index.unique(level="trap"), srs)
        ]

        concatenated = pd.concat(srs, keys=index, axis=1, sort=True).T.sort_index()
        concatenated.index.names = ["trap", "cell_label"]
        return concatenated
