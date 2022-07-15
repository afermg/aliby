from itertools import product

import igraph as ig
import leidenalg
import numpy as np
import pandas as pd
from agora.abc import ParametersABC

from postprocessor.core.abc import PostProcessABC


class leidenParameters(ParametersABC):
    """
    Parameters
    """

    _defaults = {}


class leiden(PostProcessABC):
    """
    leiden algorithm applied to a dataframe with features.
    """

    def __init__(self, parameters: leidenParameters):
        super().__init__(parameters)

    def run(self, features: pd.DataFrame):
        # Generate euclidean distance matrix
        distances = np.linalg.norm(
            features.values - features.values[:, None], axis=2
        )
        ind = [
            "_".join([str(y) for y in x[1:]])
            for x in features.index.to_flat_index()
        ]
        source, target = zip(*product(ind, ind))
        df = pd.DataFrame(
            {
                "source": source,
                "target": target,
                "distance": distances.flatten(),
            }
        )
        df = df.loc[df["source"] != df["target"]]
        g = ig.Graph.DataFrame(df, directed=False)

        return leidenalg.find_partition(g, leidenalg.ModularityVertexPartition)
