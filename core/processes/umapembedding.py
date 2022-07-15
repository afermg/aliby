#!/usr/bin/env python3

import pandas as pd
import umap
from agora.abc import ParametersABC

from postprocessor.core.abc import PostProcessABC


class umapembeddingParameters(ParametersABC):
    """
    Parameters for the 'umapembedding' process.
    """

    _defaults = {
        "n_neighbors": None,
        "min_dist": None,
        "n_components": None,
    }


class umapembedding(PostProcessABC):
    """
    Process to get UMAP embeddings from data/features.

    Methods
    -------
    run(signal: pd.DataFrame)
        Get UMAP embeddings.
    """

    def __init__(self, parameters: umapembeddingParameters):
        super().__init__(parameters)

    def run(self, signal: pd.DataFrame):
        """Get UMAP embeddings.

        This function is effectively a wrapper for umap.UMAP.

        Parameters
        ----------
        signal : pd.DataFrame
            Feature matrix.
        """
        mapper = umap.UMAP(
            n_neighbors=self.n_neighbors,
            min_dist=self.min_dist,
            n_components=self.n_components,
        ).fit(signal)
        return mapper.embedding_
