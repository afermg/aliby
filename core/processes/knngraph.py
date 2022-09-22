#!/usr/bin/env python3

import igraph as ig
import numpy as np
import pandas as pd
from agora.abc import ParametersABC
from sklearn.metrics.pairwise import euclidean_distances

from postprocessor.core.abc import PostProcessABC


class knngraphParameters(ParametersABC):
    """
    Parameters for the 'knngraph' process.
    """

    _defaults = {
        "n_neighbors": 10,
    }


class knngraph(PostProcessABC):
    """
    Process to get geometric graph from data/features, based on k-nearest
    neighbours.

    Methods
    -------
    run(signal: pd.DataFrame)
        Get graph.
    """

    def __init__(self, parameters: knngraphParameters):
        super().__init__(parameters)

    def run(self, signal: pd.DataFrame):
        """Get graph.

        Parameters
        ----------
        signal : pd.DataFrame
            Feature matrix.
        """
        distance_matrix = euclidean_distances(signal)
        distance_matrix_pruned = graph_prune(
            distance_matrix, self.n_neighbours
        )
        graph = ig.Graph.Weighted_Adjacency(
            distance_matrix_pruned.tolist(), mode="undirected"
        )
        graph.vs["strain"] = signal.index.get_level_values("strain")
        return graph


def graph_prune(distanceMatrix, neighbours):
    """
    Prunes a complete graph (input distance matrix), keeping at least a
    specified number of neighbours for each node.

    Parameters:
    -----------
    distanceMatrix = 2D numpy array
    neighbours = integer

    Return: Dij_pruned, a 2D numpy array, represents distance matrix of pruned
            graph
    """
    Dij_temp = distanceMatrix
    Adj = np.zeros(distanceMatrix.shape)
    for ii in range(distanceMatrix.shape[0]):
        idx = np.argsort(Dij_temp[ii, :])
        Adj[ii, idx[1]] = 1
        Adj[idx[1], ii] = 1
        for jj in range(neighbours):
            Adj[ii, idx[jj]] = 1
            Adj[idx[jj], ii] = 1
    Dij_pruned = Dij_temp * Adj
    return Dij_pruned
