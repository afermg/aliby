#!/usr/bin/env python3
import re
import typing as t

import numpy as np
import pandas as pd

from agora.io.bridge import groupsort
from itertools import groupby


def mb_array_to_dict(mb_array: np.ndarray):
    """
    Convert a lineage ndarray (trap, mother_id, daughter_id)
    into a dictionary of lists ( mother_id ->[daughters_ids] )
    """
    return {
        (trap, mo): [(trap, d[0]) for d in daughters]
        for trap, mo_da in groupsort(mb_array).items()
        for mo, daughters in groupsort(mo_da).items()
    }


def mb_array_to_indices(mb_array: np.ndarray):
    """
    Convert a lineage ndarray (trap, mother_id, daughter_id)
    into a dictionary of lists ( mother_id ->[daughters_ids] )
    """
    return pd.MultiIndex.from_arrays(mb_array[:, :2].T).union(
        pd.MultiIndex.from_arrays(mb_array[:, [0, 2]].T)
    )


def group_matrix(
    matrix: np.ndarray,
    n_keys: int = 2,
) -> t.Dict[t.Tuple[int], t.List[int]]:
    """Group a matrix of integers by grouping the first two columns
    and setting the third one in a list.


    Parameters
    ----------
    matrix : np.ndarray
        id_matrix, generally its columns are three integers indicating trap,
        mother and daughter.
    n_keys : int
        number of keys to use to determine groups.

    Returns
    -------
    t.Dict[t.Tuple[int], t.Collection[int, ...]]
        The column(s) not used for generaeting keys are grouped as values.

    Examples
    --------
    FIXME: Add docs.

    """
    lineage_dict = {}
    if len(matrix):

        daughter = matrix[:, n_keys]
        mother_global_id = matrix[:, :n_keys]

        iterator = groupby(
            zip(mother_global_id, daughter), lambda x: str(x[0])
        )
        lineage_dict = {key: [x[1] for x in group] for key, group in iterator}

        def str_to_tuple(k: str) -> t.Tuple[int, ...]:
            return tuple([int(x) for x in re.findall("[0-9]+", k)])

        # Convert keys from str to tuple
        lineage_dict = {
            str_to_tuple(k): sorted(v) for k, v in lineage_dict.items()
        }

    return lineage_dict
