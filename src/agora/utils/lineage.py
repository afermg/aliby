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

