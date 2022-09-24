#!/usr/bin/env jupyter
"""
Load data necessary to test agora.
"""
import typing as t
from pathlib import Path

import pytest


@pytest.fixture(scope="module")
def yaml_file():
    return Path(__file__).parent / "data/parameters.yaml"


@pytest.fixture(scope="module", autouse=True)
def parameters_example() -> t.Dict:
    return dict(
        string="abc",
        number=1,
        boolean=True,
        dictionary=dict(
            empty_dict=dict(),
            string="def",
            number=2,
        ),
    )
