#!/usr/bin/env jupyter
"""
Load data necessary to test agora.
"""
import typing as t
from pathlib import Path, PosixPath

import pytest


@pytest.fixture(scope="module")
def data_dir():
    return Path(__file__).parent / "data"


@pytest.fixture(scope="module")
def yaml_file(data_dir: PosixPath):
    return data_dir / "parameters.yaml"


@pytest.fixture(scope="module", autouse=True)
def example_dict() -> t.Dict:
    return dict(
        string="abc",
        number=1,
        boolean=True,
        dictionary=dict(
            # empty_dict=dict(),
            string="def",
            number=2,
        ),
    )
