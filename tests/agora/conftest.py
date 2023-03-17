#!/usr/bin/env jupyter
"""
Load data necessary to test agora.
"""
import typing as t
from pathlib import Path

import pytest


@pytest.fixture(scope="module")
def data_dir():
    return Path(__file__).parent / "data"


@pytest.fixture(scope="module")
def yaml_file(data_dir: Path):
    data = data_dir / "parameters.yaml"
    if not data.exists():
        pytest.fail(f"There is no file at {str( data_dir )}.")
    return data


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
