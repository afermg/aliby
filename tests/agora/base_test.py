"""
Basic ParametersIO tests
"""

import pytest

from agora.abc import ParametersABC


class DummyParameters(ParametersABC):
    # TODO add default data folder and load for all tests
    yaml_file = "tests/agora/data/parameters.yaml"

    def __init__(self):
        super().__init__()

    def test_dict(self):
        param_dict = dict(a="a", b="b", c=dict(d="d", e="e"))
        params = self.from_dict(param_dict)
        assert params.to_dict() == param_dict
        # Remove
        params.to_yaml(self.yaml_file)

    def test_yaml(self):
        # From yaml
        params = self.from_yaml(self.yaml_file)
        # To yaml
        with open(self.yaml_file, "r") as fd:
            yaml_data = fd.read()
        assert params.to_yaml() == yaml_data

    @classmethod
    def default(cls):
        return cls.from_dict({})


def test_to_yaml():
    DummyParameters.default().to_yaml()


def test_from_yaml():
    DummyParameters.default().test_yaml()


def test_to_dict():
    DummyParameters.default().to_dict()
