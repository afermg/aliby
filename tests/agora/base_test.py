"""
Basic ParametersIO tests
"""


from agora.abc import ParametersABC


class DummyParameters(ParametersABC):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    @classmethod
    def default(cls):
        # Necessary empty builder
        return cls.from_dict({})


def test_file_exists(yaml_file):
    assert yaml_file.exists()


def test_from_yaml(yaml_file):
    # From yaml
    params = DummyParameters.from_yaml(yaml_file)


def test_from_stdin(yaml_file):
    # From yaml
    params = DummyParameters.from_yaml(yaml_file)
    # To yaml
    assert isinstance(params, ParametersABC)


def test_to_yaml(yaml_file):
    with open(yaml_file, "r") as fd:
        yaml_data = fd.read()

    params = DummyParameters.from_yaml(yaml_file)

    assert params.to_yaml() == yaml_data


def test_dict(example_dict):
    params = DummyParameters(**example_dict)
    assert params.to_dict() == example_dict
    # Remove
    params.to_yaml("outfile.yml")


def test_to_dict():
    DummyParameters.default().to_dict()
