"""This is an example test file to show the structure."""
import pytest

from agora.utils.example import ExampleClass, example_function


class TestExampleClass:
    x = ExampleClass(1)

    def test_add_one(self):
        assert self.x.add_one() == 2

    def test_add_n(self):
        assert self.x.add_n(10) == 11


def test_example_function():
    x = example_function(1)
    assert isinstance(x, ExampleClass)
    assert x.parameter == 1


def test_example_function_fail():
    with pytest.raises(ValueError):
        example_function("hello")
