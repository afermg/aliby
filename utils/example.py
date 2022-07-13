"""This is an example module to show the structure."""
from typing import Union


class ExampleClass:
    """This is an example class to show the structure."""

    def __init__(self, parameter: int):
        """This class takes one parameter and is used to add one to that
        parameter.

        :param parameter: The parameter for this class
        """
        self.parameter = parameter

    def add_one(self):
        """Takes the parameter and adds one.

        >>> x = ExampleClass(1)
        >>> x.add_one()
        2

        :return: the parameter + 1
        """
        return self.parameter + 1

    def add_n(self, n: int):
        """Adds n to the class instance's parameter.

        For instance
        >>> x = ExampleClass(1)
        >>> x.add_n(10)
        11

        :param n: The number to add
        :return: the parameter + n
        """
        return self.parameter + n


def example_function(parameter: Union[int, str]):
    """This is a factory function for an ExampleClass.

    :param parameter: the parameter to give to the example class
    :return: An example class
    """
    try:
        return ExampleClass(int(parameter))
    except ValueError as e:
        raise ValueError(
            f"The parameter {parameter} could not be turned "
            f"into an integer."
        ) from e
