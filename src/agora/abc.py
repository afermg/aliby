import typing as t
from abc import ABC, abstractmethod
from collections.abc import Iterable
from copy import copy
from pathlib import Path, PosixPath
from typing import Union

from flatten_dict import flatten
from yaml import dump, safe_load

atomic = t.Union[int, float, str, bool]


class ParametersABC(ABC):
    """
    Defines parameters as attributes and allows parameters to
    be converted to either a dictionary or to yaml.

    No attribute should be called "parameters"!
    """

    def __init__(self, **kwargs):
        """
        Defines parameters as attributes
        """
        assert (
            "parameters" not in kwargs
        ), "No attribute should be named parameters"
        for k, v in kwargs.items():
            setattr(self, k, v)

    def to_dict(self, iterable="null") -> t.Dict:
        """
        Recursive function to return a nested dictionary of the
        attributes of the class instance.
        """
        if isinstance(iterable, dict):
            if any(
                [
                    True
                    for x in iterable.values()
                    if isinstance(x, Iterable) or hasattr(x, "to_dict")
                ]
            ):
                return {
                    k: v.to_dict()
                    if hasattr(v, "to_dict")
                    else self.to_dict(v)
                    for k, v in iterable.items()
                }
            else:
                return iterable
        elif iterable == "null":
            # use instance's built-in __dict__ dictionary of attributes
            return self.to_dict(self.__dict__)
        else:
            return iterable

    def to_yaml(self, path: Union[PosixPath, str] = None):
        """
        Returns a yaml stream of the attributes of the class instance.
        If path is provided, the yaml stream is saved there.

        Parameters
        ----------
        path : Union[PosixPath, str]
            Output path.
        """
        if path:
            with open(Path(path), "w") as f:
                dump(self.to_dict(), f)
        return dump(self.to_dict())

    @classmethod
    def from_dict(cls, d: dict):
        return cls(**d)

    @classmethod
    def from_yaml(cls, source: Union[PosixPath, str]):
        """
        Returns instance from a yaml filename or stdin
        """
        is_buffer = True
        try:
            if Path(source).exists():
                is_buffer = False
        except Exception as _:
            assert isinstance(source, str), "Invalid source type."

        if is_buffer:
            params = safe_load(source)
        else:
            with open(source) as f:
                params = safe_load(f)
        return cls(**params)

    @classmethod
    def default(cls, **kwargs):
        overriden_defaults = copy(cls._defaults)
        for k, v in kwargs.items():
            overriden_defaults[k] = v
        return cls.from_dict(overriden_defaults)

    def update(self, name: str, new_value):
        """
        Update values recursively
        if name is a dictionary, replace data where existing found or add if not.
        It warns against type changes.

        If the existing structure under name is a dictionary,
        it looks for the first occurrence and modifies it accordingly.

        If a leaf node that is to be changed is a collection, it adds the new elements.
        """

        assert name not in (
            "parameters",
            "params",
        ), "Attribute can't be named params or parameters"

        if name in self.__dict__:
            if check_type_recursive(getattr(self, name), new_value):
                print("Warnings:Type changes are risky")

            if isinstance(getattr(self, name), dict):
                flattened = flatten(self.to_dict())
                names_found = [k for k in flattened.keys() if name in k]
                found_idx = [keys.index(name) for keys in names_found]

                assert len(names_found), f"{name} not found as key."

                keys = None
                if len(names_found) > 1:
                    for level in zip(found_idx, names_found):
                        if level == min(found_idx):
                            keys = level
                            print(
                                f"Warning: {name} was found in multiple keys. Selected {keys}"
                            )
                            break

                else:
                    keys = names_found.pop()

                if keys:
                    current_val = flattened.get(keys, None)
                    # if isinstance(current_val, t.Collection):

            elif isinstance(getattr(self, name), t.Collection):
                add_to_collection(getattr(self, name), new_value)

            elif isinstance(getattr(self, name), set):
                pass  # TODO implement

            new_d = getattr(self, name)
            new_d.update(new_value)
            setattr(self, name, new_d)

        else:
            setattr(self, name, new_value)


def add_to_collection(
    collection: t.Collection, value: t.Union[atomic, t.Collection]
):
    # Adds element(s) in place.
    if not isinstance(value, t.Collection):
        value = [value]
    if isinstance(collection, list):
        collection += value
    elif isinstance(collection, set):
        collection.update(value)


class ProcessABC(ABC):
    """
    Base class for processes.
    Defines parameters as attributes and requires run method to be defined.
    """

    def __init__(self, parameters):
        """
        Arguments
        ---------
        parameters: instance of ParametersABC
        """
        self._parameters = parameters
        # convert parameters to dictionary
        # and then define each parameter as an attribute
        for k, v in parameters.to_dict().items():
            setattr(self, k, v)

    @property
    def parameters(self):
        return self._parameters

    @abstractmethod
    def run(self):
        pass


def check_type_recursive(val1, val2):
    same_types = True
    if not isinstance(val1, type(val2)) and not all(
        type(x) in (PosixPath, str) for x in (val1, val2)  # Ignore str->path
    ):
        return False
    if not isinstance(val1, t.Iterable) and not isinstance(val2, t.Iterable):
        return isinstance(val1, type(val2))
    elif isinstance(val1, (tuple, list)) and isinstance(val2, (tuple, list)):
        return bool(
            sum([check_type_recursive(v1, v2) for v1, v2 in zip(val1, val2)])
        )
    elif isinstance(val1, dict) and isinstance(val2, dict):
        if not len(val1) or not len(val2):
            return False
        for k in val2.keys():
            same_types = same_types and check_type_recursive(val1[k], val2[k])
    return same_types
