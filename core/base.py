from abc import ABC, abstractmethod


class ParametersABC(ABC):
    """
    Base class to add yaml functionality to parameters

    """

    def to_dict(self):
        return self.__dict__

    @classmethod
    def from_dict(cls, d):
        return cls(**d)

    def to_yaml(self, path=None):
        return dump(self.__dict__, path)

    @classmethod
    def from_yaml(cls, yam):
        with open(Path(yam)) as f:
            params = safe_load(f)
        return cls(**params)

    @classmethod
    @abstractmethod
    def default(cls):
        pass


class ProcessABC(ABC):
    "Base class for processes"

    @property
    @abstractmethod
    def parameters(self):
        return self.parameters

    @abstractmethod
    def run(self):
        pass
