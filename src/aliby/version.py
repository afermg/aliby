"""Get the version from the installed package's metadata."""
from importlib.metadata import version

__version__ = version("alibylite")
