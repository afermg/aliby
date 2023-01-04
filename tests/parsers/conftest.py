#!/usr/bin/env jupyter

from importlib_resources import files
from logfile_parser import Parser
from logfile_parser.legacy import get_legacy_log_example_interface

import pytest

examples_dir = files("aliby").parent.parent / "examples" / "logfile_parser"
grammars_dir = files("logfile_parser") / "grammars"


@pytest.fixture(scope="module", autouse=True)
def legacy_log_interface() -> dict:
    return get_legacy_log_example_interface()
