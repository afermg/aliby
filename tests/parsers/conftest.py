#!/usr/bin/env python

from importlib.resources import files
import pytest

try:
    examples_dir = files("aliby").parent.parent / "examples" / "logfile_parser"
    grammars_dir = files("logfile_parser") / "grammars"
except Exception:
    examples_dir = None
    grammars_dir = None


@pytest.fixture(scope="module", autouse=True)
def swainlab_log_interface() -> str:
    try:
        return str(
            files("aliby").parent.parent
            / "examples"
            / "parsers"
            / "swainlab_logfile_header_example.log"
        )
    except Exception:
        return "swainlab_logfile_header_example.log"
