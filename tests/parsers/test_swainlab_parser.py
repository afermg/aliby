#!/usr/bin/env python

from pathlib import Path

from logfile_parser.swainlab_parser import parse_swainlab_logs


def test_swainlab_parser(swainlab_log_interface: str):
    if not Path(swainlab_log_interface).exists():
        import pytest

        pytest.skip("Logfile not found")
    return parse_swainlab_logs(swainlab_log_interface)
