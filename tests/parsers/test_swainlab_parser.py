#!/usr/bin/env jupyter

from pathlib import Path

from logfile_parser.swainlab_parser import parse_from_swainlab_grammar


def test_swainlab_parser(swainlab_log_interface: Path):
    return parse_from_swainlab_grammar(swainlab_log_interface)
