#!/usr/bin/env jupyter


def pytest_addoption(parser):
    parser.addoption("--file", action="store", default="test_datasets")


def pytest_generate_tests(metafunc):
    # This is called for every test. Only get/set command line arguments
    # if the argument is specified in the list of test "fixturenames".
    option_value = metafunc.config.option.file
    if "file" in metafunc.fixturenames and option_value is not None:
        metafunc.parametrize("file", [option_value])


#
