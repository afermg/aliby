"""Pytest configuration and shared fixtures."""

import pytest


def pytest_addoption(parser):
    parser.addoption(
        "--update-golden",
        action="store_true",
        default=False,
        help="regenerate golden files instead of comparing",
    )
    parser.addoption(
        "--run-slow",
        action="store_true",
        default=False,
        help="include slow integration tests",
    )


def pytest_collection_modifyitems(config, items):
    if not config.getoption("--run-slow"):
        skip = pytest.mark.skip(reason="use --run-slow to include")
        for item in items:
            if "slow" in item.keywords:
                item.add_marker(skip)


@pytest.fixture
def update_golden(request):
    return request.config.getoption("--update-golden")
