# Example of argo experiment explorer
import pytest
from aliby.utils.argo import Argo


@pytest.mark.skip(reason="no way of testing this without sensitive info")
def test_load():
    argo = Argo()
    argo.load()
    return 1


@pytest.mark.skip(reason="no way of testing this without sensitive info")
def test_channel_filter():
    argo = Argo()
    argo.load()
    argo.channels("GFP")
    return 1


@pytest.mark.skip(reason="no way of testing this without sensitive info")
def test_tags():
    argo = Argo()
    argo.load()
    argo.channels("GFP")
    argo.tags(["Alan", "batgirl"])
    return 1


@pytest.mark.skip(reason="no way of testing this without sensitive info")
def test_timepoint():
    pass
