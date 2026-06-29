import pytest

from aliby.test_data import get_data_root


@pytest.fixture(scope="session")
def data_dir():
    """Path to the unpacked aliby test dataset (Zenodo record 19411429).

    Resolved by :func:`aliby.test_data.get_data_root` -- uses the legacy
    on-disk copy at ``/datastore/alan/aliby/test_dataset/data/`` when
    present, otherwise fetches from Zenodo via pooch and caches under
    ``~/.cache/pooch/aliby_tests/``.
    """
    return get_data_root()
