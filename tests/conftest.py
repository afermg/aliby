import pytest
from pathlib import Path
import pooch

@pytest.fixture(scope="session")
def data_dir():
    data_path = Path("/datastore/alan/aliby/test_dataset/data/")
    if not data_path.exists():
        marker = "aliby_tests/"
        files = pooch.retrieve(
            url="https://zenodo.org/api/records/19411429/files/aliby_test_dataset.tar.gz/content",
            known_hash="3a8b1b7b362f002098ba44e65622862057cfe46f0b459514bf270349c8bce4a7",
            fname="aliby_test_dataset.tar.gz",
            processor=pooch.Untar(extract_dir="aliby_tests"),
        )
        data_path = Path(files[0].split(marker)[0] + marker)
    return data_path
