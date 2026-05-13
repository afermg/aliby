"""
Regression test: verify dl.df numerical values are stable across runs.

To run the full pipeline and compare against golden (default) dataframe:
    pytest --run-slow

To run the pipeline and write new golden dataframe:
    pytest --run-slow --update-golden
"""

import pickle
import pytest
import pandas as pd
from pathlib import Path

GOLDEN = Path(__file__).parent / "data" / "golden_df.pkl"
OMID = "test26643"
H5DIR = "/Users/pswain/wip/aliby_output/"
OMERO_DIR = "/Users/pswain/wip/aliby_input/"

# columns to compare with golden dataframe
KEY_COLS = [
    "buddings",
    "bud_volume",
    "mother_volume",
    "mother_mean_Flavin",
    "mother_mean_mCherry",
]


def _pipeline_params():
    from aliby.pipeline import PipelineParameters

    return PipelineParameters.default(
        general={
            "expt_id": OMERO_DIR + OMID,
            "distributed": 0,
            "directory": H5DIR,
            "filter": "htb2mCherry_001",
            "overwrite": True,
            "tps": 8,
        },
    )


def _load_df() -> pd.DataFrame:
    """Load dl.df from existing H5 output, sorted for stable row order."""
    from wela.dataloader import DataLoader

    dl = DataLoader(H5DIR, ".")
    dl.load(OMID, key_index="buddings", cutoff=0)
    return dl.df.sort_values(["id", "time"]).reset_index(drop=True)


@pytest.mark.slow
def test_df_regression(update_golden):
    """
    Run the full pipeline and check key columns match the golden file.

    Rows are matched on (id, time); only cells present in both the
    current run and the golden are compared. Regenerate the golden
    file with::

        pytest tests/test_regression.py --run-slow --update-golden
    """
    from aliby.pipeline import Pipeline

    Pipeline(_pipeline_params()).run()
    actual = _load_df()
    if update_golden:
        GOLDEN.parent.mkdir(parents=True, exist_ok=True)
        with open(GOLDEN, "wb") as f:
            pickle.dump(actual, f, protocol=5)
        pytest.skip("golden file updated")
        return
    assert GOLDEN.exists(), (
        f"golden file not found: {GOLDEN}\n"
        "run with --update-golden to create it"
    )
    with open(GOLDEN, "rb") as f:
        expected = pickle.load(f)
    join_keys = ["id", "time"]
    merged = actual.merge(
        expected[join_keys + KEY_COLS],
        on=join_keys,
        suffixes=("", "_expected"),
    )
    assert (
        len(merged) > 0
    ), "no common (id, time) pairs between actual and golden df"
    for col in KEY_COLS:
        pd.testing.assert_series_equal(
            merged[col].rename(col),
            merged[f"{col}_expected"].rename(col),
            check_exact=False,
            rtol=1e-5,
            check_dtype=False,
        )
