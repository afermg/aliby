"""
Mutual information test
"""

# import pytest
import numpy as np
import pandas as pd

from postprocessor.core.multisignal.mi import mi, miParameters

# Sample data: cropped from replicate 1 in
# https://github.com/swainlab/mi-by-decoding/blob/master/matlab/ExampleScript/fig1_sfp1_replicates.json
SIGNAL_RICH = np.array(
    [
        [
            -1.39512436686014,
            -1.44046750531481,
            -1.67185664648421,
            -0.500474684796053,
            -0.97255340345062,
            -1.16250137971723,
        ],
        [
            -0.407742899002175,
            -0.619583901332133,
            -0.466156867298538,
            -0.69093319800047,
            -0.186360950573155,
            -0.791411909242518,
        ],
        [
            0.350152741857363,
            0.913407870351929,
            0.524645770050427,
            0.441917565610652,
            1.5228639153911,
            2.67310873357743,
        ],
        [
            0.524953681848925,
            0.653076029386848,
            1.24582647626295,
            0.776211754098582,
            -0.355200015764816,
            -0.0871128171616209,
        ],
        [
            -1.68461323842732,
            -1.43594025257403,
            -1.3114696359734,
            -0.956125193215477,
            -1.2863334639258,
            -0.963653392884438,
        ],
        [
            0.657671105178289,
            1.20192526734078,
            1.41272977531711,
            1.10313719899755,
            1.21218191767352,
            1.25148540716015,
        ],
    ]
)

SIGNAL_STRESS = np.array(
    [
        [
            0.360683309928096,
            0.653056477804747,
            0.609421809463519,
            0.26028011016996,
            -0.163807667201703,
            -0.725067314828773,
        ],
        [
            -1.7884489956977,
            -1.77274508168164,
            -1.38542947325363,
            -1.11368924913116,
            -1.54227678929895,
            -1.67197618502403,
        ],
        [
            0.246852644985541,
            0.961545692641162,
            -0.159373062144918,
            0.0990542384881887,
            -0.766446517169187,
            -1.20071098737371,
        ],
        [
            0.393236272245847,
            0.441250356135278,
            1.05344010654052,
            1.06399045807211,
            -0.305342136100235,
            -1.49833740305382,
        ],
        [
            -1.45454923818794,
            -1.07292739455483,
            -1.2991307659611,
            -1.15322537661844,
            -1.29894837314545,
            -1.8884055407594,
        ],
        [
            0.102130222571265,
            -1.07499178276524,
            -1.3148530608215,
            -0.765688535324232,
            -0.645377669611553,
            -0.937035540900562,
        ],
    ]
)

#  Expected output: produced by running a modified estimateMI() from
#  https://git.ecdf.ed.ac.uk/pswain/mutual-information/-/blob/master/MIdecoding.py
#  on the sample input data.  The modification was adding the few lines that
# forced a random state for each bootstrap as the 'i' variable, so that
# the output isconsistent and therefore useful for pytest.
MI_IQR = np.array(
    [
        [0, 0, 0],
        [0, 0, 0],
        [0, 0, 0],
        [0, 0, 0],
        [0.311278124459133, 0, 0.311278124459133],
        [0.311278124459133, 0.311278124459133, 0.311278124459133],
    ]
)


def convert_to_df(array, strain_name):
    multiindex_array = [[strain_name] * len(array), list(range(len(array)))]
    multiindex = pd.MultiIndex.from_arrays(multiindex_array, names=("strain", "cellID"))
    signal = pd.DataFrame(array, multiindex)

    return signal


def test_mi():
    """Tests mi.

    Tests whether an mi runner can be initialised with default
    parameters and runs on sample data, giving expected output.
    """
    dummy_signals = [
        convert_to_df(SIGNAL_RICH, "rich"),
        convert_to_df(SIGNAL_STRESS, "stress"),
    ]
    params = miParameters.default()
    params.train_test_split_seeding = True
    mi_runner = mi(params)
    res = mi_runner.run(dummy_signals)
    assert np.allclose(res, MI_IQR)
