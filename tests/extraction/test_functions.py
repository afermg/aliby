import numpy as np
from pathlib import Path
from extraction.core.extractor import Extractor, ExtractorParameters

params = ExtractorParameters.from_meta(
    {"channels/channel": ["Brightfield", "GFPFast", "pHluorin405", "mCherry"]}
)
ext = Extractor(params)
ext.load_funs()


def test_custom_output():
    self = ext
    mask = np.zeros((6, 6, 2), dtype=bool)
    mask[2:4, 2:4, 0] = True
    mask[3:5, 3:5, 1] = True
    img = np.random.randint(1, 11, size=6 ** 2 * 5).reshape(6, 6, 5)

    for i, f in self._custom_funs.items():
        if "3d" in i:
            res = f(mask, img)
        else:
            res = f(mask, np.maximum.reduce(img, axis=2))
        assert len(res) == mask.shape[2], "Output doesn't match input"
