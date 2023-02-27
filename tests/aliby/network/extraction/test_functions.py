import numpy as np

from extraction.core.extractor import Extractor, ExtractorParameters

params = ExtractorParameters.from_meta(
    {"channels": ["Brightfield", "GFPFast", "pHluorin405", "mCherry"]}
)
ext = Extractor(params)
ext.load_funs()


def test_custom_output():
    self = ext
    mask = np.zeros((2, 6, 6), dtype=bool)
    mask[0, 2:4, 2:4] = True
    mask[1, 3:5, 3:5] = True
    img = np.random.randint(1, 11, size=6**2 * 5).reshape(5, 6, 6)

    for i, f in self._custom_funs.items():
        if "3d" in i:
            res = f(mask, img)
        else:
            res = f(mask, np.maximum.reduce(img, axis=0))
        assert len(res) == mask.shape[0], "Output doesn't match input"
