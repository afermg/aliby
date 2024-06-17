import itertools
import re
import typing as t
from pathlib import Path

import numpy as np
from baby import BabyCrawler, modelsets

from agora.abc import ParametersABC, StepABC


class BabyParameters(ParametersABC):
    """Parameters used for running BABY."""

    def __init__(
        self,
        modelset_name,
        clogging_thresh,
        min_bud_tps,
        isbud_thresh,
    ):
        """Initialise parameters for BABY."""
        # pixel_size is specified in BABY's model sets
        self.modelset_name = modelset_name
        self.clogging_thresh = clogging_thresh
        self.min_bud_tps = min_bud_tps
        self.isbud_thresh = isbud_thresh

    @classmethod
    def default(cls, **kwargs):
        """Define default parameters; kwargs choose BABY model set."""
        return cls(
            modelset_name=get_modelset_name_from_params(**kwargs),
            clogging_thresh=1,
            min_bud_tps=3,
            isbud_thresh=0.5,
        )

    def update_baby_modelset(self, path: t.Union[str, Path, t.Dict[str, str]]):
        """
        Replace default BABY model and flattener.

        Both are saved in a folder by our retraining script.
        """
        if isinstance(path, dict):
            weights_flattener = {k: Path(v) for k, v in path.items()}
        else:
            weights_dir = Path(path)
            weights_flattener = {
                "flattener_file": weights_dir.parent / "flattener.json",
                "morph_model_file": weights_dir / "weights.h5",
            }
        self.update("modelset_name", weights_flattener)


class BabyRunner(StepABC):
    """
    A BabyRunner object for cell segmentation.

    Segments one time point at a time.
    """

    def __init__(self, tiler, parameters=None, **kwargs):
        """Instantiate from a Tiler object."""
        self.tiler = tiler
        modelset_name = (
            get_modelset_name_from_params(**kwargs)
            if parameters is None
            else parameters.modelset_name
        )
        tiler_z = self.tiler.image.shape[-3]
        if f"{tiler_z}z" not in modelset_name:
            raise KeyError(
                f"Tiler z-stack ({tiler_z}) and model"
                f" ({modelset_name}) do not match."
            )
        if parameters is None:
            brain = modelsets.get(modelset_name)
        else:
            brain = modelsets.get(
                modelset_name,
                clogging_thresh=parameters.clogging_thresh,
                min_bud_tps=parameters.min_bud_tps,
                isbud_thresh=parameters.isbud_thresh,
            )
        self.crawler = BabyCrawler(brain)
        self.brightfield_channel = self.tiler.ref_channel_index

    @classmethod
    def from_tiler(cls, parameters: BabyParameters, tiler):
        """Explicitly instantiate from a Tiler object."""
        return cls(tiler, parameters)

    def get_data(self, tp):
        """Get image and re-arrange axes."""
        img_from_tiler = self.tiler.get_tp_data(tp, self.brightfield_channel)
        # move z axis to the last axis; Baby expects (n, x, y, z)
        img = np.moveaxis(img_from_tiler, 1, destination=-1)
        return img

    def _run_tp(
        self,
        tp,
        refine_outlines=True,
        assign_mothers=True,
        with_edgemasks=True,
        **kwargs,
    ):
        """Segment data from one time point."""
        img = self.get_data(tp)
        segmentation = self.crawler.step(
            img,
            refine_outlines=refine_outlines,
            assign_mothers=assign_mothers,
            with_edgemasks=with_edgemasks,
            **kwargs,
        )
        res = format_segmentation(segmentation, tp)
        return res


def get_modelset_name_from_params(
    imaging_device="alcatras",
    channel="brightfield",
    camera="sCMOS",
    zoom="60x",
    n_stacks="5z",
):
    """Get the appropriate model set from BABY's trained models."""
    # list of models - microscopy setups - for which BABY has been trained
    # cameras prime95 and evolve have become sCMOS and EMCCD
    possible_models = list(modelsets.remote_modelsets()["models"].keys())

    # filter possible_models
    params = [
        str(x) if x is not None else ".+"
        for x in [imaging_device, channel.lower(), camera, zoom, n_stacks]
    ]
    params_regex = re.compile("-".join(params) + "$")
    valid_models = [
        res for res in filter(params_regex.search, possible_models)
    ]
    # check that there are valid models
    if len(valid_models) == 1:
        return valid_models[0]
    else:
        raise KeyError(
            "Error in finding BABY model sets matching {}".format(
                ", ".join(params)
            )
        )


def format_segmentation(segmentation, tp):
    """
    Format BABY's results for a single time point into a dict.

    The dict has BABY's outputs as keys and lists of the results
    for each segmented cell as values.

    Parameters
    ------------
    segmentation: list
        A list of BABY's results as dicts for each tile.
    tp: int
        The time point.
    """
    # segmentation is a list of dictionaries for each tile
    for i, tile_dict in enumerate(segmentation):
        # assign the trap ID to each cell identified
        tile_dict["trap"] = [i] * len(tile_dict["cell_label"])
        # record mothers for each labelled cell
        tile_dict["mother_assign_dynamic"] = np.array(
            tile_dict["mother_assign"]
        )[np.array(tile_dict["cell_label"], dtype=int) - 1]
    # merge into a dict with BABY's outputs as keys and
    # lists of results for all cells as values
    merged = {
        output: list(
            itertools.chain.from_iterable(
                tile_dict[output] for tile_dict in segmentation
            )
        )
        for output in segmentation[0].keys()
    }
    # remove mother_assign
    merged.pop("mother_assign", None)
    # ensure that each value is a list of the same length
    no_cells = min([len(v) for v in merged.values()])
    merged = {k: v[:no_cells] for k, v in merged.items()}
    # define time point key
    merged["timepoint"] = [tp] * no_cells
    return merged
