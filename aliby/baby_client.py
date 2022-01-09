import collections
import itertools
import json
import time
from pathlib import Path
from typing import Iterable

import h5py
import numpy as np
import pandas as pd
import re
import requests
import tensorflow as tf
from tqdm import tqdm

from agora.abc import ParametersABC, ProcessABC
import baby.errors
from baby import modelsets
from baby.brain import BabyBrain
from baby.crawler import BabyCrawler
from requests.exceptions import Timeout, HTTPError
from requests_toolbelt.multipart.encoder import MultipartEncoder

from agora.io.utils import Cache, accumulate, get_store_path


################### Dask Methods ################################
def format_segmentation(segmentation, tp):
    """Format a single timepoint into a dictionary.

    Parameters
    ------------
    segmentation: list
                  A list of results, each result is the output of the crawler, which is JSON-encoded
    tp: int
        the time point considered

    Returns
    --------
    A dictionary containing the formatted results of BABY
    """
    # Segmentation is a list of dictionaries, ordered by trap
    # Add trap information
    # mother_assign = None
    for i, x in enumerate(segmentation):
        x["trap"] = [i] * len(x["cell_label"])
        x["mother_assign_dynamic"] = np.array(x["mother_assign"])[
            np.array(x["cell_label"], dtype=int) - 1
        ]
    # Merge into a dictionary of lists, by column
    merged = {
        k: list(itertools.chain.from_iterable(res[k] for res in segmentation))
        for k in segmentation[0].keys()
    }
    # Special case for mother_assign
    # merged["mother_assign_dynamic"] = [merged["mother_assign"]]
    if "mother_assign" in merged:
        del merged["mother_assign"]
        mother_assign = [x["mother_assign"] for x in segmentation]
    # Check that the lists are all of the same length (in case of errors in
    # BABY)
    n_cells = min([len(v) for v in merged.values()])
    merged = {k: v[:n_cells] for k, v in merged.items()}
    merged["timepoint"] = [tp] * n_cells
    merged["mother_assign"] = mother_assign
    return merged


class BabyParameters(ParametersABC):
    def __init__(
        self,
        model_config,
        tracker_params,
        clogging_thresh,
        min_bud_tps,
        isbud_thresh,
        session,
        graph,
        print_info,
        suppress_errors,
        error_dump_dir,
        tf_version,
    ):
        self.model_config = model_config
        self.tracker_params = tracker_params
        self.clogging_thresh = clogging_thresh
        self.min_bud_tps = min_bud_tps
        self.isbud_thresh = isbud_thresh
        self.session = session
        self.graph = graph
        self.print_info = print_info
        self.suppress_errors = suppress_errors
        self.error_dump_dir = error_dump_dir
        self.tf_version = tf_version

    @classmethod
    def default(cls, **kwargs):
        """kwargs passes values to the model chooser"""
        return cls(
            model_config=choose_model_from_params(**kwargs),
            tracker_params=dict(ctrack_params=dict(), budtrack_params=dict()),
            clogging_thresh=1,
            min_bud_tps=3,
            isbud_thresh=0.5,
            session=None,
            graph=None,
            print_info=False,
            suppress_errors=False,
            error_dump_dir=None,
            tf_version=2,
        )


class BabyRunner:
    """A BabyRunner object for cell segmentation.

    Does segmentation one time point at a time."""

    def __init__(self, tiler, parameters=None, *args, **kwargs):
        self.tiler = tiler
        # self.model_config = modelsets()[choose_model_from_params(**kwargs)]
        self.model_config = modelsets()[
            (
                parameters.model_config
                if parameters is not None
                else choose_model_from_params(**kwargs)
            )
        ]
        self.brain = BabyBrain(**self.model_config)
        self.crawler = BabyCrawler(self.brain)
        self.bf_channel = self.tiler.get_channel_index("Brightfield")

    @classmethod
    def from_tiler(cls, parameters: BabyParameters, tiler):
        return cls(tiler, parameters)

    def get_data(self, tp):
        # Swap axes x and z, probably shouldn't swap, just move z
        return self.tiler.get_tp_data(tp, self.bf_channel).swapaxes(1, 3).swapaxes(1, 2)

    def run_tp(self, tp, with_edgemasks=True, assign_mothers=True, **kwargs):
        """Simulating processing time with sleep"""
        # Access the image
        img = self.get_data(tp)
        segmentation = self.crawler.step(
            img, with_edgemasks=with_edgemasks, assign_mothers=assign_mothers, **kwargs
        )
        return format_segmentation(segmentation, tp)


class BabyClient:
    """A dummy BabyClient object for Dask Demo.


    Does segmentation one time point at a time.
    Should work better with the parallelisation.
    """

    bf_channel = 0
    model_name = "prime95b_brightfield_60x_5z"
    url = "http://localhost:5101"
    max_tries = 50
    sleep_time = 0.1

    def __init__(self, tiler, *args, **kwargs):
        self.tiler = tiler
        self._session = None

    @property
    def session(self):
        if self._session is None:
            r_session = requests.get(self.url + f"/session/{self.model_name}")
            r_session.raise_for_status()
            self._session = r_session.json()["sessionid"]
        return self._session

    def get_data(self, tp):
        return self.tiler.get_tp_data(tp, self.bf_channel).swapaxes(1, 3)

    def queue_image(self, img, **kwargs):
        bit_depth = img.dtype.itemsize * 8  # bit depth =  byte_size * 8
        data = create_request(img.shape, bit_depth, img, **kwargs)
        status = requests.post(
            self.url + f"/segment?sessionid={self.session}",
            data=data,
            headers={"Content-Type": data.content_type},
        )
        status.raise_for_status()
        return status

    def get_segmentation(self):
        try:
            seg_response = requests.get(
                self.url + f"/segment?sessionid={self.session}", timeout=120
            )
            seg_response.raise_for_status()
            result = seg_response.json()
        except Timeout as e:
            raise e
        except HTTPError as e:
            raise e
        return result

    def run_tp(self, tp, **kwargs):
        # Get data
        img = self.get_data(tp)
        # Queue image
        status = self.queue_image(img, **kwargs)
        # Get segmentation
        for _ in range(self.max_tries):
            try:
                seg = self.get_segmentation()
                break
            except (Timeout, HTTPError):
                time.sleep(self.sleep_time)
                continue
        return format_segmentation(seg, tp)


def choose_model_from_params(
    modelset_filter=None,
    camera="prime95b",
    channel="brightfield",
    zoom="60x",
    n_stacks="5z",
    **kwargs,
):
    """
    Define which model to query from the server based on a set of parameters.

    Parameters
    ----------
    valid_models: List[str]
                  The names of the models that are available.
    modelset_filter: str
                    A regex filter to apply on the models to start.
    camera: str
            The camera used in the experiment (case insensitive).
    channel:str
            The channel used for segmentation (case insensitive).
    zoom: str
          The zoom on the channel.
    n_stacks: str
              The number of z_stacks to use in segmentation

    Returns
    -------
    model_name : str
    """
    valid_models = list(modelsets().keys())

    # Apply modelset filter if specified
    if modelset_filter is not None:
        msf_regex = re.compile(modelset_filter)
        valid_models = filter(msf_regex.search, valid_models)

    # Apply parameter filters if specified
    params = [
        str(x) if x is not None else ".+"
        for x in [camera.lower(), channel.lower(), zoom, n_stacks]
    ]
    params_re = re.compile("^" + "_".join(params) + "$")
    valid_models = list(filter(params_re.search, valid_models))
    # Check that there are valid models
    if len(valid_models) == 0:
        raise KeyError("No model sets found matching {}".format(", ".join(params)))
    # Pick the first model
    return valid_models[0]
