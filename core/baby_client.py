import collections
import itertools
import json
from pathlib import Path
from typing import Iterable

import h5py
import numpy as np
import pandas as pd
import re
import requests
from sqlalchemy.orm.exc import NoResultFound
import tensorflow as tf
from tqdm import tqdm

from baby import modelsets
from baby.brain import BabyBrain
from baby.crawler import BabyCrawler
from requests.exceptions import Timeout, HTTPError
from requests_toolbelt.multipart.encoder import MultipartEncoder

from core.utils import Cache, accumulate


class BabyNoMatches(Exception):
    pass


class BabyNoSilent(Exception):
    pass


# Todo: add defaults!
def choose_model_from_params(valid_models,
                             modelset_filter=None, camera=None, channel=None,
                             zoom=None, n_stacks=None, **kwargs):
    """
    Define which model to query from the server based on a set of parameters.
    This depends on:
    :param valid_models: The names of the models that are available.
    :param modelset_filter: A regex filter to apply on the models to start.
    :param camera: The camera used in the experiment (case insensitive).
    :param channel: The channel used for segmentation (case insensitive).
    :param zoom: The zoom on the channel.
    :param n_stacks: The number of z_stacks to use in segmentation.
    :return:
    """
    # TODO specify which z-stacks in the Segmentation class if the number of
    #  stacks is fewer than the total number of available z-stacks.

    # Apply modelset filter if specified
    if modelset_filter is not None:
        msf_regex = re.compile(modelset_filter)
        valid_models = filter(msf_regex.search, valid_models)

    # Apply parameter filters if specified
    params = [str(x) if x is not None else '.+' for x in [camera.lower(),
                                                          channel.lower(),
                                                          zoom, n_stacks]]
    params_re = re.compile('^' + '_'.join(params) + '$')
    valid_models = list(filter(params_re.search, valid_models))
    # Check that there are valid models
    if len(valid_models) == 0:
        raise BabyNoMatches(
            "No model sets found matching {}".format(', '.join(params)))
    # Pick the first model
    return valid_models[0]


def create_request(dims, bit_depth, img, **kwargs):
    """
    Construct a multipart/form-data request with the following
    information in the given order:
    :param session_id: the session ID (for tracking)
    :param dims: the dimensions of the images
    :param bit_depth: the bit-depth of the images, must be "8" or "16"
    :param img: the image to segment, flattened in order 'F'
    :return: a MultipartEncoder to use as data for the request.
    """
    fields = collections.OrderedDict([
        ("dims", json.dumps(dims)),
        ("bitdepth", json.dumps(bit_depth)),
        ("img", img.tostring(order='F'))])
    # Add optional arguments
    fields.update({kw: json.dumps(v) for kw, v in kwargs.items()})
    m = MultipartEncoder(
        fields=fields,
        boundary="----BabyFormBoundary"
    )
    return m


class BabyClient:
    def __init__(self, tiler, url='http://localhost:5101', **kwargs):
        self.tiler = tiler
        self.url = url
        self._config = kwargs
        r_model_sets = requests.get(self.url + '/models')
        self.valid_models = r_model_sets.json()
        self._model_set = choose_model_from_params(self.valid_models,
                                                   **self.config)
        self.sessions = Cache(load_fn=lambda _: self.get_new_session())
        self.processing = []
        self.z = None
        self.channel = None
        self.__init_properties(self.config)

    def __init_properties(self, config):
        n_stacks = int(config.get('n_stacks', '5z').replace('z', ''))
        self.z = list(range(n_stacks))
        self.channel = config.get('channel', 'Brightfield')

    @property
    def model_set(self):
        return self._model_set

    @model_set.setter
    def model_set(self, model_set):
        if self._model_set != model_set:
            # Need a new session if the model_set has changed
            self.session_id = ""
            self._model_set = model_set
        else:
            pass

    @property
    def config(self):
        return self._config

    @config.setter
    def config(self, config):
        if self._config is not None and config is not None:
            raise BabyNoSilent("Can only silently set a configuration "
                               "to/from None")
        else:
            self._config = config
            self.model_set = choose_model_from_params(**self.config)

    def get_new_session(self):
        try:
            r_session = requests.get(self.url +
                                     '/session/{}'.format(self.model_set))
            r_session.raise_for_status()
            return r_session.json()["sessionid"]
        except KeyError as e:
            raise e
        except HTTPError as e:
            raise e

    def queue_image(self, img, session_id, **kwargs):
        # TODO validate image type?
        # TODO character encoding options?
        bit_depth = img.dtype.itemsize * 8  # bit depth =  byte_size * 8
        data = create_request(img.shape, bit_depth, img, **kwargs)
        status = requests.post(self.url +
                               '/segment?sessionid={}'.format(session_id),
                               data=data,
                               headers={'Content-Type': data.content_type})
        status.raise_for_status()
        return status

    def get_segmentation(self, session_id):
        try:
            seg_response = requests.get(
                self.url + '/segment?sessionid={}'.format(session_id),
                timeout=120)
            seg_response.raise_for_status()
            result = seg_response.json()
        except Timeout as e:
            raise e
        except HTTPError as e:
            raise e
        return result

    def process_position(self, prompt: str, tps: Iterable[int], session,
                         store, tile_size=96):
        position_results = []
        # The prompt received by baby is the position
        try:
            for timepoint_id in tps:
                # Finish processing previously queued images
                self.flush_processing(position_results)
                self.process_timepoint(prompt, timepoint_id,
                                       tile_size=tile_size)
        except KeyError as e:
            # TODO log that this will not be processed
            raise e
        # Flush all processing before moving to the next position
        while len(self.processing) > 0:
            mother_assign = self.flush_processing(position_results)
        store_position(position_results, self.tiler.positions.index(prompt),
                       store, position_name=prompt,
                       mother_assign=mother_assign, tile_size=tile_size)
        return

    def process_timepoint(self, pos, timepoint, tile_size=96):
        channel_idx = [self.tiler.get_channel_index(self.channel)]
        traps = self.tiler[pos].get_traps_timepoint(timepoint,
                                                    channels=channel_idx,
                                                    tile_size=tile_size,
                                                    z=self.z)
        traps = np.squeeze(traps)
        timepoint_key = (pos, timepoint)
        session_id = self.sessions[pos]
        print(traps.shape)
        self.queue_image(traps, session_id)
        self.processing.append(timepoint_key)

    def flush_processing(self, position_results):
        """ Get the results of previously queued images.

        :return:
        """
        for pos, tp in self.processing:
            try:
                result = self.get_segmentation(self.sessions[pos])
                tp_dataframe, mother_assign = format_segmentation(result, tp)
                position_results.append(tp_dataframe)
                self.processing.remove((pos, tp))
            except Timeout:
                continue
            except HTTPError:
                continue
            except TypeError as e:
                raise e
        return mother_assign

    def run(self, keys, store='store.h5', **kwargs):
        # key are (pos, timepoint) tuples
        for pos, tps in accumulate(keys):
            self.process_position(pos, tps, store, **kwargs)
        return keys


def format_segmentation(segmentation, tp):
    """ Format a single timepoint into a dataframe and append to the
    position results.
    :param segmentation: A list of results, each result is the output of the
    crawler, which is JSON-encoded
    :param tp: The time point considered
    :return: A pandas dataframe containing the formatted results of BABY
    """
    # Segmentation is a list of dictionaries, ordered by trap
    # Add trap information
    mother_assign = None
    for i, x in enumerate(segmentation):
        x['trap'] = [i] * len(x['cell_label'])
    # Merge into a dictionary of lists, by column
    merged = {k: list(itertools.chain.from_iterable(
        res[k] for res in segmentation))
        for k in segmentation[0].keys()}
    if 'mother_assign' in merged:
        del merged['mother_assign']
        mother_assign = [x['mother_assign'] for x in segmentation]
    # Special case for mother_assign
    tp_dataframe = pd.DataFrame(merged)
    # Set time point value for all traps
    tp_dataframe['timepoint'] = tp
    return tp_dataframe, mother_assign


def store_position(position_results, position_index, store,
                   position_name=None, mother_assign=None, tile_size=96):
    """Store the results from a set of timepoints for a given position to
    and HDF5 store
    :param position_results: List of timepoint dataframes as returned by
    `format_segmentation`
    :param position_index: The index of the position considered
    :param store: The name of the HDF5 store to use.
    :return:
    """
    # Combine all of the results into one data frame
    position_results = pd.concat(position_results)
    # Set the position name explicitly
    # position_results['position'] = position_index
    if position_name:
        # This means to create a separate store for each position
        store = Path(store)
        store = store.with_name(position_name + store.name)
    # Append the results to the store
    df_to_hdf(position_results, store, mother_assign=mother_assign,
              tile_size=tile_size)
    return


def sparsity(arr):
    """Defines a sparsity score for a matrix based on the percentage of
    zeros."""
    try:
        return 1.0 - np.count_nonzero(arr) / arr.size
    except:
        return 1


def df_to_hdf(df, filename, mother_assign=None, tile_size=96):
    """Convert the dataframe of segmentation results into an HDF5 file.
    :param df: The dataframe.
    :param filename: The Name of the HDF5 file to use.
    :return:
    """
    datatypes = {
        'centres': ((None, 2), np.uint16),
        'position': ((None,), np.uint16),
        'angles': ((None,), h5py.vlen_dtype(np.float32)),
        'radii': ((None,), h5py.vlen_dtype(np.float32)),
        'edgemasks': ((None, tile_size, tile_size), np.bool),
        'ellipse_dims': ((None, 2), np.float32),
        'cell_label': ((None,), np.uint16),
        'trap': ((None,), np.uint16),
        'timepoint': ((None,), np.uint16),
        'mother_assign': ((None,), np.uint16)
    }

    hfile = h5py.File(filename, 'a')
    n = len(df)
    for key in df.columns:
        # We're only saving data that has a pre-defined data-type
        if key not in datatypes:
            raise KeyError(f"No defined data type for key {key}")
        if key not in hfile:
            # TODO Include sparsity check
            max_shape, dtype = datatypes[key]
            shape = (n,) + max_shape[1:]
            data = df[key].to_list()
            hfile.create_dataset(key, shape=shape, maxshape=max_shape,
                                 dtype=dtype, compression='gzip')
            hfile[key][()] = data
        else:
            # The dataset already exists, expand it
            dset = hfile[key]
            dset.resize(dset.shape[0] + n, axis=0)
            dset[-n:] = df[key].tolist()
    if mother_assign:
        # We do not append to mother_assign; raise error if already saved
        n = len(mother_assign)
        hfile.create_dataset('mother_assign', shape=(n,),
                             dtype=h5py.vlen_dtype(np.uint16),
                             compression='gzip')
        hfile['mother_assign'][()] = mother_assign
    hfile.close()
    return


class BabyRunner:
    valid_models = modelsets()
    ERROR_DUMP_DIR = 'baby-errors'

    def __init__(self, tiler, error_dump_dir=None, **kwargs):
        self.tiler = tiler
        if error_dump_dir is None:
            self.error_dump_dir = self.ERROR_DUMP_DIR
        self._config = kwargs
        model_name = choose_model_from_params(self.valid_models, **self.config)
        self.sessions = Cache(load_fn=lambda _: self.session())
        self.z = None
        self.channel = None
        self.default_image_size = None
        self.__init_properties(self.config)
        # Create tensorflow objects
        self.tf_session = None
        self.tf_graph = None
        # TODO: put the tensorflow initilization in a separate function
        tf_version = tuple(int(v) for v in tf.version.VERSION.split('.'))
        if tf_version[0] == 1:
            config = tf.ConfigProto()
            config.gpu_options.allow_growth = True
            self.tf_session = tf.Session(config=config)
            self.tf_graph = tf.get_default_graph()
        elif tf_version[0] == 2:
            gpus = tf.config.experimental.list_physical_devices('GPU')
            if gpus:
                for gpu in gpus:
                    tf.config.experimental.set_memory_growth(gpu, True)
                logical_gpus = tf.config.experimental.list_logical_devices(
                    'GPU')
                print(len(gpus), "Physical GPUs,", len(logical_gpus),
                      "Logical GPUs")
        # Overriding some of the default model values in baby to avoid errors
        model_config = self.valid_models[model_name]
        default_image_size = self.config.get("default_image_size", None)
        if default_image_size:
            model_config["default_image_size"] = default_image_size
            self.default_image_size = default_image_size
        # Getting the runner
        self.brain = BabyBrain(**model_config,
                               session=self.tf_session, graph=self.tf_graph,
                               suppress_errors=True,
                               error_dump_dir=self.error_dump_dir,
                               )

    @property
    def config(self):
        return self._config

    def __init_properties(self, config):
        n_stacks = int(config.get('n_stacks', '5z').replace('z', ''))
        self.z = list(range(n_stacks))
        self.channel = self.tiler.get_channel_index(
            config.get('channel', 'Brightfield'))

    def session(self):
        return BabyCrawler(self.brain)

    def segment(self, img, sessionid, **kwargs):
        # Getting the result for a given image
        crawler = self.sessions[sessionid]
        pred = crawler.step(img, **kwargs)
        return pred

    def process_position(self, position, tps, store, verbose, **kwargs):
        """ Segment the position for the given number of time points and save.

        :param position: The name of the position to segment
        :param tps: A list of time points on which to run the segmentation
        :param store: The file in which to save the results, as csv. Results
        are appended to this file so make sure not to use a previously used
        file name or you will have hard-to-find duplicates!
        :param verbose: Set to show progression of the time points
        :param kwargs: Additional segmentation parameters, to be given to
        the BABY crawler
        :return: None
        """
        self.tiler.current_position = position
        position_results = []
        for tp in tqdm(tps, desc=position, disable=not verbose):
            traps = np.squeeze(
                self.tiler.get_traps_timepoint(tp, channels=[self.channel],
                                               z=self.z,
                                               tile_size=self.default_image_size))
            segmentation = self.segment(traps, position, **kwargs)
            # Segmentation is a list of dictionaries, ordered by trap
            # Add trap information
            tp_dataframe, mother_assign = format_segmentation(segmentation, tp)
            position_results.append(tp_dataframe)
        store_position(position_results, self.tiler.positions.index(
            position), store, position_name=position,
                       mother_assign=mother_assign,
                       tile_size=self.default_image_size)
        return

    def run(self, keys, store, verbose=False, **kwargs):
        for pos, tps in accumulate(keys):
            self.process_position(pos, tps, store, verbose, **kwargs)
        return keys
