import collections
import itertools
import json
import time
from typing import List, Iterable

import h5py
import numpy as np
import matplotlib.pyplot as plt
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

from core.traps import get_trap_timelapse, get_traps_timepoint
from core.utils import Cache, accumulate, PersistentDict
from database.records import CellInfo, Cell, Trap, Position


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
                         store):
        # The prompt received by baby is the position
        try:
            for timepoint_id in tps:
                # Finish processing previously queued images
                self.flush_processing(session, store)
                self.process_timepoint(prompt, timepoint_id)
        except KeyError as e:
            # TODO log that this will not be processed
            raise e
        # Flush all processing before moving to the next position
        while len(self.processing) > 0:
            self.flush_processing(session, store)

    def process_timepoint(self, pos, timepoint, tile_size=96):
        channel_idx = [self.tiler.get_channel_index(self.channel)]
        traps = self.tiler[pos].get_traps_timepoint(timepoint, channels=channel_idx,
                                                    tile_size=tile_size, z=self.z)
        traps = np.squeeze(traps)
        timepoint_key = (pos, timepoint)
        session_id = self.sessions[pos]
        print(traps.shape)
        self.queue_image(traps, session_id)
        self.processing.append(timepoint_key)

    def format_seg_result(self, result, time_origin=0, max_size=16):
        # Todo: update time origin at each step.
        for i, res in enumerate(result):
            res['timepoint'] = [i + time_origin] * len(res['cell_label'])
        merged = {k: list(itertools.chain.from_iterable(
            res[k] for res in result))
            for k in result[0].keys()}
        df = pd.DataFrame(merged)
        df.set_index('timepoint', inplace=True)
        if len(df) == 0:
            return dict()

        # Todo: split more systematically
        for k in ['angles', 'radii']:
            values = df[k].tolist()
            for val in values:
                val += [np.nan] * (max_size - len(val))
            try:
                df[[k + str(i) for i in range(max_size)]] = \
                    pd.DataFrame(values, index=df.index)
            except ValueError as e:
                print(k)
                print([len(val) for val in values])
                print(result)
                raise e
        df[['centrex', 'centrey']] = pd.DataFrame(df['centres'].tolist(),
                                                  index=df.index)
        df.drop(['centres', 'angles', 'radii'], axis=1, inplace=True)

        per_cell_dfs = {i: x for i, x in df.groupby(df['cell_label'])}
        return per_cell_dfs


    def flush_processing(self, session, store):
        """
        Get the results of previously queued images.
        :return:
        """
        for pos, tp in self.processing:
            try:
                result = self.get_segmentation(self.sessions[pos])
                store_result(result, pos, tp, session, store)
                session.commit()
                self.processing.remove((pos, tp))
            except Timeout:
                continue
            except HTTPError:
                continue
            except TypeError as e:
                raise e

    def run(self, keys, session=None, store='store.h5', **kwargs):
        # key are (pos, timepoint) tuples
        for pos, tps in accumulate(keys):
            self.process_position(pos, tps, session, store)
        return keys

def store_result(result, pos, tp, session, store):
    for ix, trap in enumerate(result):
        outputs = sorted(trap.keys())
        per_cell = zip(*[trap[k] for k in outputs])
        df = pd.DataFrame(per_cell, columns=outputs)
        db_trap = session.query(Trap).filter(Trap.number==ix)\
                                     .join(Trap.position)\
                                     .filter(Position.name==pos)\
                                     .one()
        trap_id = db_trap.id
        cells_info = []
        for _, cell in df.iterrows():
            try:
                db_cell = session.query(Cell)\
                             .filter_by(trap_id=trap_id,
                                        number=cell['cell_label'])\
                             .one()
            except NoResultFound:
                # Create a new cell
                db_cell = Cell(number=cell['cell_label'],
                               trap=db_trap)
                cells_info.append(db_cell)
                #session.add(db_cell)
            data_key = '/data/{}/trap_{}/cell_{}/time_{}'.format(pos, ix,
                                                  cell['cell_label'],
                                                  tp)
            cell_info = CellInfo(number=cell['cell_label'],
                                 x=cell['centres'][0],
                                 y=cell['centres'][1],
                                 t=tp,
                                 data=data_key,
                                 cell=db_cell
                                 )
            remaining_data = set(outputs) - {'cell_label', 'centres'}
            for key in remaining_data:
                item_key = data_key + '/' + key
                store[item_key] = cell[key]
            cells_info.append(cell_info)
            #session.add(cell_info)
        session.add_all(cells_info)

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
                logical_gpus = tf.config.experimental.list_logical_devices('GPU')
                print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
        # Getting the runner
        self.brain = BabyBrain(**self.valid_models[model_name],
                         session=self.tf_session, graph=self.tf_graph,
                         suppress_errors=True,
                         error_dump_dir=self.error_dump_dir)
    @property
    def config(self):
        return self._config

    def __init_properties(self, config):
        n_stacks = int(config.get('n_stacks', '5z').replace('z', ''))
        self.z = list(range(n_stacks))
        self.channel = self.tiler.get_channel_index(config.get('channel', 'Brightfield'))

    def session(self):
        return BabyCrawler(self.brain)

    def segment(self, img, sessionid, **kwargs):
        # Getting the result for a given image
        crawler = self.sessions[sessionid]
        pred = crawler.step(img, **kwargs)
        return pred

    def process_position(self, position, tps, db, store, verbose, **kwargs):
        self.tiler.current_position=position
        for tp in tqdm(tps, desc=position, disable=not verbose): 
            traps = np.squeeze(self.tiler.get_traps_timepoint(tp, channels=[self.channel], z=self.z))
            segmentation = self.segment(traps, position, **kwargs)
            store_result(segmentation, position, tp, db, store)
        db.commit()
        if isinstance(store, PersistentDict):
            store.sync()

    def run(self, keys, db, store, verbose=False, **kwargs):
        for pos, tps in accumulate(keys):
            self.process_position(pos, tps, db, store, verbose, **kwargs)
        return keys
