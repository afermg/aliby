import itertools
import json
import numpy as np
import pandas as pd
import re
import requests
import tables
from requests.exceptions import Timeout, HTTPError
from requests_toolbelt.multipart.encoder import MultipartEncoder

from core.traps import get_trap_timelapse, get_traps_timepoint
from core.utils import Cache


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
    fields = {"dims": json.dumps(dims),
              "bitdepth": json.dumps(bit_depth),
              "img": img.tostring(order='F')}
    # Add optional arguments
    fields.update({kw: json.dumps(v) for kw, v in kwargs.items()})
    m = MultipartEncoder(
        fields=fields,
        boundary="----BabyFormBoundary"
    )
    return m


class BabyClient:
    def __init__(self, raw_expt, url='http://localhost:5101', **kwargs):
        self.raw_expt = raw_expt
        self.url = url
        self._config = kwargs
        r_model_sets = requests.get(self.url + '/models')
        self.valid_models = r_model_sets.json()
        self._model_set = choose_model_from_params(self.valid_models,
                                                   **self.config)
        self.sessions = Cache(load_fn=lambda _: self.get_new_session())
        self.processing = []
        self._store = None

    @property
    def store(self):
        return self._store

    @store.setter
    def store(self, store):
        if self._store is None:
            self._store = store
        else:
            raise ValueError("Store attribute can only be set once.")

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

    def process_position(self, prompt: str):
        # The prompt received by baby is the position
        try:
            trap_locations = self.store[prompt + '/trap_locations']
            for timepoint_id in trap_locations.index:
                # Finish processing previously queued images
                self.flush_processing()
                self.process_timepoint(prompt, timepoint_id, trap_locations)
        except KeyError as e:
            # TODO log that this will not be processed
            raise e
        # Flush all processing before moving to the next position
        while len(self.processing) > 0:
            self.flush_processing()

    def process_timepoint(self, prompt, timepoint, trap_locations,
                         tile_size=81, z=[0,1,2,3,4]):
        traps = get_traps_timepoint(self.raw_expt, trap_locations,
                                    timepoint, tile_size=tile_size, z=z)
        traps = np.squeeze(traps)
        timepoint_key = prompt + f'time{timepoint}'
        session_id = self.sessions[timepoint_key]
        self.queue_image(traps, session_id)
        self.processing.append(timepoint_key)

    # Todo: defined based on the model configuration what z should be
    def process_trap(self, prompt, trap_id, trap_locations, tile_size=81,
                     z=[0, 1, 2, 3, 4]):
        tile = get_trap_timelapse(self.raw_expt, trap_locations, trap_id,
                                  tile_size=tile_size, z=z)
        tile = np.squeeze(tile)
        # try:
        #     self.store._handle.create_group(prompt, f'trap{trap_id}')
        # except tables.exceptions.NodeError as e:
        #     pass
        # Get the corresponding session
        trap_key = prompt + f'/trap{trap_id}'
        session_id = self.sessions[trap_key]
        batches = np.array_split(tile, 8, axis=0)
        for batch in batches:
            self.queue_image(batch, session_id)
            self.processing.append(trap_key)
            self.flush_processing()

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

    def flush_processing(self):
        """
        Get the results of previously queued images.
        :return:
        """
        for time_point in self.processing:
            try:
                result = self.get_segmentation(self.sessions[time_point])
                segmentation = self.format_seg_result(result)
                for i, seg in segmentation.items():
                    cell_key = time_point + f'/cell{i}'
                    try:
                        self.store.append(cell_key, seg)
                    except Exception as e:
                        print(seg)
                        raise e
                self.processing.remove(time_point)
            except Timeout:
                continue
            except HTTPError:
                continue
            except TypeError as e:
                raise e
            except KeyError as e:
                print(self.store.keys())
                raise e

    def run(self, keys, store):
        if self.store is None:
            self.store = store
        for prompt in keys:
            self.process_position(prompt)
        return keys
