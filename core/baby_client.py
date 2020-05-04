import json
import re
import requests
from requests.exceptions import Timeout, HTTPError
from requests_toolbelt.multipart.encoder import MultipartEncoder


class BabyNoMatches(Exception):
    pass


class BabyNoSilent(Exception):
    pass


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
    def __init__(self, url='http://localhost:5101', **kwargs):
        self.url = url
        self._session_id = ""
        self._config = kwargs
        r_model_sets = requests.get(self.url + '/models')
        self.valid_models = r_model_sets.json()
        self._model_set = choose_model_from_params(self.valid_models,
                                                   **self.config)

    @property
    def session_id(self):
        return self._session_id

    @session_id.setter
    def session_id(self, session_id):
        if self._session_id is not "" and session_id is not "":
            raise BabyNoSilent("Can only silently set a session id to/from "
                               "None")
        else:
            self._session_id = session_id

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
            self.session_id = r_session.json()["sessionid"]
        except KeyError as e:
            raise e
        except HTTPError as e:
            raise e

    def queue_image(self, img, **kwargs):
        # TODO validate image type?
        # TODO character encoding options?
        bit_depth = img.dtype.itemsize * 8  # bit depth =  byte_size * 8
        data = create_request(img.shape, bit_depth, img, **kwargs)
        status = requests.post(self.url +
                               '/segment?sessionid={}'.format(self.session_id),
                               data=data,
                               headers={'Content-Type': data.content_type})
        status.raise_for_status()
        return status

    def get_segmentation(self):
        try:
            seg_response = requests.get(
                self.url + '/segment?sessionid={}'.format(self.session_id),
                timeout=120)
            seg_response.raise_for_status()
            result = seg_response.json()
        except Timeout as e:
            raise e
        except HTTPError as e:
            raise e
        return result
