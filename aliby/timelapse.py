import itertools
import logging

import h5py
import numpy as np
from pathlib import Path

from tqdm import tqdm
import cv2

from aliby.io.matlab import matObject
from agora.io.utils import Cache, imread, get_store_path

logger = logging.getLogger(__name__)


def parse_local_fs(pos_dir, tp=None):
    """
    Local file structure:
    - pos_dir
        -- exptID_{timepointID}_{ChannelID}_{z_position_id}.png

    :param pos_dirs:
    :return: Image_mapper
    """
    pos_dir = Path(pos_dir)

    img_mapper = dict()

    def channel_idx(img_name):
        return img_name.stem.split("_")[-2]

    def tp_idx(img_name):
        return int(img_name.stem.split("_")[-3]) - 1

    def z_idx(img_name):
        return img_name.stem.split("_")[-1]

    if tp is not None:
        img_list = [img for img in pos_dir.iterdir() if tp_idx(img) in tp]
    else:
        img_list = [img for img in pos_dir.iterdir()]

    for tp, group in itertools.groupby(sorted(img_list, key=tp_idx), key=tp_idx):
        img_mapper[int(tp)] = {
            channel: {i: item for i, item in enumerate(sorted(grp, key=z_idx))}
            for channel, grp in itertools.groupby(
                sorted(group, key=channel_idx), key=channel_idx
            )
        }
    return img_mapper


class Timelapse:
    """
    Timelapse class contains the specifics of one position.
    """

    def __init__(self):
        self._id = None
        self._name = None
        self._channels = []
        self._size_c = 0
        self._size_t = 0
        self._size_x = 0
        self._size_y = 0
        self._size_z = 0
        self.image_cache = None
        self.annotation = None

    def __repr__(self):
        return self.name

    def full_mask(self):
        return np.full(self.shape, False)

    def __getitem__(self, item):
        cached = self.image_cache[item]
        # Check if there are missing values, if so reload
        # TODO only reload missing
        mask = np.isnan(cached)
        if np.any(mask):
            full = self.load_fn(item)
            shape = self.image_cache[
                item
            ].shape  # TODO speed this up by  recognising the shape from the item
            self.image_cache[item] = np.reshape(full, shape)
            return full
        return cached

    def get_hypercube(self):
        pass

    def load_fn(self, item):
        """
        The hypercube is ordered as: C, T, X, Y, Z
        :param item:
        :return:
        """

        def parse_slice(s):
            step = s.step if s.step is not None else 1
            if s.start is None and s.stop is None:
                return None
            elif s.start is None and s.stop is not None:
                return range(0, s.stop, step)
            elif s.start is not None and s.stop is None:
                return [s.start]
            else:  # both s.start and s.stop are not None
                return range(s.start, s.stop, step)

        def parse_subitem(subitem, kw):
            if isinstance(subitem, (int, float)):
                res = [int(subitem)]
            elif isinstance(subitem, list) or isinstance(subitem, tuple):
                res = list(subitem)
            elif isinstance(subitem, slice):
                res = parse_slice(subitem)
            else:
                res = subitem
                # raise ValueError(f"Cannot parse slice {kw}: {subitem}")

            if kw in ["x", "y"]:
                # Need exactly two values
                if res is not None:
                    if len(res) < 2:
                        # An int was passed, assume it was
                        res = [res[0], self.size_x]
                    elif len(res) > 2:
                        res = [res[0], res[-1] + 1]
            return res

        if isinstance(item, int):
            return self.get_hypercube(
                x=None, y=None, z_positions=None, channels=[item], timepoints=None
            )
        elif isinstance(item, slice):
            return self.get_hypercube(channels=parse_slice(item))
        keywords = ["channels", "timepoints", "x", "y", "z_positions"]
        kwargs = dict()
        for kw, subitem in zip(keywords, item):
            kwargs[kw] = parse_subitem(subitem, kw)
        return self.get_hypercube(**kwargs)

    @property
    def shape(self):
        return (self.size_c, self.size_t, self.size_x, self.size_y, self.size_z)

    @property
    def id(self):
        return self._id

    @property
    def name(self):
        return self._name

    @property
    def size_z(self):
        return self._size_z

    @property
    def size_c(self):
        return self._size_c

    @property
    def size_t(self):
        return self._size_t

    @property
    def size_x(self):
        return self._size_x

    @property
    def size_y(self):
        return self._size_y

    @property
    def channels(self):
        return self._channels

    def get_channel_index(self, channel):
        return self.channels.index(channel)


def load_annotation(filepath: Path):
    try:
        return matObject(filepath)
    except Exception as e:
        raise (
            "Could not load annotation file. \n"
            "Non MATLAB files currently unsupported"
        ) from e


class TimelapseOMERO(Timelapse):
    """
    Connected to an Image object which handles database I/O.
    """

    def __init__(self, image, annotation, cache, **kwargs):
        super(TimelapseOMERO, self).__init__()
        self.image = image
        # Pre-load pixels
        self.pixels = self.image.getPrimaryPixels()
        self._id = self.image.getId()
        self._name = self.image.getName()
        self._size_x = self.image.getSizeX()
        self._size_y = self.image.getSizeY()
        self._size_z = self.image.getSizeZ()
        self._size_c = self.image.getSizeC()
        self._size_t = self.image.getSizeT()
        self._channels = self.image.getChannelLabels()
        # Check whether there are file annotations for this position
        if annotation is not None:
            self.annotation = load_annotation(annotation)
        # Get an HDF5 dataset to use as a cache.
        compression = kwargs.get("compression", None)
        self.image_cache = cache.require_dataset(
            self.name,
            self.shape,
            dtype=np.float16,
            fillvalue=np.nan,
            compression=compression,
        )

    def get_hypercube(
        self, x=None, y=None, z_positions=None, channels=None, timepoints=None
    ):
        if x is None and y is None:
            tile = None  # Get full plane
        elif x is None:
            ymin, ymax = y
            tile = (None, ymin, None, ymax - ymin)
        elif y is None:
            xmin, xmax = x
            tile = (xmin, None, xmax - xmin, None)
        else:
            xmin, xmax = x
            ymin, ymax = y
            tile = (xmin, ymin, xmax - xmin, ymax - ymin)

        if z_positions is None:
            z_positions = range(self.size_z)
        if channels is None:
            channels = range(self.size_c)
        if timepoints is None:
            timepoints = range(self.size_t)

        z_positions = z_positions or [0]
        channels = channels or [0]
        timepoints = timepoints or [0]

        zcttile_list = [
            (z, c, t, tile)
            for z, c, t in itertools.product(z_positions, channels, timepoints)
        ]
        planes = list(self.pixels.getTiles(zcttile_list))
        order = (
            len(z_positions),
            len(channels),
            len(timepoints),
            planes[0].shape[-2],
            planes[0].shape[-1],
        )
        result = np.stack([x for x in planes]).reshape(order)
        # Set to C, T, X, Y, Z order
        result = np.moveaxis(result, -1, -2)
        return np.moveaxis(result, 0, -1)

    def cache_set(self, save_dir, timepoints, expt_name, quiet=True):
        # TODO deprecate when this is default
        pos_dir = save_dir / self.name
        if not pos_dir.exists():
            pos_dir.mkdir()
        for tp in tqdm(timepoints, desc=self.name):
            for channel in tqdm(self.channels, disable=quiet):
                for z_pos in tqdm(range(self.size_z), disable=quiet):
                    ch_id = self.get_channel_index(channel)
                    image = self.get_hypercube(
                        x=None,
                        y=None,
                        channels=[ch_id],
                        z_positions=[z_pos],
                        timepoints=[tp],
                    )
                    im_name = "{}_{:06d}_{}_{:03d}.png".format(
                        expt_name, tp + 1, channel, z_pos + 1
                    )
                    cv2.imwrite(str(pos_dir / im_name), np.squeeze(image))
        # TODO update positions table to get the number of timepoints?
        return list(itertools.product([self.name], timepoints))

    def run(self, keys, store, save_dir="./", **kwargs):
        """
        Parse file structure and get images for the timepoints in keys.
        """
        save_dir = Path(save_dir)
        if keys is None:
            # TODO save final metadata
            return None
        store = save_dir / store
        # A position specific store
        store = store.with_name(self.name + store.name)
        # Create store if it does not exist
        if not store.exists():
            # The first run, add metadata to the store
            with h5py.File(store, "w") as pos_store:
                # TODO Add metadata to the store.
                pass
        # TODO check how sensible the keys are with what is available
        #   if some of the keys don't make sense, log a warning and remove
        #   them so that the next steps of the pipeline make sense
        return keys

    def clear_cache(self):
        self.image_cache.clear()


class TimelapseLocal(Timelapse):
    def __init__(
        self, position, root_dir, finished=True, annotation=None, cache=None, **kwargs
    ):
        """
        Linked to a local directory containing the images for one position
        in an experiment.
        Can be a still running experiment or a finished one.

        :param position: Name of the position
        :param root_dir: Root directory
        :param finished: Whether the experiment has finished running or the
        class will be used as part of a pipeline, mostly with calls to `run`
        """
        super(TimelapseLocal, self).__init__()
        self.pos_dir = Path(root_dir) / position
        assert self.pos_dir.exists()
        self._id = position
        self._name = position
        if finished:
            self.image_mapper = parse_local_fs(self.pos_dir)
            self._update_metadata()
        else:
            self.image_mapper = dict()
        self.annotation = None
        # Check whether there are file annotations for this position
        if annotation is not None:
            self.annotation = load_annotation(annotation)
        compression = kwargs.get("compression", None)
        self.image_cache = cache.require_dataset(
            self.name,
            self.shape,
            dtype=np.float16,
            fillvalue=np.nan,
            compression=compression,
        )

    def _update_metadata(self):
        self._size_t = len(self.image_mapper)
        # Todo: if cy5 is the first one it causes issues with getting x, y
        #   hence the sorted but it's not very robust
        self._channels = sorted(
            list(set.union(*[set(tp.keys()) for tp in self.image_mapper.values()]))
        )
        self._size_c = len(self._channels)
        # Todo: refactor so we don't rely on there being any images at all
        self._size_z = max([len(self.image_mapper[0][ch]) for ch in self._channels])
        single_img = self.get_hypercube(
            x=None, y=None, z_positions=None, channels=[0], timepoints=[0]
        )
        self._size_x = single_img.shape[2]
        self._size_y = single_img.shape[3]

    def get_hypercube(
        self, x=None, y=None, z_positions=None, channels=None, timepoints=None
    ):
        xmin, xmax = x if x is not None else (None, None)
        ymin, ymax = y if y is not None else (None, None)

        if z_positions is None:
            z_positions = range(self.size_z)
        if channels is None:
            channels = range(self.size_c)
        if timepoints is None:
            timepoints = range(self.size_t)

        def z_pos_getter(z_positions, ch_id, t):
            default = np.zeros((self.size_x, self.size_y))
            names = [
                self.image_mapper[t][self.channels[ch_id]].get(i, None)
                for i in z_positions
            ]
            res = [imread(name) if name is not None else default for name in names]
            return res

        # nested list of images in C, T, X, Y, Z order
        ctxyz = []
        for ch_id in channels:
            txyz = []
            for t in timepoints:
                xyz = z_pos_getter(z_positions, ch_id, t)
                txyz.append(np.dstack(list(xyz))[xmin:xmax, ymin:ymax])
            ctxyz.append(np.stack(txyz))
        return np.stack(ctxyz)

    def clear_cache(self):
        self.image_cache.clear()

    def run(self, keys, store, save_dir="./", **kwargs):
        """
        Parse file structure and get images for the time points in keys.
        """
        if keys is None:
            return None
        elif isinstance(keys, int):
            keys = [keys]
        self.image_mapper.update(parse_local_fs(self.pos_dir, tp=keys))
        self._update_metadata()
        # Create store if it does not exist
        store = get_store_path(save_dir, store, self.name)
        if not store.exists():
            # The first run, add metadata to the store
            with h5py.File(store, "w") as pos_store:
                # TODO Add metadata to the store.
                pass
        # TODO check how sensible the keys are with what is available
        #   if some of the keys don't make sense, log a warning and remove
        #   them so that the next steps of the pipeline make sense
        return keys
