import itertools
import logging
import numpy as np
import pandas as pd
from pathlib import Path

from tqdm import tqdm
import cv2
from core.utils import Cache

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
        return img_name.stem.split('_')[-2]

    def tp_idx(img_name):
        return int(img_name.stem.split('_')[-3]) - 1

    def z_idx(img_name):
        return img_name.stem.split('_')[-1]

    if tp is not None:
        img_list = [img for img in pos_dir.iterdir() if tp_idx(img) in tp]
    else:
        img_list = [img for img in pos_dir.iterdir()]

    for tp, group in itertools.groupby(sorted(img_list, key=tp_idx),
                                       key=tp_idx):
        img_mapper[int(tp)] = {channel: {i: item
                                         for i, item in
                                         enumerate(sorted(grp, key=z_idx))}
                               for channel, grp in
                               itertools.groupby(
                                   sorted(group, key=channel_idx),
                                   key=channel_idx)
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

    def __repr__(self):
        return self.name

    def __getitem__(self, item):
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
            if isinstance(subitem, int):
                res = [subitem]
            elif isinstance(subitem, list) or isinstance(subitem, tuple):
                res = list(subitem)
            elif isinstance(subitem, slice):
                res = parse_slice(subitem)

            if kw in ['x', 'y']:
                # Need exactly two values
                if res is not None:
                    if len(res) < 2:
                        # An int was passed, assume it was
                        res = [res[0], self.size_x]
                    elif len(res) > 2:
                        res = [res[0], res[-1] + 1]
            return res

        if isinstance(item, int):
            return self.get_hypercube(x=None, y=None, z_positions=None,
                                      channels=[item], timepoints=None)
        elif isinstance(item, slice):
            return self.get_hypercube(channels=parse_slice(item))
        keywords = ['channels', 'timepoints', 'x', 'y', 'z_positions']
        kwargs = dict()
        for kw, subitem in zip(keywords, item):
            kwargs[kw] = parse_subitem(subitem, kw)
        return self.get_hypercube(**kwargs)

    @property
    def shape(self):
        return (self.size_c, self.size_t,
                self.size_x, self.size_y, self.size_z)

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


class TimelapseOMERO(Timelapse):
    """
    Connected to an Image object which handles database I/O.
    """

    def __init__(self, image):
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

    def get_hypercube(self, x=None, y=None,
                      z_positions=None, channels=None,
                      timepoints=None):
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

        zcttile_list = [(z, c, t, tile) for z, c, t in
                        itertools.product(z_positions, channels, timepoints)]
        planes = list(self.pixels.getTiles(zcttile_list))
        order = (len(z_positions), len(channels), len(timepoints),
                 planes[0].shape[-1], planes[0].shape[-2])
        result = np.stack([x for x in planes]).reshape(order)
        # Set to C, T, X, Y, Z order
        return np.moveaxis(result, 0, -1)

    def cache_set(self, save_dir, timepoints, expt_name, quiet=True):
        pos_dir = save_dir / self.name
        if not pos_dir.exists():
            pos_dir.mkdir()
        for tp in tqdm(timepoints, desc=self.name):
            for channel in tqdm(self.channels, disable=quiet):
                for z_pos in tqdm(range(self.size_z), disable=quiet):
                    ch_id = self.get_channel_index(channel)
                    image = self.get_hypercube(x=None, y=None,
                                               channels=[ch_id],
                                               z_positions=[z_pos],
                                               timepoints=[tp])
                    im_name = "{}_{:06d}_{}_{:03d}.png".format(expt_name,
                                                               tp + 1,
                                                               channel,
                                                               z_pos + 1)
                    cv2.imwrite(str(pos_dir / im_name), np.squeeze(image))
        # TODO update positions table to get the number of timepoints?
        return list(itertools.product([self.name], timepoints))

    def run(self, keys, positions, expt_name="", save_dir="./", **kwargs):
        """
        Parse file structure and get images for the timepoints in keys.
        """
        save_dir = Path(save_dir)
        if keys is None:
            return None
        n_timepoints = positions[self.name].loc['n_timepoints']
        start_tp = min(n_timepoints, min(keys))
        end_tp = min(self.size_t, max(keys))

        timepoints = list(range(start_tp, end_tp + 1))
        cached = []
        if len(timepoints) > 0 and n_timepoints <= max(timepoints):
            try:
                cached = self.cache_set(save_dir, timepoints, expt_name)
                positions[self.name].loc['n_timepoints'] = max(timepoints)
            finally:
                # Write the new pos_df to file?
                pass
        return cached


class TimelapseLocal(Timelapse):
    def __init__(self, position, root_dir, finished=False):
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
        self.image_cache = Cache()
        if finished:
            self.image_mapper = parse_local_fs(self.pos_dir)
            self._update_metadata()
        else:
            self.image_mapper = dict()

    def _update_metadata(self):
        self._size_t = len(self.image_mapper)
        # Todo: if cy5 is the first one it causes issues with getting x, y
        #   hence the sorted but it's not very robust
        self._channels = sorted(list(set.union(*[set(tp.keys())
                                                 for tp in
                                                 self.image_mapper.values()])))
        self._size_c = len(self._channels)
        # Todo: refactor so we don't rely on there being any images at all
        self._size_z = max([len(self.image_mapper[0][ch])
                            for ch in self._channels])
        single_img = self.get_hypercube(x=None, y=None,
                                        z_positions=None, channels=[0],
                                        timepoints=[0])
        self._size_x = single_img.shape[2]
        self._size_y = single_img.shape[3]

    def get_hypercube(self, x=None, y=None,
                      z_positions=None, channels=None,
                      timepoints=None):
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
            names = [self.image_mapper[t][self.channels[ch_id]].get(i, None)
                     for i in z_positions]
            res = [self.image_cache[name] if name is not None else default
                   for name in names]
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

    def run(self, keys, **kwargs):
        """
        Parse file structure and get images for the timepoints in keys.
        """
        if keys is None:
            return None
        elif isinstance(keys, int):
            keys = [keys]
        self.image_mapper.update(parse_local_fs(self.pos_dir, tp=keys))
        self._update_metadata()
        return keys
