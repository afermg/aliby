import itertools
import logging
import numpy as np
from pathlib import Path

import imageio

logger = logging.getLogger(__name__)


def parse_local_fs(pos_dir):
    """
    Local file structure:
    - pos_dir
        -- exptID_{timepointID}_{ChannelID}_{z_position_id}.png

    :param pos_dir:
    :return: Image_mapper
    """
    pos_dir = Path(pos_dir)

    img_mapper = dict()
    img_list = [img for img in pos_dir.iterdir()]

    def channel_idx(img_name):
        return img_name.stem.split('_')[-2]

    def tp_idx(img_name):
        return img_name.stem.split('_')[-3]

    def z_idx(img_name):
        return img_name.stem.split('_')[-1]

    for channel, group in itertools.groupby(sorted(img_list, key=channel_idx),
                                            key=channel_idx):
        img_mapper[channel] = [{i: item for i, item in enumerate(sorted(grp,
                                                                key=z_idx))}
                               for _, grp in
                               itertools.groupby(sorted(group, key=tp_idx),
                                                 key=tp_idx)]
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


class TimelapseLocal(Timelapse):
    """
    Local file structure:

    - pos_dir
        -- exptID_{timepointID}_{ChannelID}_{z_position_id}.png

    """

    def __init__(self, position, root_dir):
        super(TimelapseLocal, self).__init__()
        self.pos_dir = root_dir / position
        assert self.pos_dir.exists()
        self._id = position
        self._name = position
        self.image_mapper = parse_local_fs(self.pos_dir)
        self._init_metadata()

    def _init_metadata(self):
        """
        Initialize the metadata based on the image_mapper

        :return:
        """
        self._channels = list(self.image_mapper.keys())
        self._size_c = len(self._channels)
        self._size_t = min([len(item) for item in self.image_mapper.values()])
        self._size_z = max([len(item) for item in itertools.chain(
                            *self.image_mapper.values())])
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
            names = [self.image_mapper[self.channels[ch_id]][t].get(i, None)
                   for i in z_positions]
            res = [imageio.imread(name) if name is not None else default
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

