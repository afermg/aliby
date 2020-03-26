import abc
import itertools
import numpy as np
import imageio
from operator import itemgetter

import logging

logger = logging.getLogger(__name__)

class Timelapse:
    """
    Timelapse class contains the specifics of one position.
    """
    __metaclass__ = abc.ABCMeta

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
            else: # both s.start and s.stop are not None
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

    @abc.abstractproperty
    def id(self):
        return

    @abc.abstractproperty
    def name(self):
        return

    @abc.abstractproperty
    def size_z(self):
        return

    @abc.abstractproperty
    def size_c(self):
        return

    @abc.abstractproperty
    def size_t(self):
        return

    @abc.abstractproperty
    def size_x(self):
        return

    @abc.abstractproperty
    def size_y(self):
        return

    @abc.abstractproperty
    def channels(self):
        return

    @abc.abstractmethod
    def get_hypercube(self, x=None, y=None,
                      z_positions=None, channels=None,
                      timepoints=None):
        return

    @abc.abstractmethod
    def get_channel_index(self, channel):
        return


class TimelapseOMERO:
    """
    Connected to an Image object which handles database I/O.
    """

    def __init__(self, image):
        self.image = image
        # Pre-load pixels
        self.pixels = self.image.getPrimaryPixels()
        self._size_z = None
        self._size_c = None
        self._size_t = None
        self._channels = None
        self._id = None
        self._name = None

    @property
    def id(self):
        if self._id is None:
            self._id = self.image.getId()
        return self._id

    @property
    def name(self):
        if self._name is None:
            self._name = self.image.getName()
        return self._name

    @property
    def size_z(self):
        if self._size_z is None:
            self._size_z = self.image.getSizeZ()
        return self._size_z

    @property
    def size_c(self):
        if self._size_c is None:
            self._size_c = self.image.getSizeC()
        return self._size_c

    @property
    def size_t(self):
        if self._size_t is None:
            self._size_t = self.image.getSizeT()
        return self._size_t

    @property
    def size_x(self):
        if self._size_x is None:
            self._size_x = self.image.getSizeX()
        return self._size_x

    @property
    def size_y(self):
        if self._size_y is None:
            self._size_y = self.image.getSizeY()
        return self._size_y

    @property
    def channels(self):
        if self._channels is None:
            self._channels = self.image.getChannelLabels()
        return self._channels

    def get_channel_index(self, channel):
        return self.channels.index(channel)

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
            tile = (xmin, ymin, xmax - xmin, ymax-ymin)

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

    - pos001
        -- exptID_{timepointID}_{ChannelID}_{z_position_id}.png

    """

    def __init__(self, position, root_dir, metadata):
        self.pos_dir = root_dir / position
        assert self.pos_dir.exists()
        self.image_mapper = None
        self._size_z = None
        self._size_c = None
        self._size_t = None
        self._size_x = None
        self._size_y = None
        self._channels = None
        self._id = position
        self._name = position
        self.parse_metadata(metadata)
        self.parse_file_structure(self.pos_dir)

    def parse_metadata(self, metadata):
        self._size_z = int(metadata.zsections.sections.values[0])
        self._channels = metadata.channels.names.values
        self._size_c = len(self._channels)
        self._size_t = int(metadata.times['ntimepoints'])

    def parse_file_structure(self, pos_dir):
        # Only needs to check the results of parse_metadata
        img_mapper = dict()

        for channel in self.channels:
            img_list = [img for img in self.pos_dir.iterdir()
                        if channel in img.name]
            # Check that the metadata was correct/we are not missing any images
            assert len(img_list) != 0, "Channel {} not available, incorrect " \
                                       "metadata"
            img_mapper[channel] = [sorted(list((group)), key=lambda item:
            item.stem.split('_')[-1])
                                   for _, group in
                                   itertools.groupby(sorted(img_list),
                                                     key=lambda img:
                                                     img.stem.split('_')[-3])]

        for ch, item in img_mapper.items():
            if len(item) != int(self.size_t):
                logger.warning("Not enough timepoints in position {}, "
                               "channel {}: {} out of {}".format(self.id, ch,
                                                                 len(item),
                                                                 self.size_t))
        self._size_t = min([len(item) for item in img_mapper.values()])

        for ix, (ch, im_list) in enumerate(img_mapper.items()):
            for item in im_list:
                if len(item) != int(self.size_z):
                    logger.warning("Not enough z-stacks for position {}, " \
                                   "channel {}, tp {}; {} out of " \
                                   "{}".format(self.id, ch, ix, len(item),
                                               self.size_z))
        self._size_z = min([len(item) for item in im_list])

        self.image_mapper = img_mapper

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
        if self._size_x is None:
            single_img = self.get_hypercube(x=None, y=None,
                                            z_positions=None, channels=[0],
                                            timepoints=[0])
            self._size_x = single_img.shape[2]
            self._size_y = single_img.shape[3]
        return self._size_x

    @property
    def size_y(self):
        return self._size_y

    @property
    def channels(self):
        return self._channels

    def get_channel_index(self, channel):
        return self.channels.index(channel)

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

        z_pos_getter = itemgetter(*z_positions)
        # nested list of images in C, T, X, Y, Z order
        ctxyz = []
        for ch_id in channels:
            txyz = []
            for t in timepoints:
                xyz = map(imageio.imread, z_pos_getter(self.image_mapper[
                                                    self.channels[ch_id]][t]))
                txyz.append(np.dstack(xyz)[xmin:xmax, ymin:ymax])
            ctxyz.append(np.stack(txyz))
        return np.stack(ctxyz)


Timelapse.register(TimelapseOMERO)
Timelapse.register(TimelapseLocal)
