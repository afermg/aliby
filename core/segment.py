"""Segment/segmented pipelines.
Includes splitting the image into traps/parts,
cell segmentation, nucleus segmentation."""
from skimage import feature
import numpy as np
import pandas as pd
from pathlib import Path

from core.traps import identify_trap_locations, get_trap_timelapse, \
    get_traps_timepoint, align_timelapse_images

trap_template_directory = Path(__file__).parent / 'trap_templates'
trap_template = np.load(trap_template_directory / 'trap_bg_1.npy')


def get_tile_shapes(x, tile_size, max_shape):
    half_size = tile_size // 2
    xmin = int(x[0] - half_size)
    ymin = max(0, int(x[1] - half_size))
    if xmin + tile_size > max_shape[0]:
        xmin = max_shape[0] - tile_size
    if ymin + tile_size > max_shape[1]:
        ymin = max_shape[1] - tile_size
    return xmin, xmin + tile_size, ymin, ymin + tile_size


class Tiler(object):
    """
    Pipeline element that takes raw data and finds and tracks traps.
    """
    def __init__(self, raw_expt):
        """
        :param raw_expt:
        """
        self.raw_expt = raw_expt
        self.rotation = None
        self.trap_locations = dict()
        self.cell_outlines = None
        self.compartment_outlines = None


        # Tile the current position
        self.trap_locations[self.current_position] = self.tile_timelapse(
            self.raw_expt.current_position)

    @property
    def n_traps(self):
        return self.trap_locations[self.current_position].shape[1]

    @property
    def n_timepoints(self):
        return self.trap_locations[self.current_position].shape[0]//2

    @property
    def positions(self):
        return self.raw_expt.positions

    @property
    def current_position(self):
        return str(self.raw_expt.current_position)

    @current_position.setter
    def current_position(self, position):
        self.raw_expt.current_position = position
        # Tile that position
        if self.current_position not in self.trap_locations.keys():
            self.trap_locations[self.current_position] = \
                self.tile_timelapse(self.raw_expt.current_position)

    @property
    def channels(self):
        return self.raw_expt.channels

    def get_channel_index(self, channel):
        return self.raw_expt.current_position.get_channel_index(channel)

    @staticmethod
    def tile_timelapse(timelapse, channel: int =0) -> pd.DataFrame:
        """
        Finds the tile positions in a time lapse (including drifts).
        :param timelapse: The Timelapse object holding the raw data
        :param channel: Which channel to use for tiling, default=0
        :return: A Dataframe containing trap centers as data, rows are
        timepoints and columns are traps.
        """
        drifts = align_timelapse_images(timelapse, channel=channel)
        # Find traps in the first image
        trap_locations = {0: identify_trap_locations(
            np.squeeze(timelapse[channel, 0, :, :, 0]), trap_template)}
        for i in range(len(drifts)):
            trap_locations[i] = trap_locations[0] \
                                - np.sum(drifts[:i, [1, 0]], axis=0)

        # Reorganize into a pandas dataframe
        trap_df = pd.concat([pd.DataFrame(x.T, index=['x', 'y'])
                             for x in trap_locations.values()],
                             keys=np.arange(len(trap_locations)),
                            names=['timepoint', 'coordinate'])
        return trap_df

    def get_trap_timelapse(self, trap_id, tile_size=96, channels=None, z=None):
        """
        Get a timelapse for a given trap by specifying the trap_id
        :param trap_id: An integer defining which trap to choose. Counted
        between 0 and Tiler.n_traps - 1
        :param tile_size: The size of the trap tile (centered around the
        trap as much as possible, edge cases exist)
        :param channels: Which channels to fetch, indexed from 0.
        If None, defaults to [0]
        :param z: Which z_stacks to fetch, indexed from 0.
        If None, defaults to [0].
        :return: A numpy array with the timelapse in (C,T,X,Y,Z) order
        """
        return get_trap_timelapse(self.raw_expt,
                                  self.trap_locations[self.current_position],
                                  trap_id, tile_size=tile_size,
                                  channels=channels, z=z)

    def get_traps_timepoint(self, tp, tile_size=96, channels=None, z=None):
        """
        Get all the traps from a given timepoint
        :param tp:
        :param tile_size:
        :param channels:
        :param z:
        :return: A numpy array with the traps in the (trap, C, T, X, Y,
        Z) order
        """
        return get_traps_timepoint(self.raw_expt,
                                   self.trap_locations[self.current_position],
                                   tp, tile_size=tile_size, channels=channels,
                                   z=z)

    def fit(self, store_path):
        """
        Populates the HDF5 store with the trap annotations for each position.
        :param store_path:
        :returns: The store object
        """
        store = pd.HDFStore(store_path)

        expt_root = self.raw_expt.exptID
        for pos in self.positions:
            self.current_position = pos #So tiling occurs
            store_key = '/'.join([expt_root, pos, 'trap_locations'])
            store.append(store_key, self.trap_locations[pos])
        return

    def fit_to_pipe(self, pipe, split_results=True):
        """
        Takes a pipe of core.timelapse.Timelapse objects and their
        corresponding experiment ID and in return yields Results objects
        with trap location results.

        :param pipe: Input generator of Timelapse objects.
        :param split_results: Determines whether the output Results are
        split by trap or not.
        :returns:
        """

        return

