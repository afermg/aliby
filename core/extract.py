"""
A module to extract data from a processed experiment.
"""
import h5py
import numpy as np
from tqdm import tqdm

from core.io.matlab import matObject
from growth_rate.estimate_gr import estimate_gr


class Extracted:
    # TODO write the filtering functions.
    def __init__(self):
        self.volume = None
        self._keep = None

    def filter(self, filename=None, **kwargs):
        """
        1. Filter out small non-growing tracks. This means:
            a. the cell size never reaches beyond a certain size-threshold
            volume_thresh or
            b. the cell's volume doesn't increase by at least a minimum
            amount over its lifetime
        2. Join daughter tracks that are contiguous and within a volume
           threshold of each other
        3. Discard tracks that are shorter than a threshold number of
           timepoints

        This function is used to fix tracking/bud errors in post-processing.
        The parameters define the thresholds used to determine which cells are
        discarded.
        FIXME Ideally we get to a point where this is no longer needed.
        :return:
        """
        #self.join_tracks()
        filter_out = self.filter_size(**kwargs)
        filter_out += self.filter_lifespan(**kwargs)
        # TODO save data or just filtering parameters?
        #self.to_hdf(filename)
        self.keep = ~filter_out

    def filter_size(self, volume_thresh=7, growth_thresh=10, **kwargs):
        """Filter out small and non-growing cells.
        :param volume_thresh: Size threshold for small cells
        :param growth_thresh: Size difference threshold for non-growing cells
        """
        filter_out = np.where(np.max(self.volume, axis=1) < volume_thresh,
                              True, False)
        growth = [v[v > 0] for v in self.volume]
        growth = np.array([v[-1] - v[0] if len(v) > 0 else 0 for v in growth])
        filter_out += np.where(growth < growth_thresh, True, False)
        return filter_out

    def filter_lifespan(self, min_time=5, **kwargs):
        """Remove daughter cells that have a small life span.

        :param min_time: The minimum life span, under which cells are removed.
        """
        # TODO What if there are nan values?
        filter_out = np.where(np.count_nonzero(self.volume, axis=1) <
                              min_time, True, False)
        return filter_out

    def join_tracks(self, threshold=7):
        """ Join contiguous tracks that are within a certain volume
        threshold of each other.

        :param threshold: Maximum volume difference to join contiguous tracks.
        :return:
        """
        # For all pairs of cells
        #
        pass


class ExtractedHDF(Extracted):
    # TODO pull all the data out of the HFile and filter!
    def __init__(self, file):
        # We consider the data to be read-only
        self.hfile = h5py.File(file, 'r')


class ExtractedMat(Extracted):
    """ Pulls the extracted data out of the MATLAB cTimelapse file.

    This is mostly a convenience function in order to run the
    gaussian-processes growth-rate estimation
    """
    def __init__(self, file, debug=False):
        ct = matObject(file)
        self.debug = debug
        # Pre-computed data
        # TODO what if there is no timelapseTrapsOmero?
        self.metadata = ct['timelapseTrapsOmero']['metadata']
        self.extracted_data = ct['timelapseTrapsOmero']['extractedData']
        self.channels = ct['timelapseTrapsOmero']['extractionParameters'][
            'functionParameters']['channels'].tolist()
        self.time_settings = ct['timelapseTrapsOmero']['metadata']['acq'][
            'times']
        # Get filtering information
        n_cells = self.extracted_data['cellNum'][0].shape
        self.keep = np.full(n_cells, True)
        # Not yet computed data
        self._growth_rate = None
        self._daughter_index = None


    def get_channel_index(self, channel):
        """Get index of channel based on name. This only considers
        fluorescence channels."""
        return self.channels.index(channel)

    @property
    def trap_num(self):
        return self.extracted_data['trapNum'][0][self.keep]

    @property
    def cell_num(self):
        return self.extracted_data['cellNum'][0][self.keep]

    def identity(self, cell_idx):
        """Get the (position), trap, and cell label given a cell's global
        index."""
        # Todo include position when using full strain
        trap = self.trap_num[cell_idx]
        cell = self.cell_num[cell_idx]
        return trap, cell

    def global_index(self, trap_id, cell_label):
        """Get the global index of a cell given it's trap/cellNum
        combination."""
        candidates = np.where(np.logical_and(
                            (self.trap_num == trap_id), # +1?
                            (self.cell_num == cell_label)
                        ))[0]
        # TODO raise error if number of candidates != 1
        if len(candidates) == 1:
            return candidates[0]
        elif len(candidates) == 0:
            return -1
        else:
            raise(IndexError("No such cell/trap combination"))

    @property
    def daughter_label(self):
        """Returns the cell label of the daughters of each cell over the
        timelapse.

        0 corresponds to no daughter. This *not* the index of the daughter
        cell within the data. To get this, use daughter_index.
        """
        return self.extracted_data['daughterLabel'][0][self.keep]

    def _single_daughter_idx(self, mother_idx, daughter_labels):
        trap_id, _ = self.identity(mother_idx)
        daughter_index = [self.global_index(trap_id, cell_label) for
                          cell_label
                          in daughter_labels]
        return daughter_index

    @property
    def daughter_index(self):
        """Returns the global index of the daughters of each cell.

        This is different from the daughter label because it corresponds to
        the index of the daughter when counting all of the cells. This can
        be used to index within the data arrays.
        """
        if self._daughter_index is None:
            daughter_index = [self._single_daughter_idx(i, daughter_labels)
                          for i, daughter_labels in enumerate(
                                  self.daughter_label)]
            self._daughter_index = np.array(daughter_index)
        return self._daughter_index

    @property
    def births(self):
        return np.array(self.extracted_data['births'][0].todense())[self.keep]

    @property
    def volume(self):
        """Get the volume of all of the cells"""
        return np.array(self.extracted_data['volume'][0].todense())[self.keep]

    def _gr_estimation(self):
        dt = self.time_settings['interval'] / 360  # s to h conversion
        results = []
        for v in tqdm(self.volume):
            results.append(estimate_gr(v, dt))
        merged = {k: np.stack([x[k] for x in results]) for k in results[0]}
        self._gr_results = merged
        return

    @property
    def growth_rate(self):
        """Get the growth rate for all cells.

        Note that this uses the gaussian processes method of estimating
        growth rate by default. If there is no growth rate in the given file
        (usually the case for MATLAB), it needs to run estimation first.
        This can take a while.
        """
        # TODO cache the results of growth rate estimation.
        if self._gr_results is None:
            dt = self.time_settings['interval'] / 360  # s to h conversion
            self._growth_rate = [estimate_gr(v, dt) for v in self.volume]
        return self._gr_results['growth_rate']

    def _fluo_attribute(self, channel, attribute):
        channel_id = self.get_channel_index(channel)
        res = np.array(self.extracted_data[attribute][channel_id].todense())
        return res[self.keep]

    def protein_localisation(self, channel, method='nucEstConv'):
        """Returns protein localisation data for a given channel.

        Uses the 'nucEstConv' by default. Alternatives are 'smallPeakConv',
        'max5px', 'max2p5pc'
        """
        return self._fluo_attribute(channel, method)

    def background_fluo(self, channel):
        return self._fluo_attribute(channel, 'imBackground')

    def mean(self, channel):
        return self._fluo_attribute(channel, 'mean')

    def median(self, channel):
        return self._fluo_attribute(channel, 'median')

    def filter(self, filename=None):
        """Filters and saves results to and HDF5 file.

        This is necessary because we cannot write to the MATLAB file,
        so the results of the filter cannot be saved in the object.
        """
        super().filter(filename=filename)
        self._growth_rate = None  # reset growth rate so it is recomputed

    def to_hdf(self, filename):
        """Store the current results, including any filtering done, to a file.

        TODO Should we save filtered results or just re-do?
        :param filename:
        :return:
        """
        store = h5py.File(filename, 'w')
        try:
            # Store (some of the) metadata
            for meta in ['experiment', 'username', 'microscope',
                              'comments', 'project', 'date', 'posname',
                              'exptid']:
                store.attrs[meta] = self.metadata[meta]
            # TODO store timing information?
            store.attrs['time_interval'] = self.time_settings['interval']
            store.attrs['timepoints'] = self.time_settings['ntimepoints']
            store.attrs['total_duration'] = self.time_settings['totalduration']
            # Store volume, births, daughterLabel, trapNum, cellNum
            for key in ['volume', 'births', 'daughter_label', 'trap_num',
                        'cell_num']:
                store[key] = getattr(self, key)
            # Store growth rate results
            if self._gr_results:
                grp = store.create_group('gaussian_process')
                for key, val in self._gr_results.items():
                    grp[key] = val
            for channel in self.channels:
                # Create a group for each channel
                grp = store.create_group(channel)
                # Store protein_localisation, background fluorescence, mean, median
                # for each channel
                grp['protein_localisation'] = self.protein_localisation(channel)
                grp['background_fluo'] = self.background_fluo(channel)
                grp['mean'] = self.mean(channel)
                grp['median'] = self.median(channel)
        finally:
            store.close()

