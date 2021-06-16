import h5py


class BridgeH5:
    def __init__(self, file):
        self._hdf = h5py.File(file, "r")

    def close(self):
        self._hdf.close()
