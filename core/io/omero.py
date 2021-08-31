from omero.gateway import BlitzGateway
from core.experiment import get_data_lazy
from core.cells import CellsHDF

class Argo:
    # TODO use the one in extraction?
    def __init__(self, host='islay.bio.ed.ac.uk', username='upload',
                 password='***REMOVED***'):
        self.conn = None
        self.host = host
        self.username = username
        self.password = password

    def get_meta(self):
        pass

    def __enter__(self):
        self.conn = BlitzGateway(host=self.host, username=self.username,
                                 passwd=self.password)
        self.conn.connect()
        return self

    def __exit__(self, *exc):
        self.conn.close()
        return False


class Dataset(Argo):
    def __init__(self, expt_id):
        super().__init__()
        self.expt_id = expt_id

    def get_images(self):
        dataset = self.conn.getObject("Dataset", self.expt_id)
        return {im.getName(): im.getId() for im in dataset.listChildren()}


class Image(Argo):
    def __init__(self, image_id):
        super().__init__()
        self.image_id = image_id
        self._image_wrap = None

    @property
    def image_wrap(self):
        # TODO check that it is alive/ connected
        if self._image_wrap is None:
            self._image_wrap = self.conn.getObject("Image", self.image_id)
        return self._image_wrap

    @property
    def name(self):
        return self.image_wrap.getName()

    @property
    def data(self):
        return get_data_lazy(self.image_wrap)


class Cells(CellsHDF):
    def __init__(self, filename):
        file = h5py.File(filename, 'r')
        super().__init__(file)

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close
        return False
