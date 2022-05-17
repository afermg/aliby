import h5py
import omero
from omero.gateway import BlitzGateway
from aliby.experiment import get_data_lazy
from agora.io.cells import CellsHDF


class Argo:
    def __init__(
        self, host="islay.bio.ed.ac.uk", username="upload", password="***REMOVED***"
    ):
        self.conn = None
        self.host = host
        self.username = username
        self.password = password

    def get_meta(self):
        pass

    def __enter__(self):
        self.conn = BlitzGateway(
            host=self.host, username=self.username, passwd=self.password
        )
        self.conn.connect()
        return self

    def __exit__(self, *exc):
        self.conn.close()
        return False


class Dataset(Argo):
    def __init__(self, expt_id, **server_info):
        super().__init__(**server_info)
        self.expt_id = expt_id
        self._files = None

    @property
    def dataset(self):
        return self.conn.getObject("Dataset", self.expt_id)

    @property
    def name(self):
        return self.dataset.getName()

    @property
    def date(self):
        return self.dataset.getDate()

    @property
    def unique_name(self):
        return "_".join(
            (
                str(self.expt_id),
                self.date.strftime("%Y_%m_%d").replace("/", "_"),
                self.name,
            )
        )

    def get_images(self):
        return {im.getName(): im.getId() for im in self.dataset.listChildren()}

    @property
    def files(self):
        if self._files is None:
            self._files = {
                x.getFileName(): x
                for x in self.dataset.listAnnotations()
                if isinstance(x, omero.gateway.FileAnnotationWrapper)
            }
        if not len(self._files):
            raise Exception("Exception:Metadata: Experiment has no annotation files.")
        return self._files

    @property
    def tags(self):
        if self._tags is None:
            self._tags = {
                x.getName(): x
                for x in self.dataset.listAnnotations()
                if isinstance(x, omero.gateway.TagAnnotationWrapper)
            }
        return self._tags

    def cache_logs(self, root_dir):
        for name, annotation in self.files.items():
            filepath = root_dir / annotation.getFileName().replace("/", "_")
            if str(filepath).endswith("txt") and not filepath.exists():
                # Save only the text files
                with open(str(filepath), "wb") as fd:
                    for chunk in annotation.getFileInChunks():
                        fd.write(chunk)
        return True


class Cells(CellsHDF):
    def __init__(self, filename):
        file = h5py.File(filename, "r")
        super().__init__(file)

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close
        return False
