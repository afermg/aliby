from omero.gateway import BlitzGateway

class Argo:
    """
    Base class to interact with OMERO.
    See
    https://docs.openmicroscopy.org/omero/5.6.0/developers/Python.html
    """

    def __init__(
        self,
        host="islay.bio.ed.ac.uk",
        username="upload",
        password="***REMOVED***",
    ):
        """
        Parameters
        ----------
        host : string
            web address of OMERO host
        username: string
        password : string
        """
        self.conn = None
        self.host = host
        self.username = username
        self.password = password

    # standard method required for Python's with statement
    def __enter__(self):
        self.conn = BlitzGateway(
            host=self.host, username=self.username, passwd=self.password
        )
        self.conn.connect()
        return self

    # standard method required for Python's with statement
    def __exit__(self, *exc):
        self.conn.close()
        return False
