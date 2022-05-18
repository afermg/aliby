from omero.gateway import BlitzGateway


class Argo:
    """
    Base OMERO-interactive class
    """

    def __init__(
        self, host="islay.bio.ed.ac.uk", username="upload", password="***REMOVED***"
    ):
        self.conn = None
        self.host = host
        self.username = username
        self.password = password

    def __enter__(self):
        self.conn = BlitzGateway(
            host=self.host, username=self.username, passwd=self.password
        )
        self.conn.connect()
        return self

    def __exit__(self, *exc):
        self.conn.close()
        return False
