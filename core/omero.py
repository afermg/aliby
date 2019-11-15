"""
Describes interaction with Omero, including the Dataset and Database.
"""


# TODO Fake Database interface based on filesystem organisation and local
#  configuration files

# TODO ImageCache as a transparent object? Dataset + Timelapse + Results can
#  be linked to local file structure instead of OmeroDatabase.


class Database:
    """
    Interface to the Omero Database.
    """

    def __init__(self, server):
        self.server = server
        self._init_connection()

    def _init_connection(self):
        pass


class Dataset:
    """
    A single Dataset, obtained from Omero.

    Contains attributes and properties:
    * A Unique ID
    * Dataset metadata
    * A list of position names
    * A list of channels
    * Image size
    """

    def __init__(self, id):
        self.id = id

    def getHyperCube(self, x=None, y=None, z=None, c=None, t=None):
        """
        Get a set of images from the database.
        :param x: Range of pixels in the x direction
        :param y: Range of pixels in the y direction
        :param z: Range of stacks in the z direction
        :param c: List channel positions
        :param t: List of time points
        :return: A 5 dimensional nd.array ordered (x,y,z,c,t)
        """
        pass
