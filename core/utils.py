"""
Utility functions
"""
import imageio


def repr_obj(obj, indent=0):
    """
    Helper function to display info about OMERO objects.
    Not all objects will have a "name" or owner field.
    """
    string = """%s%s:%s  Name:"%s" (owner=%s)""" % (
        " " * indent,
        obj.OMERO_CLASS,
        obj.getId(),
        obj.getName(),
        obj.getAnnotation())

    return string


class ImageCache:
    """
    Cache of images to avoid multiple loading.
    """
    def __init__(self, max_len=200):
        """

        :param max_len:
        """
        self._dict = dict()
        self._queue = []
        self.max_len=max_len

    def __getitem__(self, item):
        if item not in self._dict:
            self.load_image(item)
        return self._dict[item]

    # TODO make the loading function a parameter so we can use this for both
    #  the local and the OMERO timelapses
    def load_image(self, item):
        self._dict[item] = imageio.imread(item)
        # Clean up the queue
        self._queue.append(item)
        if len(self._queue) > self.max_len:
            del self._dict[self._queue.pop(0)]

