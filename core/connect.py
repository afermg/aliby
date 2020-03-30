#!/usr/bin/env python
# -*- coding: utf-8 -*-

#
# Copyright (C) 2014 University of Dundee & Open Microscopy Environment.
#                    All Rights Reserved.
# Use is subject to license terms supplied in LICENSE.txt
import itertools
import numpy as np

from omero.gateway import BlitzGateway

from core.utils import repr_obj

# TODO inheritance
#   Advantages:
#       * lots of things not to rewrite
#       * the omero wrapper is inheritance-based anyway
#       * great functionality out of the box
#   Disadvantages:
#       * camelCase :(



class Database:
    def __init__(self, username, password, host, port):
        self.conn = BlitzGateway(username, password, host=host, port=port)

    def __repr__(self):
        return repr_obj(self.conn)
        _

    def connect(self, secure=False):
        connected = self.conn.connect()
        self.conn.setSecure(secure)
        return connected

    def disconnect(self):
        self.conn.seppuku()

    @property
    def user(self):
        # TODO cache
        user = self.conn.getUser()
        return dict(ID=user.getId(), Username=user.getName())

    @property
    def groups(self):
        # TODO cache
        return [dict(ID=g.getId(), Name=g.getName()) for g in
                self.conn.getGroupsMemberOf()]

    @property
    def current_group(self):
        # TODO cache
        g = self.conn.getGroupFromContext()
        return dict(ID=g.getId(), Name=g.getName())

    def isAdmin(self):
        # TODO cache
        return self.conn.isAdmin()

    def getDataset(self, dataset_id):
        ds = self.conn.getObject("Dataset", dataset_id)
        return Dataset(ds)

    def getDatasets(self, n):
        top_n = itertools.islice(self.conn.getObjects("Dataset"), n)
        return [Dataset(ds) for ds in top_n]


class Dataset:
    def __init__(self, dataset_wrapper):
        self.dataset = dataset_wrapper
        self._name = None

    def __repr__(self):
        return repr_obj(self.dataset)

    @property
    def name(self):
        if self._name is None:
            self._name = self.dataset.getName()
        return self._name

    def getImages(self, n=None):
        if n is None:
            return [Image(im) for im in self.dataset.listChildren()]
        top_n = itertools.islice(self.dataset.listChildren(), n)
        return [Image(im) for im in top_n]


class Image:
    def __init__(self, image_wrapper):
        self.image = image_wrapper
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
    def channels(self):
        if self._channels is None:
            self._channels = self.image.getChannelLabels()
        return self._channels

    @property
    def annotations(self):
        return self.dataset.listAnnotations()

    def __repr__(self):
        return repr_obj(self.image)

    def getThumbnail(self):
        thumb_str = self.image.getThumbnail(z=0, t=0)
        # FIXME thumbnail returns None, there is an error

    def getHypercube(self, x=None, y=None, width=None, height=None,
                     z_positions=None,
                     channels=None,
                     timepoints=None):

        if None in [x, y, width, height]:
            tile = None  # Get full plane
        else:
            tile = (x, y, width, height)

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
