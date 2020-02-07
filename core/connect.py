#!/usr/bin/env python
# -*- coding: utf-8 -*-

#
# Copyright (C) 2014 University of Dundee & Open Microscopy Environment.
#                    All Rights Reserved.
# Use is subject to license terms supplied in LICENSE.txt
import itertools

from omero.gateway import BlitzGateway

from utils import repr_obj

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

    def __repr__(self):
        return repr_obj(self.dataset)

    def getImages(self, n):
        top_n = itertools.islice(self.dataset.listChildren(), n)
        return [Image(im) for im in top_n]

class Image:
    def __init__(self, image_wrapper):
        self.image = image_wrapper

    def __repr__(self):
        return repr_obj(self.image)

    def getHypercube(self, x, y, z, c, t):
        pass
