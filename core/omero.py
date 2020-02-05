#!/usr/bin/env python
# -*- coding: utf-8 -*-

#
# Copyright (C) 2014 University of Dundee & Open Microscopy Environment.
#                    All Rights Reserved.
# Use is subject to license terms supplied in LICENSE.txt


USERNAME = 'upload'
PASSWORD = '***REMOVED***'
HOST = 'sce-bio-c04287.bio.ed.ac.uk'
PORT = 4064
from omero.gateway import BlitzGateway

from utils import repr_obj

class Database:
    def __init__(self):
        self.conn = BlitzGateway(USERNAME, PASSWORD, host=HOST, port=PORT)

    def __repr__(self):
        return repr_obj(self.conn)
        _
    def connect(self):
        connected = self.conn.connect()

        # Check if you are connected
        # ==========================
        if not connected:
            import sys
            sys.stderr.write(
                "Error: Connection not available, please check your user name and"
                " password.\n")
            sys.exit(1)

        # Using secure connection
        # =======================
        # By default, once we have logged in, data transfer is not encrypted
        # (faster)
        # To use a secure connection, call setSecure(True):

        #self.conn.setSecure(True)         # <--------- Uncomment this
    
    def disconnect(self):
        self.conn.seppuku()

    @property
    def user(self):
        user = self.conn.getUser()
        return dict(ID=user.getId(), Username=user.getName())

    @property
    def groups(self):
        return [dict(ID=g.getId(), Name=g.getName()) for g in
                self.conn.getGroupsMemberOf()]

    @property
    def current_group(self):
        g = self.conn.getGroupFromContext()
        return dict(ID=g.getId(), Name=g.getName())

    def isAdmin(self):
        return self.conn.isAdmin()

    def getDataset(self, dataset_id):
        ds = self.conn.getObject("Dataset", dataset_id)
        return Dataset(ds)


class Dataset:
    def __init__(self, dataset_wrapper):
        self.dataset = dataset_wrapper

    def __repr__(self):
        return repr_obj(self.dataset)

class Image:
    def __init__(self, image_wrapper):
        self.image = image_wrapper

    def __repr__(self):
        return repr_obj(self.image)

    def getHypercube(self, x, y, z, c, t):
        pass
