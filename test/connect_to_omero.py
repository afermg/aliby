
#!/usr/bin/env python
# -*- coding: utf-8 -*-

#
# Copyright (C) 2014 University of Dundee & Open Microscopy Environment.
#                    All Rights Reserved.
# Use is subject to license terms supplied in LICENSE.txt
#

"""
FOR TRAINING PURPOSES ONLY!
"""
import sys
sys.path.insert(0, './omero_py')


USERNAME = 'upload'
PASSWORD = '***REMOVED***'
HOST = 'sce-bio-c04287.bio.ed.ac.uk'
PORT = 4064
from omero.gateway import BlitzGateway

def print_obj(obj, indent=0):
    """
    Helper method to display info about OMERO objects.
    Not all objects will have a "name" or owner field.
    """
    print """%s%s:%s  Name:"%s" (owner=%s)""" % (
        " " * indent,
        obj.OMERO_CLASS,
        obj.getId(),
        obj.getName(),
        obj.getAnnotation())


if __name__ == '__main__':
    """
    NB: This block is only run when calling this file directly
    and not when imported.
    """
    """
    start-code
    """

    # Connect to the Python Blitz Gateway
    # ===================================
    # Make a simple connection to OMERO, printing details of the
    # connection. See OmeroPy/Gateway for more info
    conn = BlitzGateway(USERNAME, PASSWORD, host=HOST, port=PORT)
    connected = conn.connect()

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

    # conn.setSecure(True)         # <--------- Uncomment this

    # Current session details
    # =======================
    # By default, you will have logged into your 'current' group in OMERO.
    # This can be changed by switching group in the OMERO.insight or OMERO.web
    # clients.

    user = conn.getUser()
    print "Current user:"
    print "   ID:", user.getId()
    print "   Username:", user.getName()
    print "   Full Name:", user.getFullName()

    # Check if you are an Administrator
    print "   Is Admin:", conn.isAdmin()

    print "Member of:"
    for g in conn.getGroupsMemberOf():
        print "   ID:", g.getId(), " Name:", g.getName()
    group = conn.getGroupFromContext()
    print "Current group: ", group.getName()

    # List the group owners and other members
    owners, members = group.groupSummary()
    print "   Group owners:"
    for o in owners:
        print "     ID: %s UserName: %s Name: %s" % (
            o.getId(), o.getOmeName(), o.getFullName())
    print "   Group members:"
    for m in members:
        print "     ID: %s UserName: %s Name: %s" % (
            m.getId(), m.getOmeName(), m.getFullName())

    print "Owner of:"
    for g in conn.listOwnedGroups():
        print "   ID: ", g.getId(), " Name:", g.getName()

    # New in OMERO 5
    print "Admins:"
    for exp in conn.getAdministrators():
        print "   ID: %s UserName: %s Name: %s" % (
            exp.getId(), exp.getOmeName(), exp.getFullName())

    # The 'context' of our current session
    ctx = conn.getEventContext()
    # print ctx     # for more info

    # Read Data
    my_exp_id = conn.getUser().getId()
    default_group_id = conn.getEventContext().groupId
    i = 0
    for project in conn.getObjects("Project"):
        if i > 10:
            break
        print_obj(project)
        i+=1
    
    i = 0
    for dataset in conn.searchObjects(["Dataset"], "sga", batchSize=12):
        if i > 10:
            break
        print_obj(dataset)
        i+=1

    # Close connection
    # ================
    # When you are done, close the session to free up server resources.
    conn.seppuku()
