"""
This code requires a functional OMERO database on localhost at port 4064
See the README for instructions as to how to set these up with docker.
"""
# TODO remove and use unittest to run tests
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

USERNAME = "root"
PASSWORD = "omero-root-password"
HOST = "localhost"
PORT = 4064
from omero.gateway import BlitzGateway


def print_obj(obj, indent=0):
    """
    Helper method to display info about OMERO objects.
    Not all objects will have a "name" or owner field.
    """
    print(
        """%s%s:%s  Name:"%s" """
        % (" " * indent, obj.OMERO_CLASS, obj.getId(), obj.getName())
    )


if __name__ == "__main__":

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
            " password.\n"
        )
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
    print("Current user:")
    print("   ID:", user.getId())
    print("   Username:", user.getName())
    print("   Full Name:", user.getFullName())

    # Check if you are an Administrator
    print("   Is Admin:", conn.isAdmin())

    # Close connection
    # ================
    # When you are done, close the session to free up server resources.
    conn.close()
