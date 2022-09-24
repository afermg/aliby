from omero.gateway import BlitzGateway
from tqdm import tqdm


# Helper funs
def connect_omero():
    conn = BlitzGateway(*get_creds(), host="islay.bio.ed.ac.uk", port=4064)
    conn.connect()
    return conn


def get_creds():
    return (
        "upload",
        "***REMOVED***",  # OMERO Password
    )


def download_file(f):
    """
    Download file in chunks using FileWrapper object
    """
    desc = (
        "Downloading "
        + f.getFileName()
        + " ("
        + str(round(f.getFileSize() / 1000**2, 2))
        + "Mb)"
    )

    down_file = bytearray()
    for c in tqdm(f.getFileInChunks(), desc=desc):
        down_file += c

    return down_file
