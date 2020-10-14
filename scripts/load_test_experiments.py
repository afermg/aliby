import sys
import json
import numpy as np
from pathlib import Path

import omero_py as op
import omero

from core.connect import Database


def load_experiment(expt_id, save_dir):
    save_dir = Path(save_dir)
    with open('config.json', 'r') as fd:
        config = json.load(fd)
    db = Database(config['user'],
                  config['password'],
                  config['host'],
                  config['port'])
    db.connect()
    ds = db.getDataset(expt_id)
    print('Experiment: {}'.format(ds.name))

    save_dir = save_dir / ds.name
    if not save_dir.exists():
        save_dir.mkdir(parents=True)

    # Load the annotation files
    tag_fd = open(str(save_dir / 'tags.txt'), 'w')

    for ann in ds.dataset.listAnnotations():
        if isinstance(ann, omero.gateway.FileAnnotationWrapper):
            with open(str(save_dir / ann.getFileName()), 'w') as fd:
                for chunk in ann.getFileInChunks():
                    fd.write(chunk)
        else:
            tag_fd.write('{} : {}\n'.format(ann.getDescription(),
                                            ann.getValue()))
    tag_fd.close()

    for img in list(ds.getImages()):
        im_name = img.name
        print("Getting image {}".format(im_name))
        for ix, channel in enumerate(img.channels):
            print('Getting channel {}'.format(channel))
            channel_array = img.getHypercube(channels=[ix])
            print('Saving to {}'.format(save_dir / (channel + str(im_name))))
            np.save(save_dir / (channel + str(im_name)), channel_array)

    db.disconnect()

if __name__ == "__main__":
    if len(sys.argv) > 1:
        save_dir = Path(sys.argv[1])
        if not save_dir.exists():
            save_dir = './'

    experiment_id = 10863

    load_experiment(10863, save_dir)
