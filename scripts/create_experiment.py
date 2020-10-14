import argparse 
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
import json
import logging
from logging.handlers import RotatingFileHandler

import click
import sqlalchemy as sa
from sqlalchemy.orm import sessionmaker

from database.records import Base

from core.experiment import Experiment
logger = logging.getLogger('core')
logger.handlers = []
logger.setLevel(logging.DEBUG)

console = logging.StreamHandler()
console.setLevel(logging.WARNING)
formatter = logging.Formatter('%(name)-12s: %(levelname)-8s %(message)s')
console.setFormatter(formatter)
logger.addHandler(console)

file_handler = RotatingFileHandler(filename='test.log',
                                   maxBytes=1e5,
                                   backupCount=1)

file_handler.setLevel(logging.DEBUG)
file_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s '
                                   '- %(message)s')
file_handler.setFormatter(file_formatter)
logger.addHandler(file_handler)

@click.command()
@click.argument('config_file')
@click.option('-i', '--expt_id', type=int, help="Experiment ID")
@click.option('-t', '--time', type=int)
@click.option('--save_dir', default='./data/')
@click.option('--db', default='sqlite:///out.db')
def download(config_file, expt_id, time, save_dir, db):
    with open(config_file, 'r') as fd:
        config = json.load(fd)
    if not expt_id:
        expt_id = config['experiment']
    else:
        expt_id = expt_id
    expt = Experiment.from_source(expt_id, config['user'],
                                  config['password'], config['host'],
                                  config['port'], save_dir=save_dir)
    if time:
        timepoints = time
    else:
        timepoints = 0
    print(expt.name)
    print(expt.shape)
    
    # Create SQL database
    engine = sa.create_engine(db)
    Base.metadata.create_all(engine)
    Session = sessionmaker(engine)
    session = Session()
    try:
        expt.run(timepoints, session)
    except Exception as e:
        raise e
    finally:
        expt.connection.close()


@click.group()
def cli():
    pass

cli.add_command(download)

if __name__ == "__main__":
    try:
        cli()
    except Exception as e: 
        print("Caught the thing returning error")
        sys.exit(1)
    finally:
        sys.exit(0)
