import json
import time
import numpy as np

from core.experiment import ExperimentLocal
from core.segment import Tiler
from core.baby_client import BabyClient

root_dir = '/Users/s1893247/PhD/pipeline-core/data/glclvl_0' \
           '.1_mig1_msn2_maf1_sfp1_dot6_03'

expt = ExperimentLocal(root_dir, finished=True)
seg_expt = Tiler(expt, finished=True)

print(seg_expt.positions)
seg_expt.current_position = 'pos007'

config = {"camera": "evolve",
          "channel": "brightfield",
          "zoom": "60x",
          "n_stacks": "5z"}

baby_client = BabyClient(expt, **config)

print("The session is {}".format(baby_client.sessions['default']))

# Channel 0, 0, X,Y,Z all
num_timepoints = 5

traps_tps = [seg_expt.get_traps_timepoint(tp, tile_size=81, channels=[0],
                                         z=[0, 1, 2, 3, 4]).squeeze()
            for tp in range(num_timepoints)]

segmentations = []
try:
    for i, timpoint in enumerate(traps_tps):
        print("Sending timepoint {};".format(i))
        status = baby_client.queue_image(timpoint,
                                         baby_client.sessions['default'],
                                         assign_mothers=True,
                                         return_baprobs=True,
                                         with_edgemasks=True)
        while True:
            try:
                print('Loading.', end='')
                result = baby_client.get_segmentation(baby_client.sessions[
                                                          'default'])
            except:
                print('.', end='')
                time.sleep(1)
                continue
            break
        print("Received timepoint {}".format(i))
        segmentations.append(result)
except Exception as e:
    print(segmentations)
    raise e

with open('segmentations.json', 'w') as fd:
    json.dump(segmentations, fd)

print('Done.')
# print(len(segmentations[0]))
# for i in range(5):
#     print("trap {}".format(i))
#     for k, v in segmentations[0][i].items():
#         print(k, v)
#
# import matplotlib.pyplot as plt
# plt.imshow(np.squeeze(batches[0][0, ..., 0]))
# plt.savefig('test_baby.pdf')