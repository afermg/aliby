import time
import numpy as np

from core.experiment import ExperimentLocal
from core.segment import Tiler
from core.baby_client import BabyClient

root_dir = '/Users/s1893247/PhD/pipeline-core/data/glclvl_0' \
           '.1_mig1_msn2_maf1_sfp1_dot6_03'

expt = ExperimentLocal(root_dir)
seg_expt = Tiler(expt)

print(seg_expt.positions)
seg_expt.current_position = 'pos007'

config = {"camera": "evolve",
          "channel": "brightfield",
          "zoom": "60x",
          "n_stacks": "5z"}

baby_client = BabyClient(expt, **config)

print("The session is {}".format(baby_client.sessions['default']))

# Channel 0, TP all, X,Y,Z all
full_img = seg_expt.get_trap_timelapse(15, tile_size=81, z=[0,1,2,3,4])[0, :]
print(full_img.shape)

batches = np.array_split(full_img, 8, axis=0)

segmentations = []
try:
    for i, batch in enumerate(batches):
        print("Sending batch {};".format(i))
        status = baby_client.queue_image(batch,
                                         baby_client.sessions['default'],
                                         assignbuds=True,
                                         with_edgemasks=False)
        while True:
            try:
                result = baby_client.get_segmentation()
            except:
                time.sleep(2)
                continue
            break
        print("Received batch {}".format(i))
        segmentations.append(result)
except Exception as e:
    print(segmentations)
    raise e

print(len(segmentations[0]))
for i in range(5):
    print("timepoint {}".format(i))
    for k, v in segmentations[0][i].items():
        print(k, v)

import matplotlib.pyplot as plt
plt.imshow(np.squeeze(batches[0][0, ..., 0]))
plt.savefig('test_baby.pdf')