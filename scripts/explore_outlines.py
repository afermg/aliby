import h5py

import numpy as np
from skimage.color import label2rgb

from core.io.signal import Signal

s = Signal(
    "/shared_libs/pydask/pipeline-core/data/2021_08_21_KCl_pH_00/YST_1510_009.h5"
)
# s = Signal(
#     "/shared_libs/pydask/pipeline-core/data/2021_06_15_pypipeline_unit_test_00/pos001.h5"
# )
# s = Signal(
#     "/shared_libs/pydask/pipeline-core/data/2021_06_27_downUpShift_2_0_2_glu_dual_phl_ura8__00/phl_ura8_012.h5"
# )
# df = s["extraction/general/None/volume"]
df = s["extraction/em_ratio/np_max/mean"]
df = df[df.notna().sum(axis=1) > 150]
# df = df.loc[df.notna().sum(axis=1) > 100]

with h5py.File(s.filename, "r") as f:
    indices = f["cell_info/edgemasks/indices"][()]
    values = f["cell_info/edgemasks/values"][()]
    labels = f["cell_info/cell_label"][()]
    trap_ids = f["cell_info/trap"][()]
    timepoint = f["cell_info/timepoint"][()]

# # Find the trap with the longest-lasting cell
# trap_id = df.notna().groupby("trap").apply(sum).sum(axis=1).argmax()
# cell_id = 1

# traps = values[(indices == (trap_id, cell_id)).all(axis=1)]
# traps = traps.sum(axis=0)
# bgs = values[indices[:, 0] == trap_id]
# bgs = bgs.sum(axis=0).astype(bool)

# overlays = [
#     label2rgb(trap, image=bg, bg_label=0, alpha=0.5) for bg, trap in zip(bgs, traps)
# ]
# area = [t.sum() for t in traps]

# import numpy as np
# import matplotlib.pyplot as plt
# from utils_find_1st import find_1st, cmp_larger

# if len(area) > 2:
#     start = find_1st(np.array(area), 0, cmp_larger)
#     end = len(area) - find_1st(np.array(area[::-1]), 0, cmp_larger)
# else:
#     start = 0
#     end = 2


# nrows = 5
# ncols = 8
# fig, axes = plt.subplots(nrows, ncols)
# if nrows > 1 and ncols > 1:
#     for i in range(nrows):
#         for j in range(ncols):
#             axes[i, j].set_title(i * ncols + j)
#             axes[i, j].imshow(overlays[start:end][i * ncols + j])
# else:
#     for i in range(max(nrows, ncols)):
#         axes[i].set_title(i)
#         axes[i].imshow(overlays[start:end][i])

# plt.show()

# Now bring the brightfield for comparison

from core.io.omero import Dataset, Image

expt_id = 20191
# expt_id = 19993
tps = [0, 1]

with Dataset(int(expt_id)) as conn:
    image_ids = conn.get_images()
pos_name = s.filename.split("/")[-1][:-3]
with Image(image_ids[pos_name]) as image:
    dimg = image.data
    print("computing")
    imgs = dimg[tps, image.metadata["channels"].index("Brightfield"), 2, ...].compute()

# And crop the trap of interest

with h5py.File(s.filename, "r") as f:
    traplocs = f["trap_info/trap_locations"][()]
    drifts = f["trap_info/drifts"][()]
    drifts = drifts[-2:, :]


def get_tile(tile_size=117):
    tile = np.ones((tile_size, tile_size))
    tile[1:-1, 1:-1] = False
    return tile


tile = get_tile()


def stretch_image(image):
    image = ((image - image.min()) / (image.max() - image.min())) * 255
    minval = np.percentile(image, 2)
    maxval = np.percentile(image, 98)
    image = np.clip(image, minval, maxval)
    image = (image - minval) / (maxval - minval)
    return image


over_time = []
for i, img in enumerate(imgs):
    mask = np.zeros_like(img)
    traplocs_corrected = (traplocs - np.sum(drifts[: i + 1], axis=0)).astype(int)
    cell_ids = []
    for t_id in range(traplocs.shape[0]):  # assign labels to masks
        outlines = values[indices[:, 0] == t_id].astype(int)
        if outlines.any():
            outlines = outlines[:, i]
            tmp = labels[(trap_ids == t_id) & (timepoint == i)]
            for j, label in enumerate(tmp):
                outlines[j][outlines[j] == True] = label + 1
            outlines = outlines.max(axis=0)
        else:
            outlines = np.zeros(outlines.shape[-2:])
        cell_ids.append(outlines)
    for t_id, (x, y) in enumerate(traplocs_corrected):
        dist = int(tile.shape[0] / 2)
        tile_outlines = tile + cell_ids[t_id]
        size_okay = (
            np.array(mask[x - dist : x + dist + 1, y - dist : y + dist + 1].shape)
            == np.array(tile.shape)
        ).all()
        if size_okay:
            maxes = np.maximum.reduce(
                (mask[x - dist : x + dist + 1, y - dist : y + dist + 1], tile_outlines)
            )
            mask[x - dist : x + dist + 1, y - dist : y + dist + 1] = maxes

    traps_img = label2rgb(mask, image=stretch_image(img), bg_label=0, alpha=0.5)
    over_time.append(traps_img)

# Save multiple plots to file
# for i, pos in enumerate(over_time[19:], 19):
#     plt.imshow(pos)
#     plt.axis("off")
#     plt.savefig("pos_" + str(i), dpi=400)
