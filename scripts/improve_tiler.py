#!/usr/bin/env python3

expts = [18616, 19232, 19995, 19993, 20191, 19831]

# fetch images
test_imgs = []
for e in expts:
    with Dataset(int(e)) as conn:
        image_ids = conn.get_images()
    for im_id in image_ids.values():
        with Image(im_id) as image:
            dimg = image.data
            print("computing")
            img = dimg[
                0, image.metadata["channels"].index("Brightfield"), 2, ...
            ].compute()
            test_imgs.append(img)

from numpy import save, load

# save
for i, nd in enumerate(test_imgs):
    save("raw_" + str(i) + ".png", nd)

# load


def stretch_image(image):
    image = ((image - image.min()) / (image.max() - image.min())) * 255
    minval = np.percentile(image, 2)
    maxval = np.percentile(image, 98)
    image = np.clip(image, minval, maxval)
    image = (image - minval) / (maxval - minval)
    return image


def segment_traps(image, tile_size, downscale=0.4):
    # Make image go between 0 and 255
    img = image  # Keep a memory of image in case need to re-run
    stretched = stretch_image(image)
    img = stretch_image(image)
    # TODO Optimise the hyperparameters
    disk_radius = int(min([0.01 * x for x in img.shape]))
    min_area = 0.2 * (tile_size ** 2)
    if downscale != 1:
        img = transform.rescale(image, downscale)
    entropy_image = entropy(img, disk(disk_radius))
    if downscale != 1:
        entropy_image = transform.rescale(entropy_image, 1 / downscale)

    # apply threshold
    thresh = threshold_otsu(entropy_image)
    bw = closing(entropy_image > thresh, square(3))

    # remove artifacts connected to image border
    cleared = clear_border(bw)

    # label image regions
    label_image = label(cleared)
    areas = [
        region.area
        for region in regionprops(label_image)
        if region.area > min_area and region.area < tile_size ** 2 * 0.8
    ]
    traps = (
        np.array(
            [
                region.centroid
                for region in regionprops(label_image)
                if region.area > min_area and region.area < tile_size ** 2 * 0.8
            ]
        )
        .round()
        .astype(int)
    )

    rprops = regionprops_table(
        label_image,
        properties=[
            "area",
            "eccentricity",
            "convex_area",
            "feret_diameter_max",
            "orientation",
            "solidity",
            "minor_axis_length",
        ],
    )
    trapmask = (rprops["area"] > min_area) & (rprops["area"] < tile_size ** 2 * 0.8)
    candidates = [
        stretched[
            x - tile_size // 2 : x + tile_size // 2,
            y - tile_size // 2 : y + tile_size // 2,
        ]
        for x, y in np.array(traps).round().astype(int)
    ]
    # valleys = [find_valley(c) for c in candidates]

    from copy import copy

    bak = copy(candidates)

    candidates = [bak[x] for x in np.argsort(rprops["minor_axis_length"][trapmask])]
    return candidates[:5]

    # fig, axes = plt.subplots(5, 8)
    # indices = np.concatenate((np.arange(20), -np.arange(1, 21)[::-1]))
    # for i in range(5):
    #     for j in range(8):
    #         if i * 8 + j < len(candidates):
    #             # axes[i, j].imshow(candidates[i * 8 + j])
    #             axes[i, j].imshow(candidates[indices[i * 8 + j]])
    # plt.show()

    # chosen_trap_coords = np.round(traps[np.argsort(area)[len(area) // 2]]).astype(int)
    # chosen_trap_coords = np.round(traps[np.argsort(ma)[len(ma) // 2]]).astype(int)
    x, y = chosen_trap_coords
    template = image[
        x - tile_size // 2 : x + tile_size // 2, y - tile_size // 2 : y + tile_size // 2
    ]
    return template

    new_coords = identify_trap_locations(image, template)

    # def get_tile(tile_size=117):
    #     tile = np.ones((tile_size, tile_size))
    #     tile[1:-1, 1:-1] = False
    #     return tile

    # tile = get_tile(tile_size)
    # # tmp
    # mask = np.zeros_like(image, dtype="bool")
    # # for x, y in np.array(traps).round().astype(int):
    # for x, y in new_coords:
    #     dist = int(tile_size / 2)
    #     size_okay = (
    #         np.array(mask[x - dist : x + dist + 1, y - dist : y + dist + 1].shape)
    #         == np.array(tile.shape)
    #     ).all()
    #     if size_okay:
    #         maxes = np.maximum.reduce(
    #             (mask[x - dist : x + dist + 1, y - dist : y + dist + 1], tile)
    #         )
    #         mask[x - dist : x + dist + 1, y - dist : y + dist + 1] = maxes

    # from skimage.color import label2rgb

    # traps_img = label2rgb(mask, image=stretched, bg_label=0, alpha=0.5)

    if len(traps) < 10 and downscale != 1:
        print("Trying again.")
        return segment_traps(image, tile_size, downscale=1)
    # return traps
    return traps_img


ncols = 10
rands = np.random.randint(0, 138, ncols)
top_cands = [segment_traps(test_imgs[r], tile_size=117) for r in rands]
fig, axes = plt.subplots(5, ncols)
for i in range(ncols):
    for j in range(5):
        axes[j, i].imshow(top_cands[i][j])
plt.show()

# res = [segment_traps(im, tile_size=117) for im in test_imgs[rands]]

from scipy.signal import find_peaks


def find_valley(template):
    template = ((template - template.min()) / (template.max() - template.min())) * 255
    summed = template.sum(axis=1)
    norm = summed / summed.max()
    find_peaks(norm[20:-20])
    max1, max2 = np.argsort(norm[peaks[0]])[:2]
    if max2 < max1:
        tmp = max2
        max2 = max1
        max1 = tmp
    return norm[max1:max2].min()


for i, im in enumerate(res):
    plt.imshow(im)
    plt.axis("off")
    plt.savefig("tiles" + str(i), dpi=400)
