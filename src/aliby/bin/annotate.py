"""
Asynchronous annotation (in one thread). Used as a base to build threading-based annotation.
Currently only works on UNIX-like systems due to using "/" to split addresses.

Usage example

From python
$ python annotator.py --image_path path/to/folder/with/h5files --results_path path/to/folder/with/images/zarr --pos position_name --ncells max_n_to_annotate

As executable (installed via poetry)
$ annotator.py --image_path path/to/folder/with/h5files --results_path path/to/folder/with/images/zarr --pos position_name --ncells max_n_to_annotate

During annotation:
- Assign a (binary) label by typing '1' or '2'.
- Type 'u' to undo.
- Type 's' to skip.
- Type 'q' to quit.

File will be saved in: ./YYYY-MM-DD_annotation/annotation.csv, where YYYY-MM-DD is the current date.

"""
import argparse
import logging
import typing as t

from copy import copy
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import readchar
import trio
from agora.utils.cast import _str_to_int
from aliby.utils.vis_tools import _sample_n_tiles_masks
from aliby.utils.plot import stretch

# Remove logging warnings
logging.getLogger("aliby").setLevel(40)

# Defaults

essential = {"image_path": "zarr", "results_path": "h5"}
param_values = dict(
    out_dir=f"./{datetime.today().strftime('%Y_%m_%d')}_annotation/",
    pos=None,
    ncells=100,
    min_tp=100,
    max_tp=150,
    seed=0,
)

annot_filename = "annotation.csv"

# Parsing
parser = argparse.ArgumentParser(
    prog="aliby-annot-binary",
    description="Annotate cells in a binary manner",
)
for i, arg in enumerate((*essential, *param_values)):
    parser.add_argument(
        f"--{arg}",
        action="store",
        default=param_values.get(arg),
        required=i < len(essential),
    )

args = parser.parse_args()

for i, k in enumerate((*essential, *param_values.keys())):
    # Assign essential values as-is
    if i < len(essential):
        param_values[k] = getattr(args, k)

    # Fill additional values
    if passed_value := getattr(args, k):
        param_values[k] = passed_value
        try:
            param_values[k] = _str_to_int(passed_value)
        except Exception as exc:
            pass

for k, suffix in essential.items():  # Autocomplete if fullpath not provided
    if not str(param_values[k]).endswith(suffix):
        param_values[k] = (
            Path(param_values[k]) / f"{ param_values['pos'] }.{suffix}"
        )


# Functions
async def generate_image(stack, skip: bool = False):
    await trio.sleep(1)
    result = np.random.randint(100, size=(10, 10))
    stack.append(result)


async def draw(data, drawing):
    if len(drawing) > 1:
        for ax, img in zip(drawing, data):
            if np.isnan(img).sum():  # Stretch masked channel
                img = stretch(img)
            ax.set_data(img)
    else:
        drawing.set_data(data)
    plt.draw()
    plt.pause(0.1)


def annotate_image(current_key=None, valid_values: t.Tuple[int] = (1, 2)):
    # Show image to annotate
    while current_key is None or current_key not in valid_values:
        if current_key is not None:
            print(
                f"Invalid value. Please try with valid values {valid_values}"
            )

        if (current_key := readchar.readkey()) in "qsu":
            # if (current_key := input()) in "qsu":
            break

        current_key = _parse_input(current_key, valid_values)

    return current_key


async def generate_image(
    generator,
    location_stack: t.List[t.Tuple[np.ndarray, t.Tuple[int, int, int]]],
):
    new_location_image = next(generator)
    location_stack.append((new_location_image[0], new_location_image[1]))


def _parse_input(value: str, valid_values: t.Tuple[int]):
    try:
        return int(value)
    except:
        print(
            f"Non-parsable value. Please try again with valid values {valid_values}"
        )
        return None


def write_annotation(
    experiment_position: str,
    out_dir: Path,
    annotation: str,
    location_stack: t.Tuple[t.Tuple[int, int, int], np.ndarray],
):
    location, stack = location_stack
    unique_location = list(map(str, (*experiment_position, *location)))

    write_into_file(
        out_dir / annot_filename,
        ",".join((*unique_location, str(annotation))) + "\n",
    )
    bg_zero = copy(stack[1])
    bg_zero[np.isnan(bg_zero)] = 0
    tosave = np.stack((stack[0], bg_zero.astype(int)))
    # np.savez(out_dir / f"{'_'.join( unique_location )}.npz", tosave)
    np.save(out_dir / f"{'.'.join( unique_location )}.npy", tosave)


def write_into_file(file_path: str, line: str):
    with open(file_path, "a") as f:
        f.write(str(line))


async def annotate_images(
    image_path, results_path, out_dir, ncells, seed, interval
):

    preemptive_cache = 3

    location_stack = []
    out_dir = Path(out_dir)
    out_annot_file = str(out_dir / annot_filename)

    generator = _sample_n_tiles_masks(
        image_path, results_path, ncells, seed=seed, interval=interval
    )

    # Fetch a few positions preemtively
    async with trio.open_nursery() as nursery:
        for _ in range(preemptive_cache):
            nursery.start_soon(generate_image, generator, location_stack)

        print("parent: waiting for first annotations.")

        _, ax = plt.subplots(figsize=(10, 8))
        while not location_stack:  # Wait until first image is loaded
            await trio.sleep(0.1)

        from aliby.utils.plot import plot_overlay

        # drawing = ax.imshow(location_stack[0][1])
        axes = plot_overlay(*location_stack[0][1], ax=ax.axes)
        plt.show(block=False)
        plt.draw()
        plt.pause(0.5)  # May be adjusted based on display speed

    try:
        out_dir.mkdir(parents=True)
    except:
        pass
    if not Path(out_annot_file).exists():
        write_into_file(
            out_annot_file,
            ",".join(
                (
                    "experiment",
                    "position",
                    "tile",
                    "cell_label",
                    "tp",
                    "annotation",
                )
            )
            + "\n",
        )

    # Loop until n_max or quit
    for i in range(1, ncells - preemptive_cache + 1):
        # Wait for input
        print("Enter a key")
        annotation = str(annotate_image())

        if annotation == "q":
            break
        elif annotation == "s":
            print("Skipping...")
            # continue
        elif annotation == "u":
            i -= 1
        elif isinstance(_str_to_int(annotation), int):
            write_annotation(
                str(results_path).split(".")[0].split("/")[-2:],
                out_dir,
                annotation,
                location_stack[i],
            )

        print(location_stack[i][0])
        # Append into annotations file
        async with trio.open_nursery() as nursery:
            nursery.start_soon(generate_image, generator, location_stack)
            nursery.start_soon(draw, location_stack[i][1], axes)

    print("Annotation done!")


# if __name__ == "__main__":
def annotate():
    if any([param_values.get(k) is None for k in ("min_tp", "max_tp")]):
        interval = None
    else:
        interval = (param_values["min_tp"], param_values["max_tp"])

    print(param_values)
    trio.run(
        annotate_images,
        param_values["image_path"],
        param_values["results_path"],
        param_values["out_dir"],
        param_values["ncells"],
        param_values["seed"],
        interval,
    )
