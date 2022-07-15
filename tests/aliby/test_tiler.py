import argparse

from aliby.io.image import ImageLocal

# from aliby.experiment import ExperimentLocal
from aliby.tile.tiler import Tiler, TilerParameters


def define_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("root_dir")
    parser.add_argument("--position", default=None)
    parser.add_argument("--template", default=None)
    parser.add_argument("--trap", type=int, default=0)
    parser.add_argument("--channel", type=str, default="Brightfield")
    parser.add_argument("-z", "--z_positions", type=int, default=5)
    parser.add_argument("--time", type=int, default=0)
    parser.add_argument("--tile_size", type=int, default=96)
    return parser


def initialise_objects(data_path, template=None):
    image = ImageLocal(data_path)
    tiler = Tiler.from_image(image, TilerParameters.default())
    return tiler


def change_position(position, tiler):
    tiler.current_position = position


def get_n_traps_timepoints(tiler):
    return tiler.n_traps, tiler.n


def trap_timelapse(tiler, trap_idx, channel, z):
    channel_id = tiler.get_channel_index(channel)
    timelapse = tiler.get_trap_timelapse(
        trap_idx, channels=[channel_id], z=list(range(z))
    )
    return timelapse


def timepoint_traps(tiler, tp_idx, channel, z, tile_size):
    channel_id = tiler.get_channel_index(channel)
    traps = tiler.get_traps_timepoint(
        tp_idx, tile_size=tile_size, channels=[channel_id], z=list(range(z))
    )
    return traps


if __name__ == "__main__":
    parser = define_parser()
    args = parser.parse_args()

    tiler = initialise_objects(args.root_dir, template=args.template)

    if args.position is not None:
        tiler.current_position = args.position

    n_traps, n_tps = get_n_traps_timepoints(tiler)

    timelapse = trap_timelapse(
        tiler, args.trap, args.channel, args.z_positions
    )
    traps = timepoint_traps(
        tiler, args.time, args.channel, args.z_positions, args.tile_size
    )
