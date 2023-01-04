#!/usr/bin/env jupyter
from importlib_resources import files
from logfile_parser import Parser

grammars_dir = files("logfile_parser") / "grammars"


def get_examples_dir():
    return files("aliby").parent.parent / "examples" / "logfile_parser"


def get_logfile_grammars_dir():
    return files("logfile_parser") / "grammars"


def get_legacy_log_example_interface() -> dict:
    parsed = {}
    examples_dir = get_examples_dir()
    grammars_dir = get_logfile_grammars_dir()

    for gram in ("acq", "log"):
        for gram_file in grammars_dir.glob(f"multiDGUI_{gram}_format.json"):
            parser = Parser(gram_file)
            for file_to_parse in examples_dir.glob(f"*{gram}.txt"):
                with open(file_to_parse, "r") as f:
                    parsed = {**parsed, **parser.parse(f)}
    return parsed


def to_legacy(parsed_logfile: dict) -> dict:
    """
    Convert the output of the new logfile parsing to legacy to a minimal working set of metadata.
    This converts the new more complex metadata structure to the previous one that did not have configuration profiles, but instead one configuration per channel.
    This is a temporal solution as we transition into a more general metadata structure that accounts for heterogeneous groups.

    We convert image configs to channels, and add general metadata to the root.
    """
    name_translation = {
        "Microscope name": "microscope",
    }
    channel_name_translation = {
        "Image config": "channel",
        "Channel": "channel_hardware",
        "Exposure (ms)": "exposure",
        "Z spacing (um)": "zsect",
    }

    # Translate general data
    general = {v: d["general"][k] for k, v in name_translation.items()}
    # Translate and cast image configs
    channels = {
        v: list(map(_cast_type, parsed_logfile["image_config"][k]))
        for k, v in channel_name_translation.items()
    }
    legacy_format = {"channels": channels, **general}
    return legacy_format
