#!/usr/bin/env jupyter


def dispatch_tracker(kind: str, **kwargs):
    match kind:
        case "stitch":
            from aliby.track.trackers import stitch

            return stitch
        case "baby":
            raise Exception("Not yet implemented ")
        case _:
            return None
