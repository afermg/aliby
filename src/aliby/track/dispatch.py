#!/usr/bin/env jupyter
"""
The kwargs must be lists of masks.
"""


def dispatch_tracker(kind: str, **kwargs):
    match kind:
        case "stitch":
            from aliby.track.trackers import stitch_rois

            return stitch_rois
        case "baby":
            raise Exception("Not yet implemented ")
        case _:
            return None
