#!/usr/bin/env jupyter
"""
Wrappers for tracking implementations.
"""


def dispatch_tracker(kind: str, **kwargs):
    match kind:
        case "stitch":  # Takes masks as input
            from aliby.track.trackers import stitch_rois

            return stitch_rois
        case "baby":  # Baby itself produces tracking, we just need to fetch it
            assert "crawler" in kwargs, "Baby must be passed a crawler object"

            crawler = kwargs["crawler"]

            def fetch_tracking_info() -> list[list[int]]:
                # Pass the tracking info for the last timepoint of every tile
                latest_tracking_info = {
                    i: x["cell_lbls"][-1] for i, x in enumerate(crawler.tracker_states)
                }
                return latest_tracking_info

            return fetch_tracking_info
        case _:
            return None
