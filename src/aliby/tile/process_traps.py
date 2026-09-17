"""
Find ALCATRAS traps.

The detector lives in tiler.detect; this module keeps its old import path.
"""

from tiler.detect import (  # noqa: F401
    correct_illumination,
    half_ceil,
    half_floor,
    identify_trap_locations,
    plot_trap_locations,
    segment_traps,
)
