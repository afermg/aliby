#!/usr/bin/env python3
"""
Automatic summarissed report to condense results of an experiment.
It should NOT contain text beyond labels and titles.
The most efficient way to communicate information is (inteligently) colour-coded figures and tables.

The structure page-wise is as follows:
Page:Contents
1: Technical summary
2-4: Summarised signal results
5-End:
  Part 1: Extended results
  Part 2: Extended technicalities
"""

from agora.base import ParametersABC, ProcessABC


class Reporter(ProcessABC):
    """ """

    def __init__(self, parameters: ParametersABC):
        super().__init__(parameters)


class ReporterParameters(ParametersABC):
    """ """

    def __init__(self, technical: dict, results: dict, grouper_kws: dict):
        super().__init__()
