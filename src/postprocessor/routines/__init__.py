"""Routines for analysing post-processed data that don't follow the parameters-processes structure.

Routines for analysing post-processed data that don't follow the
parameters-processes structure.

Currently, these consist of plotting routines.  There is one module for each
plotting routine.  Each module consists of two components and is structured as
follows:
1. An internal class.
    The class defines the parameters and defines additional class attributes to
    help with plotting.  The class also has one method (`plot`) that takes a
    `matplotlib.Axes` object as an argument.  This method draws the plot on the
    `Axes` object.
2. A plotting function.
    The user accesses this function.  This function defines the default
    parameters in its arguments.  Within the function, a 'plotter' object is
    defined using the internal class and then the function draws the plot on a
    specified `matplotlib.Axes` object.

This structure follows that of plotting routines in `seaborn`
(https://github.com/mwaskom/seaborn), a Python visualisation library that is
based on `matplotlib`.
"""
