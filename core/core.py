"""Barebones implementation of the structure/organisation of experiments."""


class Experiment:
    def __init__(self):
        self.strains = dict()
        self._metadata = None

    def add_strains(self, name, strain):
        self.strains[name] = strain


class Strain:
    def __init__(self):
        self.positions = dict()

    def add_position(self, name, position):
        self.positions[name] = position


class Position:
    def __init__(self):
        self.traps = []

    def add_trap(self, trap):
        self.traps.append(trap)


class Trap:  # TODO Name this Tile?
    def __init__(self):
        self.cells = []

    def add_cell(self, cell):
        self.cells.append(cell)
