"""
A module to extract data from a processed experiment.
"""

class Extraction: 
    """
    A class to extract cell information from a database and store. 
    It can extract cells based on the initialization requirements. 

    For example: 
        * extract all cells from group 1 that are in frame for more than 100
        timepoints (analysis)
        * extract all cells from trap 10 in position one (Visualization)
        * extract all daughter cells 
        * extract all mother cells
    
    The Extraction class can (should) be used as a context so as to correctly
    open/close the storage files in which the information is available. 

    Within the context you can get certain attributes of your cell by default:
        * position within the image/trap
        * number of timepoints available
        * volume
        * outline
    
    Other attributes you have to specify in your extraction:
        * mother/daughter relationship (extract cells and daughters)
        * nucleus (extract cells and nucleus)
        * vacuole (extract cells and vacuole)
    This is because these attributes are relationships between separate
    objects, and need to be queried separately from the database. 

    WIP: in future, it will be possible to nest these contexts.
    """
    def __init__(self, store, db, **kwargs):
        self.store = store
        self.db = db
        self.config = kwargs

    def __enter__(self):
        # TODO open the store and database here
        pass

    def __exit__(self, type, value, traceback):
        # TODO close the store and database session here
        pass
  
    @property
    def cells(self):
        pass


class ExtractedObject: 
    """
    A Wrapper around a database record that links the database to the storage.
    Used as a base for Cell, Vacuole, and Nucleus, objects for easy access to
    common attributes such as location, edge_mask, n_timepoints. 

    :param obj: the database record object
    :param dataset: the dataset (HDF5 or dictionary) where the larger arrays
    are stored.
    """
    def __init__(self, obj, dataset):
        self._obj = obj
        self.dataset = dataset
    
    @property
    def edge_mask(self):
        # TODO return full edge mask
        return

    @property
    def n_timepoints(self):
        # TODO compute number of timepoints and save
        return
    
    @property
    def location(self):
        # TODO return the position of the object within the trap/position (?)
        return


class ExtractedTrap(ExtractedObject):
    """
    A Wrapper around the database Trap record that links the database to the
    storage.
    """
    def __init__(self, trap, dataset):
        super(ExtractedTrap, self).__init__(trap, dataset)

    @property
    def cells(self):
        pass

class ExtractedCell(ExtractedObject): 
    """
    A Wrapper around the database Cell record that links the database to the
    storage. 
    
    :param cell: the database Cell object to get the relationships of that cell
    :param dataset: the dataset (HDF5 or dictionary) where the larger arrays
    are stored.
    """
    def __init__(self, cell, dataset):
        super(ExtractedCell, self).__init__(cell, dataset)

    @property 
    def nucleus(self): 
        # TODO return the nucleus or None if it was not requested
        return

    @property
    def vacuole(self):
        # TODO return the vacuole or None if it was not requested
        return


class ExtractedVacuole(ExtractedObject):
    """
    A Wrapper around the database Vacuole record that links the database to the
    storage.
    """
    def __init__(self, vacuole, dataset):
        #TODO raise a NotImplementedError
        super(ExtractedVacuole, self).__init__(vacuole, dataset)

class ExtractedNucleus(ExtractedObject):
    """
    A Wrapper around the database Nucleus record that links the database to the
    storage.
    """
    def __init__(self, nucleus, dataset):
        #TODO raise a NotImplementedError
        super(ExtractedNucleus, self).__init__(nucleus, dataset)


