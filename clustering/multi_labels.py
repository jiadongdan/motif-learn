import numpy as np

class lbsarray(np.ndarray):

    def __new__(cls, input_array, n=None, m=None, lbs=None):
        # Input array is an already formed ndarray instance
        # We first cast to be our class type
        obj = np.asarray(input_array).view(cls)
        return obj

    @property
    def levels(self):
        return np.unique(self[:, -1])

    def add_lbs(self, lbs):
        pass

    def select(self, level=0):
        ind = np.where(self[:, -1])[0]
        pass

