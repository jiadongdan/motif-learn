import numpy as np
from scipy.sparse import csgraph


class Graph:

    def __init__(self, matrix, lbs):
        self.matrix = matrix
        self.lbs = lbs


    @property
    def degrees(self):
        return self.matrix.sum(axis=0)

    @property
    def adj_list(self):
        return 0

    @property
    def csr_matrix(self):
        return csgraph.csgraph_from_dense(self.matrix)

    def remove_nodes(self, lbs):
        # need to update self.matrix and self.lbs
        # graph also shrinks to a smaller size
        ind = np.where(np.in1d(self.lbs, lbs) == True)[0]
        self.matrix = np.delete(self.matrix, ind, axis=0)
        self.matrix = np.delete(self.matrix, ind, axis=1)
        self.lbs = np.delete(self.lbs, ind)

    def components(self):
        pass

    # Todo
    def randesu(self, size=4, p=1):
        pass


