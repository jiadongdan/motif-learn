import numpy as np
from scipy.sparse import lil_matrix, coo_matrix, csr_matrix, issparse
from scipy.sparse.linalg import norm
from scipy.sparse.csgraph import connected_components



def matrix2edges(matrix):
    coo = coo_matrix(matrix)
    ijs = np.array([coo.row, coo.col]).T
    return ijs


def matrix2ijs(matrix):
    coo = coo_matrix(matrix)
    ijs = np.array([coo.row, coo.col]).T
    return ijs


# convert to list of list
def matrix2lil(matrix):
    return lil_matrix(matrix).rows


def matrix2inds(matrix):
    return lil_matrix(matrix).rows


def edges2matrix(ijs, shape=None, fmt='coo'):
    rows, cols = ijs.T
    data = np.ones_like(rows)
    if shape is None:
        ijs_max = int(ijs.max() + 1)
        shape = (ijs_max, ijs_max)
    matrix = coo_matrix((data, (rows, cols)), shape=shape)
    if fmt == 'csr':
        matrix = matrix.tocsr()
    elif fmt == 'csc':
        matrix = matrix.tocsc()
    elif fmt == 'lil':
        matrix = matrix.tolil()
    elif fmt == 'dense':
        # make it np.ndarray instead of np.matrix
        matrix = np.array(matrix.todense())
    return matrix


def ijs2matrix(ijs, shape=None):
    rows, cols = ijs.T
    data = np.ones_like(rows)
    if shape is None:
        ijs_max = int(ijs.max() + 1)
        shape = (ijs_max, ijs_max)
    matrix = coo_matrix((data, (rows, cols)), shape=shape)
    return matrix


def is_symmetric(matrix):
    if issparse(matrix):
        return norm(matrix - matrix.T) == 0
    else:
        return (matrix == matrix.T).all()


def make_symmetric(matrix):
    if issparse(matrix):
        # convert to lil_matrix, faster
        matrix = lil_matrix(matrix)
        i, j = matrix.nonzero()
        matrix[j, i] = matrix[i, j]
        # convert to csr format
        matrix = csr_matrix(matrix)
    else:
        matrix = np.maximum(matrix.T, matrix)
    return matrix

# make_symmetric_more, make_symmetric_less return symmetric matrix with entries with only 0 and 1
def make_symmetric_more(matrix):
    # convert to matrix with values with 0 and 1
    matrix = (matrix > 0) * 1
    matrix = ((matrix + matrix.T) / 2 > 0) * 1
    return matrix


def make_symmetric_less(matrix):
    # convert to matrix with values with 0 and 1
    matrix = (matrix > 0) * 1
    matrix = ((matrix + matrix.T) / 2 > 0.5) * 1
    return matrix

def get_num_faces_from_matrix(matrix):
    matrix = make_symmetric(matrix)
    ijs = matrix2ijs(matrix)
    # e - number of  edges
    e = len(ijs)//2
    # v - number of nodes
    v = matrix.shape[0]
    return e - v + 1

def get_num_faces(matrix):
    n_components, component_lbs = connected_components(matrix, directed=False)

    if n_components == 1:
        faces = get_num_faces_from_matrix(matrix)
    else:
        faces = []
        for e in np.unique(component_lbs):
            inds = np.where(component_lbs == e)[0]
            # subgraphs
            m = matrix[:, inds][inds, :]
            faces.append(get_num_faces_from_matrix(m))
        faces = np.sum(faces)
    return faces

