from sklearn.neighbors import KDTree

def nbrs(X, query_X):
    assert(len(X.shape)==2)
    kdt = KDTree(X)
    d, ind = kdt.query(query_X, k=1)
    ind = ind[:, 0]
    return X[ind]