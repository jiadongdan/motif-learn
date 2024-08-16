from sklearn.decomposition import PCA

def pca(X, n_components=2, reconstruct=False):
    pca_model = PCA(n_components=n_components)
    X_pca = pca_model.fit_transform(X)
    return X_pca