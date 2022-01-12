import numpy as np
from matplotlib.colors import hsv_to_rgb
from sklearn.decomposition import PCA


def xy2rgb(xy, method='rt'):
    if method == 'rt':
        angles = (np.arctan2(xy[:, 1], xy[:, 0]) + np.pi) / np.pi / 2
        r = np.hypot(xy[:, 0], xy[:, 1])
        r = r / r.max()
        hsv = np.vstack([angles, r, np.ones_like(r) * 0.9]).T
        c = hsv_to_rgb(hsv)
    elif method in ['0', '1', 0, 1]:
        i = int(method)
        xmin, xmax = xy[:, i].max(), xy[:, i].min()
        x = (xy[:, i] - xmin) / (xmax - xmin)
        hsv = np.vstack([x, np.ones_like(x) * 0.5, np.ones_like(x) * 0.9]).T
        c = hsv_to_rgb(hsv)
    else:
        i = 0
        xmin, xmax = xy[:, i].max(), xy[:, i].min()
        x = (xy[:, i] - xmin) / (xmax - xmin)
        hsv = np.vstack([x, np.ones_like(x) * 0.5, np.ones_like(x) * 0.9]).T
        c = hsv_to_rgb(hsv)

    return c

def fftpca(ps):
    # fft
    aa = np.log(np.abs(np.fft.fft2(ps)) + 1)
    aa[:, 0, 0] = 0
    # pca
    pca = PCA(n_components=2)
    xy = pca.fit_transform(aa.reshape(-1, ps.shape[1] * ps.shape[2]))
    return xy


