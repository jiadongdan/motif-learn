import numpy as np
import matplotlib.pyplot as plt

from skimage import filters
from sklearn.neighbors import NearestNeighbors
from scipy.optimize import least_squares
from sklearn.utils import check_random_state


# Two-dimensional Gaussian function fit
def multi_gaussians(params, X, Y):
    c = params[-1]
    ps = np.array(params[:-1]).reshape(-1, 4)
    g = np.zeros_like(X)
    for (x0, y0, A, sigma) in ps:
        g = g + A*np.exp(-((X-x0)**2+(Y-y0)**2)/(2*sigma**2))
    return g+c

def init_params(pts, ind, num, data):
    x0, y0 = pts[num]
    pts_selected = pts[ind[num]]
    sigma = estimate_sigma(pts_selected)
    p0 = np.array([(x-x0, y-y0, data[y, x], sigma) for (x, y) in pts_selected])
    p0 = p0.flatten()
    c = np.min(data)
    p0 = np.append(p0, c)
    return p0

def fit_single_patch(p0, y_data, xtol, ftol, gtol):
    size = (y_data.shape[0] - 1)//2
    t = np.arange(-size, size+1)
    X, Y = np.meshgrid(t, t)
    def errorfunction(p0, X=X, Y=Y, y_data=y_data):
        return np.ravel(multi_gaussians(p0, X, Y) - y_data)
    results = least_squares(errorfunction, x0=p0, bounds=(-np.inf, np.inf), xtol=xtol, ftol=ftol, gtol=gtol, verbose=0)
    return results['x']

def critical_radius_otsu(pts, num_bins=256):
    nbrs = NearestNeighbors(n_neighbors=11, algorithm='auto', metric='euclidean')
    nbrs.fit(pts)
    d, ind = nbrs.kneighbors(pts, n_neighbors=11)
    d = d[:, 1:]
    t = filters.threshold_otsu(d, nbins=num_bins)
    return t

def estimate_sigma(pts):
    nbrs = NearestNeighbors(n_neighbors=2, algorithm='auto', metric='euclidean')
    nbrs.fit(pts)
    d, ind = nbrs.kneighbors(pts, n_neighbors=2)
    dist = d[:, 1].mean()
    sigma = dist/5
    return sigma

def radius_nbrs(pts, r):
    nbrs = NearestNeighbors(n_neighbors=11, algorithm='auto', metric='euclidean')
    nbrs.fit(pts)
    d, ind = nbrs.kneighbors(pts, n_neighbors=11)
    mask = d < r
    d = [e[m] for e, m in zip(d, mask)]
    ind = [e[m] for e, m in zip(ind, mask)]
    return d, ind


def nbrs(pts, k=None):
    if k is None:
        r = critical_radius_otsu(pts)
        d, ind = radius_nbrs(pts, r)
    else:
        nbrs = NearestNeighbors(n_neighbors=k+1, algorithm='auto', metric='euclidean')
        nbrs.fit(pts)
        d, ind = nbrs.kneighbors(pts, n_neighbors=k+1)
    return d, ind


# Convert between pts and pts_array
def pts2array(pts, shape):
    pts_array = np.zeros(shape)
    for (x, y) in pts:
        if x < shape[1] and y < shape[0]:
            pts_array[y, x] = 1
    return pts_array

def array2pts(pts_array):
    coords = np.column_stack(np.nonzero(pts_array))
    coords[:, 0], coords[:, 1] = coords[:, 1], coords[:, 0].copy()
    return coords

class KeyPoints:
    def __init__(self, pts, shape, pts_array=None, data=None):
        if pts_array is None:
            self.pts = pts
            self.shape = shape
            self.pts_array = pts2array(pts, shape)
            self.num_pts = pts.shape[0]
        else:
            self.pts_array = pts_array
            self.shape = pts_array.shape
            self.pts = array2pts(pts_array)
            self.num_pts = self.pts.shape[0]
        self.data = data
        self.critical_radius = critical_radius_otsu(pts)
        self.sigma = estimate_sigma(pts)

    def clear_border(self, size):
        for i in range(self.pts_array.ndim):
            self.pts_array = self.pts_array.swapaxes(0, i)
            self.pts_array[:size] = self.pts_array[-size:] = 0
            self.pts_array = self.pts_array.swapaxes(0, i)
        self.pts = array2pts(self.pts_array)
        self.num_pts = self.pts.shape[0]

    def set_data(self, data):
        self.data = data

    def extract_patches(self, size, flat=False, max_patches=None, random_state=None):
        # Make sure patch_size is odd number
        patch_size = 2*size + 1
        # Clear border to avoid out of index error
        self.clear_border(size)
        if max_patches and max_patches < len(self.pts):
            rng = check_random_state(random_state)
            ind = rng.randint(len(self.pts), size=max_patches)
            pts = self.pts[ind]
        else:
            pts = self.pts.copy()
        if flat == True:
            self.patches = np.array([self.data[y-size:y+size+1, x-size:x+size+1].flatten() for (x, y) in pts])

        else:
            self.patches = np.array([self.data[y-size:y+size+1, x-size:x+size+1] for (x, y) in pts])
        return self.patches

    def fit_gaussians(self, xtol=1e-8, ftol=1e-8, gtol=1e-8):
        d, ind = nbrs(self.pts)
        params = []
        for i, (x, y) in enumerate(self.pts):
            p0 = init_params(self.pts, ind, i, self.data)
            patch = self.patches
            p = fit_single_patch(p0, self.patches[i], xtol=xtol, ftol=ftol, gtol=gtol)
            params.append([x+p[0], y+p[1], p[2], p[3], p[-1]])
            if i%10 == 0:
                print(i, end=',')
        return params



