import numpy as np
from scipy.ndimage import center_of_mass
from sklearn.utils import check_random_state


def disk(radius, dtype=np.uint8):
    L = np.arange(-radius, radius + 1)
    X, Y = np.meshgrid(L, L)
    return np.array((X ** 2 + Y ** 2) <= radius ** 2, dtype=dtype)


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
        if size % 2 == 0:
            s1 = size // 2
            s2 = size // 2
        else:
            s1 = size // 2
            s2 = size // 2 + 1

        # Clear border to avoid out of index error
        self.clear_border(s2)
        if max_patches and max_patches < len(self.pts):
            rng = check_random_state(random_state)
            ind = rng.randint(len(self.pts), size=max_patches)
            pts = self.pts[ind]
        else:
            pts = self.pts.copy()
        if flat == True:
            self.patches = np.array([self.data[y - s1:y + s2, x - s1:x + s2].flatten() for (x, y) in pts])

        else:
            self.patches = np.array([self.data[y - s1:y + s2, x - s1:x + s2] for (x, y) in pts])
        return self.patches

    def optimize(self, r=3, subpix=False):
        x, y = np.indices(disk(r).shape)
        mask = disk(r) == 1
        indxy = np.vstack([x[mask], y[mask]]).T - np.array([r, r])

        pts = []
        for pt in self.pts:
            pt_extended = indxy + pt
            small_patches = np.array([self.data[y - r:y + r + 1, x - r:x + r + 1] for (x, y) in pt_extended])
            lbs_image = np.array([np.ones(small_patches.shape[1:3]) * i for i in range(len(small_patches))])
            xy0 = np.array(center_of_mass(small_patches, lbs_image, index=np.arange(len(small_patches))))[:, 1:]
            ind = np.argmin(np.linalg.norm(xy0 - r, axis=1))
            pts.append(pt_extended[ind])

        self.pts = np.asarray(pts)

        if subpix:
            ps = self.extract_patches(2 * r)
            lbs_image = np.array([np.ones(ps.shape[1:3]) * i for i in range(len(ps))])
            pts = np.array(center_of_mass(ps, labels=lbs_image, index=np.arange(len(ps))))[:, 1:]
            self.pts = (self.pts).astype(float) + np.asarray(pts) - np.array([2 * r, 2 * r]).astype(float)
