import numpy as np

def process_zeros(data, eps=1e-9):
    data[np.abs(data) < eps] = 0.0
    return data


def rotation_matrix(angle):
    angle = np.radians(angle)
    s = np.sin(angle)
    c = np.cos(angle)
    R = np.array([(c, -s), (s, c)])
    return process_zeros(R)


def rotate_pts(pts, angle):
    pts = np.array(pts)
    R = rotation_matrix(angle)
    return np.dot(pts, R)


# Convert between pts and pts_array
def _pts2array(pts, shape):
    pts_array = np.zeros(shape)
    for (x, y) in pts.astype(np.int):
        if x < shape[1] and y < shape[0]:
            pts_array[y, x] = 1
    return pts_array

def _array2pts(pts_array):
    coords = np.column_stack(np.nonzero(pts_array))
    coords[:, 0], coords[:, 1] = coords[:, 1], coords[:, 0].copy()
    return coords


class ptsarray(np.ndarray):

    def __new__(cls, pts, u=None, v=None):
        # Input array is an already formed ndarray instance
        # We first cast to be our class type
        obj = np.asarray(pts).view(cls)
        if u is None:
            obj.u = u
        else:
            obj.u = np.array(u)
        if v is None:
            obj.v = v
        else:
            obj.u = np.array(u)
        obj.x = pts[:, 0]
        obj.y = pts[:, 1]

        return obj

    @property
    def length(self):
        return np.hypot(self.x, self.y)

    @property
    def angle(self):
        return np.rad2deg(np.arctan2(self.y, self.x))

    def clear_border(self, shape, size):
        a = _pts2array(self, shape)
        for i in range(a.ndim):
            a = a.swapaxes(0, i)
            a[:size] = a[-size:] = 0
            a = a.swapaxes(0, i)
        pts = _array2pts(a)
        return ptsarray(pts)

    def clear(self, delta=0.1):
        uv = np.vstack([self.u, self.v])
        pq = self.dot(np.linalg.inv(uv))
        dist = np.round(pq) - pq
        r = np.hypot(dist[:, 0], dist[:, 1])
        return ptsarray(self[r < delta], u=self.u, v=self.v)

    def extract_patches(self, img, size):
        # clear borders
        pts = self.clear_border(shape=img.shape, size=size)
        ps = np.array([img[y-size:y+size+1, x-size:x+size] for (x, y) in pts])
        return ps

    def generate_mask(self, size):
        pass

