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
    """
    An array of points in two-dimensional space, a subclass of :py:class:`numpy.ndarray`.

    Parameters
    ----------
    pts: array_like
        points data which contain `x` and `y` information.
    u: array_like, optional
        basis vector `u` in the context of periodic lattice
    v: array_like, optional
        basis vector `v` in the context of periodic lattice
    Returns
    -------
    obj: :py:class:`pts_array`
    Notes
    -----
    """
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
        """
        the length values of the points from the origin.
        """
        return np.hypot(self.x, self.y)

    @property
    def angle(self):
        """
        the angle vlaues of the points in degree.
        """
        return np.rad2deg(np.arctan2(self.y, self.x))

    def clear_border(self, shape, size):
        """
        Clear the border points.
        Parameters
        ----------
        shape

        size

        Returns
        -------

        """
        a = _pts2array(self, shape)
        for i in range(a.ndim):
            a = a.swapaxes(0, i)
            a[:size] = a[-size:] = 0
            a = a.swapaxes(0, i)
        pts = _array2pts(a)
        return ptsarray(pts)

    def clear(self, delta=0.1):
        """
        Clear non-lattice points.
        Parameters
        ----------
        delta

        Returns
        -------

        """
        uv = np.vstack([self.u, self.v])
        pq = self.dot(np.linalg.inv(uv))
        dist = np.round(pq) - pq
        r = np.hypot(dist[:, 0], dist[:, 1])
        return ptsarray(self[r < delta], u=self.u, v=self.v)

    def extract_patches(self, img, size):
        """
        Extract patches from the location points.

        Parameters
        ----------
        img: `2d-array`
            the image to be extracted.
        size: int
            the shape of single patch is `(2*size+1)`.
        Returns
        -------
        ps: `3d-array`
            the extracted patches in stack.
        """
        # clear borders
        pts = self.clear_border(shape=img.shape, size=size)
        ps = np.array([img[y-size:y+size+1, x-size:x+size] for (x, y) in pts])
        return ps

    def shift(self, x, y):
        """
        Shift the point by giving `x` and `y`.
        Parameters
        ----------
        x: float

        y: float

        Returns
        -------

        """
        pts_ = np.stack([self.x - x, self.y - y]).T
        return ptsarray(pts_, u=self.u, v=self.v)

    def generate_mask(self, size):
        pass

