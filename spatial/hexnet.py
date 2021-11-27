import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import NearestNeighbors
from scipy.spatial.transform import Rotation as R
from scipy.linalg import inv
from sklearn.cluster import KMeans
from scipy.interpolate import griddata



def rotation_matrix(angle):
    return R.from_euler('z', angle, degrees=True).as_matrix()[0:2, 0:2]


def rotate_pts(pts, angle):
    pts = np.array(pts)
    R = rotation_matrix(angle)
    return np.dot(pts, R)


# you must sort then np.roll() are safe to apply
def rotate_and_roll(pts, angle):
    shape = (pts.shape[0] // 7, 7, 2)
    pts_ = rotate_pts(pts, angle).reshape(shape)
    pts_not_roll = pts_[:, 0:1, :]
    pts_need_roll = pts_[:, 1:, :]
    shift = int(angle // 60)
    pts_roll = np.roll(pts_need_roll, -shift, axis=1)
    return np.concatenate([pts_not_roll, pts_roll], axis=1)


# get many hexagons
def get_hexagonal_ind(pts):
    nbrs = NearestNeighbors(n_neighbors=2, algorithm='ball_tree').fit(pts)
    d = nbrs.kneighbors(pts)[0]
    dmax = d[:, 1].mean() + 2

    nbrs = NearestNeighbors(n_neighbors=7, algorithm='ball_tree').fit(pts)
    d, ind = nbrs.kneighbors(pts)
    mask = d < dmax
    idx = np.where(mask.sum(axis=1) == 7)[0]
    ind = ind[idx]
    d = d[idx]
    # sort ind according to angle
    pts_mean = pts[ind].mean(axis=1)
    pts_ = (pts[ind] - pts_mean[:, np.newaxis, :])
    angles = np.rad2deg(np.arctan2(pts_[:, 1:, 1], pts_[:, 1:, 0])) + 180
    angles[angles < 358] = angles[angles < 358] - 360

    sort_ind = np.argsort(angles, axis=1)
    ind = np.array([[e[0]] + e[1:][idx].tolist() for e, idx in zip(ind, sort_ind)])

    return ind


def get_hexagonal_pts(pts, ind):
    # centering
    pts_mean = pts[ind].mean(axis=1)
    pts_ = (pts[ind] - pts_mean[:, np.newaxis, :]).reshape(-1, 2)
    angles = np.linspace(0, 360, 7)[0:6]
    pp = np.vstack([rotate_pts(pts_, a) for a in angles])

    kmeans = KMeans(n_clusters=7, random_state=0).fit(pp)
    lbs = kmeans.labels_
    pp1 = np.array([pp[lbs == e].mean(axis=0) for e in range(7)])
    # sort pp1
    d7 = dist(pp1)
    mask = d7 == d7.min()
    angles6 = angle(pp1[~mask]) + 180
    idx = np.argsort(angles6)
    return np.vstack([pp1[mask], pp1[~mask][idx]])


# this has issue when angles is around 0, so kmeans method is more robust
def get_hexagonal_pts_(pts, ind):
    # centering
    pts_mean = pts[ind].mean(axis=1)
    pts_ = (pts[ind] - pts_mean[:, np.newaxis, :]).reshape(-1, 2)
    angles = np.linspace(0, 360, 7)[0:6]
    pts_rots = np.array([rotate_and_roll(pts_, angle) for angle in angles])
    pts_rots_mean = pts_rots.mean(axis=0)
    return pts_rots_mean


def get_local_angles(pts, ind):
    # centering
    pts_mean = pts[ind].mean(axis=1)
    pts_ = (pts[ind] - pts_mean[:, np.newaxis, :])
    angles = np.rad2deg(np.arctan2(pts_[:, 1:, 1], pts_[:, 1:, 0])) + 180
    angles[angles < 358] = angles[angles < 358] - 360
    angles = angles.min(axis=1)
    idx = ind[:, 0]
    return idx, angles


def get_matrix_from_ab(a, b):
    ax, ay = np.atleast_1d(a)
    bx, by = np.atleast_1d(b)
    matrix = np.array([[ax, ay], [bx, by]])
    matrix_inv = inv(matrix)
    return matrix_inv


def get_reference_hexagon(pts):
    if isinstance(pts, list):
        refs = []
        for e in pts:
            ind = get_hexagonal_ind(e)
            ref = get_hexagonal_pts(e, ind)
            refs.append(ref)
        refs = np.array(refs)
        return refs.mean(axis=0)
    else:
        ind = get_hexagonal_ind(pts)
        ref = get_hexagonal_pts(pts, ind)
        return ref


def dist(pts):
    pts = np.atleast_2d(pts)
    return np.hypot(pts[:, 0], pts[:, 1])


def angle(pts):
    pts = np.atleast_2d(pts)
    return np.rad2deg(np.arctan2(pts[:, 1], pts[:, 0]))


class HexNet:

    def __init__(self, pts, a=None, b=None, ref=None):
        self.pts_all = pts
        self.inds = get_hexagonal_ind(pts)
        self.pts = pts[self.inds[:, 0]]

        if ref is None:
            self.ref = get_reference_hexagon(pts)
        else:
            self.ref = ref

        if a is None and b is None:
            a = self.ref[1]
            b = self.ref[2]
        self.a = np.atleast_1d(a)
        self.b = np.atleast_1d(b)
        self.matrix = get_matrix_from_ab(a, b)
        self.angles = get_local_angles(pts, self.inds)[1]
        self.mean_angle = 0

        self.exx, self.eyy, self.exy = self.get_strain()

        # https://www.ovito.org/docs/current/reference/pipelines/modifiers/atomic_strain.html#particles-modifiers-atomic-strain
        self.shear_strain = np.sqrt(self.exy ** 2 + 0.5 * (self.exx - self.eyy) ** 2)
        self.volumetric_strain = (self.exx + self.eyy) / 2

    # from The Peak Pairs algorithm for strain mapping from HRTEM images
    def get_strain(self):
        pts1 = self.pts + self.a
        pts2 = self.pts + self.b

        nbrs = NearestNeighbors(n_neighbors=1, algorithm='ball_tree').fit(self.pts_all)
        ind1 = nbrs.kneighbors(pts1, return_distance=False)[:, 0]
        ind2 = nbrs.kneighbors(pts2, return_distance=False)[:, 0]

        u = self.pts_all[ind1] - pts1
        v = self.pts_all[ind2] - pts2

        ux, uy = u[:, 0], u[:, 1]
        vx, vy = v[:, 0], v[:, 1]

        uvx = np.stack([ux, vx])
        uvy = np.stack([uy, vy])

        exx, exy = self.matrix.dot(uvx)
        eyx, eyy = self.matrix.dot(uvy)

        return exx, eyy, (exy+eyx)*0.5

    def show_exx(self, ax=None, n=1024, **kwargs):
        if ax is None:
            fig, ax = plt.subplots(1, 1, figsize=(7.2, 7.2))

        grid_x, grid_y = np.mgrid[0:n:n*1j, 0:n:n*1j]
        gg = griddata(self.pts, self.exx, (grid_x, grid_y), method='cubic')
        ax.imshow(gg.T, **kwargs)

    def show_eyy(self, ax=None, n=1024, **kwargs):
        if ax is None:
            fig, ax = plt.subplots(1, 1, figsize=(7.2, 7.2))

        grid_x, grid_y = np.mgrid[0:n:n*1j, 0:n:n*1j]
        gg = griddata(self.pts, self.eyy, (grid_x, grid_y), method='cubic')
        ax.imshow(gg.T, **kwargs)

    def show_ev(self, ax=None, n=1024, **kwargs):
        if ax is None:
            fig, ax = plt.subplots(1, 1, figsize=(7.2, 7.2))

        grid_x, grid_y = np.mgrid[0:n:n*1j, 0:n:n*1j]
        gg = griddata(self.pts, self.volumetric_strain, (grid_x, grid_y), method='cubic')
        ax.imshow(gg.T, **kwargs)
