import numpy as np
from scipy.special import binom
from sklearn.utils import check_array


def zp_j2nm(j):
    if not np.isscalar(j):
        j = check_array(j, ensure_2d=False)
    n = (np.ceil((-3 + np.sqrt(9 + 8 * j)) / 2)).astype(np.int)
    m = 2 * j - n * (n + 2)
    return np.array([n, m]).T


def zp_nm2j(n, m):
    j = ((n + 2) * n + m) // 2
    return j


def zp_nm2j_complex(n, m):
    j = np.array(n**2+2*n+2*m)
    mask = np.array(n)%2==0
    j[mask] = j[mask]//4
    j[~mask] = (j[~mask]-1)//4
    return j

def zp_j2nm_complex(j):
    if not np.iterable(j):
        j = np.array([j])
    else:
        j = np.array(j)
    n = np.floor(np.sqrt(4*j+1)-1).astype(np.int)
    mask = n%2 == 0
    m = (4*j - n**2 -2*n + 1)//2
    m[mask] = (4*j[mask] - n[mask]**2 -2*n[mask] + 1)//2
    nm  = np.array([n, m]).T
    if nm.shape[0] == 1:
        nm = nm[0]
    return nm

def zp_nm2j_complex_(n, m):
    if n%2 == 0:
        j = (n//2+1)**2+(m-n)//2 - 1
    else:
        j = ((n+1)//2)*((n+1)//2+1)+(m-n)//2 - 1
    return j


def calculate_relative_angle(m1, m2):
    m_array = zp_j2nm(np.arange(len(m1)))
    m_abs = np.abs(m1) + np.abs(m2)
    m_abs[np.where(m_array==0)[0]] = 0
    ind = np.argmax(m_abs)
    return np.angle(m1[ind]/m2[ind])


def get_j1_j2(N):
    n, m = zp_j2nm(N - 1)
    if m < n:
        n = n - 1
    j1 = zp_nm2j(n, n)
    j2 = zp_nm2j_complex(n, n)
    return j1, j2

def get_m_array(N, mode='real'):
    if mode == 'real':
        m = np.array([abs(zp_j2nm(j)[1]) for j in range(N)])
    elif mode == 'complex':
        m = np.array([zp_j2nm_complex(j)[1] for j in range(N)])
    return m


def grid_rt(size):
    """Get grid array r and t"""
    y, x = np.ogrid[-1:1:1j * size, -1:1:1j * size]
    r = np.sqrt(x * x + y * y)
    t = np.arctan2(y, x)
    return r, t


def zp_calculate_coefficients(n, m):
    """Calculate coefficients from (n, m)"""
    m = np.abs(m)
    assert ((n - m) % 2 == 0)
    num = (n - m) // 2
    coeffs = np.array([np.power(-1, k) * binom(n - k, k) * binom(n - 2 * k, num - k) for k in np.arange(0, num + 1)])
    return coeffs


def zp_calculate_norm(n, m):
    if m == 0:
        norm = np.sqrt(n + 1)
    else:
        norm = np.sqrt(2 * n + 2)
    return norm


def zp_calculate_coefficients_matrix(N, normalize=True):
    """Calculate coefficient matrix"""
    n_max, m = zp_j2nm(N - 1)
    shape = (N, n_max + 1)
    c_matrix = np.zeros(shape)
    for j in range(N):
        n, m = zp_j2nm(j)
        coeffs = zp_calculate_coefficients(n, m)
        if normalize:
            norm = zp_calculate_norm(n, m)
        else:
            norm = 1.
        # different from gpzp case
        c_matrix[j, n_max - n:n_max - n + 2 * len(coeffs):2] = coeffs * norm
    return c_matrix


def zp_calculate_Rn_matrix(N, size):
    n, m = zp_j2nm(N - 1)
    r, t = grid_rt(size)
    mask = (r <= 1) * 1
    r, mask = r.ravel(), mask.ravel()
    Rn_matrix = np.array([np.power(r, i) for i in np.arange(n + 1)[::-1]])
    return Rn_matrix * mask


def zp_calculate_mt_matrix(N, size):
    nm = zp_j2nm(range(N))
    n, m = nm[:, 0], nm[:, 1]
    t = grid_rt(size)[1].ravel()
    f = np.array([np.sin, np.cos])
    mask = (m >= 0) * 1
    funcs = f[mask]
    return np.array([func(e * t) for e, func in zip(np.abs(m), funcs)])


def generate_zps(N, size):
    c_matrix = zp_calculate_coefficients_matrix(N)
    Rn_matrix = zp_calculate_Rn_matrix(N, size)
    Rnm_ = c_matrix.dot(Rn_matrix)
    mt_matrix = zp_calculate_mt_matrix(N, size)
    return (Rnm_ * mt_matrix).reshape((N, size, size))


def Rnm_r(r, n, m):
    coeffs = zp_calculate_coefficients(n, m)
    norm = zp_calculate_norm(n, m)
    Rnm = np.zeros_like(r)
    for k, c in enumerate(coeffs):
        Rnm = Rnm + c * np.power(r, n - k)
    return Rnm * norm


def zp_nm(n, m, size):
    r, t = grid_rt(size)
    mask = (r <= 1) * 1
    Rnm = Rnm_r(r, n, m)
    if m >= 0:
        tt = np.cos(m * t)
    else:
        tt = np.sin(-m * t)
    zp = Rnm * tt
    return zp * mask


def zp_j(j, size):
    n, m = zp_j2nm(j)
    return zp_nm(n, m, size)

def zp_calculate_rot_matrix(n_zps, theta):
    rot_matrix = np.zeros(shape=(n_zps, n_zps))
    theta = np.deg2rad(theta)
    for j, row in enumerate(rot_matrix):
        n, m = zp_j2nm(j)
        j1 = zp_nm2j(n, -abs(m))
        j2 = zp_nm2j(n, abs(m))
        if m == 0:
            rot_matrix[j, j1] = 1.0
        elif m > 0:
            rot_matrix[j, j1] = np.cos(m*theta)
            rot_matrix[j, j2] = -np.sin(m*theta)
        elif m < 0:
            rot_matrix[j, j1] = np.sin(-m*theta)
            rot_matrix[j, j2] = np.cos(-m*theta)
    return rot_matrix


def calculate_complex_matrix(num_cols):
    # Calculate num_rows from num_cols
    n, m = zp_j2nm(num_cols - 1)
    assert (n == m)
    if n % 2 == 0:
        num_rows = (n // 2 + 1) ** 2
    else:
        num_rows = ((n + 1) // 2 + 1) * ((n + 1) // 2)
    c_matrix = np.zeros(shape=(num_rows, num_cols), dtype=np.complex)
    for i in np.arange(num_rows):
        n, m = zp_j2nm_complex(i)
        j1, j2 = zp_nm2j(n, -m), zp_nm2j(n, m)
        if j1 == j2:
            c_matrix[i, j1] = 1
        else:
            c_matrix[i, j1] = 1
            c_matrix[i, j2] = 1j
    return c_matrix

def get_complex_matrix(num_rows, num_cols):
    c_matrix = np.zeros(shape=(num_rows, num_cols), dtype=np.complex)
    for i in np.arange(num_rows):
        n, m = zp_j2nm_complex(i)
        j1, j2 = zp_nm2j(n, -m), zp_nm2j(n, m)
        if j1 == j2:
            c_matrix[i, j1] = 1
        else:
            c_matrix[i, j1] = 1
            c_matrix[i, j2] = 1j
    return c_matrix



def get_orientation(moment_complex, n_fold):
    a = np.rad2deg(np.angle(moment_complex))
    ind= np.array([zp_j2nm_complex(j)[1] for j in range(len(a))])
    mask = ind == n_fold
    angles = a[mask]/ind[mask]
    l = []
    if n_fold == 5:
        for i, e in enumerate(angles):
            if i%2 == 0:
                l.append(e)
            elif i%2 == 1:
                v1, v2 = e+36, e-36
                j = np.argmin([abs(v1), abs(v2)])
                v = [v1, v2][j]
                l.append(v)
    return np.array(l).mean()

class ZPs:

    def __init__(self, N, size):
        self.num_zps = N
        self.num_components = N
        self.size = size
        self.data = generate_zps(N, size).reshape((-1, size, size))
        self.ind = np.arange(self.num_zps)
        self.data_selected = self.data[self.ind]
        self.nm = zp_j2nm(np.arange(N))

        self.j = N - 1
        self.j1, self.j2 = get_j1_j2(N)
        self.m_array = zp_j2nm(np.arange(N))[:, 1]
        self.m1_array = zp_j2nm(np.arange(self.j1+1))[:, 1]
        self.m2_array = zp_j2nm_complex(np.arange(self.j2+1))[:, 1]

        self.complex_matrix = get_complex_matrix(self.j2+1, self.j1+1)

        self.moments = None
        self.moments_rot = None
        self.moments_complex = None
        self.moments_abs = None
        self.n_folds = None
        self.angles = None

        self.symm_states = None

    def set_symm_states(self, states):
        self.reset_symm_states()
        if not np.iterable(states):
            states = [states]
        self.ind = np.hstack([self.ind[self.m_array==s] for s in states])
        self.ind.sort()
        self.data_selected = self.data[self.ind]

    def reset_symm_states(self):
        self.ind = np.arange(self.num_zps)
        self.data_selected = self.data[self.ind]

    def extract_moments(self, patches):
        zps = self.data
        if len(patches.shape) == 2:
            patches = patches[np.newaxis, :, :]
        s = patches.shape[1]
        num_patches = patches.shape[0]
        m = np.array([(p * zp).sum() for p in patches for zp in zps]).reshape((num_patches, self.num_zps))
        m = m / (s * s / 4)
        if m.shape[0] == 1:
            m = m[0]
        return m

    def extract_n_folds_v1(self):
        tt = np.vstack([(self.moments[:, np.where(np.abs(self.m_array)==fold)[0]]**2).mean(axis=1) for fold in np.arange(2, 9)]).T
        ind = np.argmax(tt, axis=1)
        self.n_folds = ind + 2

    # Version 2 is better in experiment data
    def extract_n_folds_v2(self):
        m2 = self.moments ** 2
        l = []
        for fold in range(2, 9):
            t = np.array([2 * np.pi / fold * i for i in range(fold)])
            m = zp_j2nm(np.arange(0, self.num_components))[:, 1]
            T, M = np.meshgrid(t, m)
            sin2 = (np.sin(M * T / 2)) ** 2
            m2sin2 = m2.dot(sin2).mean(axis=1)
            l.append(m2sin2)
        self.n_folds = np.argmin(np.array(l).T, axis=1) + 2

    def extract_angles(self):
        if self.moments_abs is not None:
            ind = np.argmax(self.moments_abs, axis=1)
            self.angles = np.array([np.angle(self.moments_complex[i, ind[i]], deg=True) for i in range(len(ind))])

    def fit(self, patches):
        self.moments = self.extract_moments(patches)
        self.moments_rot = self.moments[:, 0: self.j1+1]
        self.moments_complex = self.complex_matrix.dot(self.moments_rot.T).T
        self.moments_abs = np.abs(self.moments_complex)
        self.moments_abs[:, self.m2_array==0] = 0
        self.extract_n_folds_v2()
        #self.extract_angles()

    def extract_moments_rot(self, patches):
        self.reset_symm_states()
        n, m = self.nm[-1]
        if n > abs(m):
            n = n - 1
        moments = self.extract_moments(patches)
        l = []
        num_zps = (n * n + 3 * n) // 2
        for j in np.arange(0, num_zps):
            n, m = zp_j2nm(j)
            if m == 0:
                l.append(np.abs(moments[:, j]))
            elif m > 0:
                j_ = (n * (n + 2) - m) // 2
                mm = np.sqrt(moments[:, j] ** 2 + moments[:, j_] ** 2)
                l.append(mm)
        return np.column_stack(l)


    def estimate_symm(self, patches):
        moments = self.extract_moments(patches)
        m2 = moments ** 2
        t = np.linspace(0, 2 * np.pi, 100)
        m = zp_j2nm(np.arange(0, self.num_components))[:, 1]
        T, M = np.meshgrid(t, m)
        sin2 = (np.sin(M * T / 2)) ** 2
        return m2.dot(sin2)

    def reconstruct(self):
        moments_selected = self.moments[:, self.ind]
        n = moments_selected.shape[0]
        num_zps_selected = len(self.ind)
        zps = self.data_selected.reshape(num_zps_selected, -1)
        reconst = moments_selected.dot(zps)
        return reconst.reshape(n, self.size, self.size)


def rotation_matrix(angle):
    angle = np.radians(angle)
    s = np.sin(angle)
    c = np.cos(angle)
    R = np.array([(c, -s), (s, c)])
    return process_zeros(R)


def process_zeros(data, eps=1e-9):
    data[np.abs(data) < eps] = 0.0
    return data


def rotate(pts, angle):
    pts = np.array(pts)
    R = rotation_matrix(angle)
    return np.dot(pts, R)


def many_gaussians(pts, sigma, s):
    data = np.zeros((s, s))
    Y, X = np.ogrid[-s // 2:s // 2:1j * s, -s // 2:s // 2:1j * s]
    for (x, y) in pts:
        data = data + np.exp((-(X - x) ** 2 - (Y - y) ** 2) / (2 * sigma * sigma))
    return data


def g3(size=128):
    p = (0, size/4)
    sigma = size/20
    pts = np.array([rotate(p, i) for i in np.arange(0, 360, 120)])
    return many_gaussians(pts, sigma, size)


def gg(size=128):
    p = (0, size/4)
    sigma = size/20
    pts = np.array([rotate(p, i) for i in np.arange(0, 360, 120)]+[(0, -35), (0, 35)])
    return many_gaussians(pts, sigma, size)

def gaussians(n_fold, sigma, l, theta, size):
    p0 = np.array([0, l])
    p0 = rotate(p0, theta)
    pts = np.array([rotate(p0, 360*i/n_fold) for i in range(n_fold)])
    return many_gaussians(pts, sigma, size)