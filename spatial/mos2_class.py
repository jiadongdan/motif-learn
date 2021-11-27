import numbers
import numpy as np
import numpy.ma as ma
import matplotlib.pyplot as plt
from sklearn.neighbors import KDTree

from ..plot.colormaps import color_palette

chemical_symbols = [
    # 0
    'X',
    # 1
    'H', 'He',
    # 2
    'Li', 'Be', 'B', 'C', 'N', 'O', 'F', 'Ne',
    # 3
    'Na', 'Mg', 'Al', 'Si', 'P', 'S', 'Cl', 'Ar',
    # 4
    'K', 'Ca', 'Sc', 'Ti', 'V', 'Cr', 'Mn', 'Fe', 'Co', 'Ni', 'Cu', 'Zn',
    'Ga', 'Ge', 'As', 'Se', 'Br', 'Kr',
    # 5
    'Rb', 'Sr', 'Y', 'Zr', 'Nb', 'Mo', 'Tc', 'Ru', 'Rh', 'Pd', 'Ag', 'Cd',
    'In', 'Sn', 'Sb', 'Te', 'I', 'Xe',
    # 6
    'Cs', 'Ba', 'La', 'Ce', 'Pr', 'Nd', 'Pm', 'Sm', 'Eu', 'Gd', 'Tb', 'Dy',
    'Ho', 'Er', 'Tm', 'Yb', 'Lu',
    'Hf', 'Ta', 'W', 'Re', 'Os', 'Ir', 'Pt', 'Au', 'Hg', 'Tl', 'Pb', 'Bi',
    'Po', 'At', 'Rn',
    # 7
    'Fr', 'Ra', 'Ac', 'Th', 'Pa', 'U', 'Np', 'Pu', 'Am', 'Cm', 'Bk',
    'Cf', 'Es', 'Fm', 'Md', 'No', 'Lr',
    'Rf', 'Db', 'Sg', 'Bh', 'Hs', 'Mt', 'Ds', 'Rg', 'Cn', 'Nh', 'Fl', 'Mc',
    'Lv', 'Ts', 'Og']

atomic_numbers = {}
for Z, symbol in enumerate(chemical_symbols):
    atomic_numbers[symbol] = Z

def split_e_num(s):
    head = s.rstrip('0123456789')
    tail = s[len(head):]
    if tail == '':
        tail = -1
    else:
        tail = int(tail)
    return head, tail

def element2num(s):
    e, num = split_e_num(s)
    if num == -1:
        return atomic_numbers[e]
    else:
        n = np.floor(np.log10(num)+1)
        return atomic_numbers[e] + num/(10**n)

def num2element(num):
    s = str(num).split('.')
    if len(s) == 1:
        return chemical_symbols[int(s[0])]
    else:
        n = int(s[0])
        f = int(s[1])
        if f == 0:
            return chemical_symbols[n]
        else:
            return chemical_symbols[n] + str(f)


class Atoms(ma.MaskedArray):

    def __new__(cls, xyz, lbs):
        xyz = np.asarray(xyz)
        if len(xyz.shape) == 1:
            xyz = xyz[:, np.newaxis]
        lbs = np.asarray(lbs)[:, np.newaxis]
        xyzl = np.hstack([xyz, lbs])
        mask = np.zeros_like(xyzl)
        mask[:, 3] = 1

        obj = np.ma.array(xyzl, mask=mask).view(cls)

        return obj

    @property
    def lbs(self):
        return self.data[:, 3]


    @property
    def x(self):
        return self[:, 0].view(np.ndarray)

    @property
    def y(self):
        return self[:, 1].view(np.ndarray)

    @property
    def z(self):
        return self[:, 2].view(np.ndarray)


    def add(self, atoms):
        if isinstance(atoms, tuple):
            atoms = list(atoms)
        elif isinstance(atoms, np.ndarray):
            atoms = [atoms]
        atoms_new = np.vstack([self]+atoms)
        mask = np.zeros_like(atoms_new)
        mask[:, 3] = 1
        atoms_new.mask = mask

        return atoms_new

    def center_around(self, atom):
        atom = list(np.asarray(atom).ravel())
        atom.append(0)
        return self - atom

    def select(self, elements):
        if isinstance(elements, str):
            elements = [elements]
        if isinstance(elements, numbers.Number):
            elements = [elements]
        s = np.array([element2num(e) if isinstance(e, str) else e for e in elements])
        mask = np.in1d(self.lbs, s)
        return self[mask]

    def query(self, atoms, k=3, leaf_size=40, metric='minkowski', return_distance=True):
        kdtree = KDTree(self[:, 0:3], leaf_size=leaf_size, metric=metric)
        return  kdtree.query(atoms[:, 0:3], k=k, return_distance=return_distance)


    def query_radius(self, atoms, r=10, leaf_size=40, metric='minkowski', return_distance=True):
        kdtree = KDTree(self[:, 0:3], leaf_size=leaf_size, metric=metric)
        return  kdtree.query_radius(atoms[:, 0:3], r=r, return_distance=return_distance)

class MoS2(ma.MaskedArray):

    def __new__(cls, xyz, lbs):
        xyz = np.asarray(xyz)
        if len(xyz.shape) == 1:
            xyz = xyz[:, np.newaxis]
        lbs = np.asarray(lbs)[:, np.newaxis]
        xyzl = np.hstack([xyz, lbs])
        # mask the labels
        mask = np.zeros_like(xyzl)
        mask[:, 3] = 1

        obj = np.ma.array(xyzl, mask=mask).view(cls)

        return obj

    @property
    def lbs(self):
        return self.data[:, 3]


    @property
    def x(self):
        return self[:, 0].view(np.ndarray)

    @property
    def y(self):
        return self[:, 1].view(np.ndarray)

    @property
    def z(self):
        return self[:, 2].view(np.ndarray)

    @property
    def mo(self):
        lbs_int = self.lbs.astype(np.int)
        mask = np.in1d(lbs_int, 42)
        return self[mask]

    @property
    def s(self):
        lbs_int = self.lbs.astype(np.int)
        mask = np.in1d(lbs_int, 16)
        return self[mask]

    @property
    def s1(self):
        z = self.s[:, 2]
        z_mean = z.mean()
        mask = z > z_mean
        return self.s[mask]


    @property
    def s2(self):
        z = self.s[:, 2]
        z_mean = z.mean()
        mask = z < z_mean
        return self.s[mask]


#=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
# MoS2 model
#=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=


def rot_matrix(angle=30):
    theta = np.radians(angle)
    c, s = np.cos(theta), np.sin(theta)
    R = np.array(((c, -s), (s, c)))
    return process_zeros(R)


def process_zeros(data, eps=1e-9):
    data[np.abs(data) < eps] = 0.0
    return data


def _generate_lattice(u, v, n, m=None, z=0., dx=0, dy=0, element='X', clip=False):
    if m is None:
        m = n
    uv = np.stack([u, v])
    X, Y = np.meshgrid(range(-n, n), range(-m, m))
    xy = np.dstack([X, Y])
    xy = xy.reshape(-1, 2).dot(uv)
    xy = xy + [dx, dy]
    xyz = np.column_stack([xy, [z] * len(xy)])

    d2 = X ** 2 + Y ** 2 + X * Y
    mask = d2 <= np.unique(d2)[n]
    xyz = xyz[mask.ravel()]

    l = 2 * np.sqrt(np.unique(d2)[n])
    if clip:
        mask1 = np.logical_and(np.abs(xyz[:, 0]) <= l, np.abs(xyz[:, 1]) <= l)
        xyz = xyz[mask1]

    lbs = [element] * len(xyz)
    return xyz, lbs

def mos2(n, m=None, z=0., theta=0., center='Mo', clip=False, elements=('Mo', 'S1', 'S2')):
    a = 3.12
    u = np.array([a, 0])
    v = np.array([a / 2, np.sqrt(3) / 2 * a])
    R = rot_matrix(theta)
    u = R.dot(u)
    v = R.dot(v)

    dx, dy = (u + v) / 3
    mo, s1, s2 = elements
    if center == 'Mo':
        Mo, lbs1 = _generate_lattice(u, v, n, m, z=z, element=mo, clip=clip)
        S1, lbs2 = _generate_lattice(u, v, n, m, z=-a / 2 + z, dx=dx, dy=dy, element=s2, clip=clip)
        S2, lbs3 = _generate_lattice(u, v, n, m, z=a / 2 + z, dx=dx, dy=dy, element=s1, clip=clip)
    elif center == 'S':
        S1, lbs2 = _generate_lattice(u, v, n, m, z=a / 2 + z, element=s1, clip=clip)
        S2, lbs3 = _generate_lattice(u, v, n, m, z=-a / 2 + z, element=s2, clip=clip)
        Mo, lbs1 = _generate_lattice(u, v, n, m, z=z, dx=dx, dy=dy, element=mo, clip=clip)
    xyz = np.vstack([Mo, S1, S2])
    lbs = np.array(lbs1 + lbs2 + lbs3)
    lbs_num = np.array([element2num(e) for e in lbs])

    return MoS2(xyz, lbs_num)


def _bilayer_mos2(n, m=None, theta=0, clip=True, elements=('Mo', 'S1', 'S2'), mode='mos'):
    def _validate_mode(mode):
        valid_modes = ['momo', 'mos', 'ss']
        if mode not in valid_modes:
            raise ValueError('valid modes must be one be \'momo\', \'mos\', \'ss\'')

    _validate_mode(mode)
    if mode == 'momo':
        center1, center2 = 'Mo', 'Mo'
    elif mode == 'mos':
        center1, center2 = 'Mo', 'S'
    elif mode == 'ss':
        center1, center2 = 'S', 'S'

    c = 12.32
    layer1 = mos2(n, m, z=-c/4, theta=0, center=center1, clip=clip, elements=elements)
    layer2 = mos2(n, m, z=c/4, theta=theta, center=center2, clip=clip, elements=elements)
    return layer1, layer2


#=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
# bilayer class
#=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=

from skimage.filters import gaussian
# this only approximates
def get_layer_image(mos2, shape=(1024, 1024), ratio=0.6, sigma=3, scale=None):
    dy, dx = shape[0] // 2, shape[1] // 2
    if scale is None:
        scale = dx/(mos2.max()+1)
    a = np.zeros(shape, dtype=np.float)
    for (x, y) in mos2.mo[:, 0:2]:
        a[int(y*scale)+dx, int(x*scale)+dy] = 1.
    for (x, y) in mos2.s1[:, 0:2]:
        a[int(y*scale)+dx, int(x*scale)+dy] = 1*ratio
    #for (x, y) in mos2.s2[:, 0:2]:
    #    a[int(y*scale)+dx, int(x*scale)+dy] = 1*ratio
    layer_image = gaussian(a, sigma)
    return layer_image/layer_image.max()


def gray2color(img, cmap):
    img = img/img.max()
    return cmap(img)

class BilayerMoS2:

    def __init__(self, n, m=None, theta=0, clip=True, elements=('Mo', 'S1', 'S2'), mode='mos'):
        self.layer1, self.layer2 = _bilayer_mos2(n, m, theta, clip, elements, mode)


    def to_image(self, shape=512, ratio=0.6, sigma=None, scale=None):
        if isinstance(shape, numbers.Number):
            shape = (shape, shape)

        dy, dx = shape[0] // 2, shape[1] // 2
        if scale is None:
            a = 3.12
            l1 = (int(self.layer1.max()/a)+1)*a
            l2 = (int(self.layer2.max()/a)+1)*a
            scale1 = dx / l1
            scale2 = dx / l2
            scale = np.minimum(scale1, scale2)
        if sigma is None:
            sigma = scale/3.20569096160721
        # use the same scale for both layers
        layer1_image = get_layer_image(self.layer1, shape, ratio, sigma, scale)
        layer2_image = get_layer_image(self.layer2, shape, ratio, sigma, scale)
        layer12_image = layer1_image + layer2_image
        return layer12_image

    def to_color_image(self, shape=512, ratio=0.6, sigma=None, scale=None, cmap1=None, cmap2=None):
        if isinstance(shape, numbers.Number):
            shape = (shape, shape)

        dy, dx = shape[0] // 2, shape[1] // 2
        if scale is None:
            a = 3.12
            l1 = (int(self.layer1.max()/a)+1)*a
            l2 = (int(self.layer2.max()/a)+1)*a
            scale1 = dx / l1
            scale2 = dx / l2
            scale = np.minimum(scale1, scale2)
        if sigma is None:
            sigma = scale/3.20569096160721
        # use the same scale for both layers
        layer1_image = get_layer_image(self.layer1, shape, ratio, sigma, scale)
        layer2_image = get_layer_image(self.layer2, shape, ratio, sigma, scale)

        if cmap1 is None:
            cmap1 = color_palette('C0', 0., 0.9)
        if cmap2 is None:
            cmap2 = color_palette('C1', 0., 0.9)
        layer1_image = gray2color(layer1_image, cmap1)
        layer2_image = gray2color(layer2_image, cmap2)
        return (layer1_image + layer2_image)/2

    def plot(self, ax=None, c1='C0', c2='C1', show_mo_only=True):
        s = 3
        if ax is None:
            fig, ax = plt.subplots(1, 1, figsize=(7.2, 7.2))

        if show_mo_only:
            ax.scatter(self.layer1.mo[:, 0], self.layer1.mo[:, 1], c=c1, s=s)
            ax.scatter(self.layer2.mo[:, 0], self.layer2.mo[:, 1], c=c2, s=s)
        else:
            ax.scatter(self.layer1.mo[:, 0], self.layer1.mo[:, 1], c=c1, s=2*s, zorder=-2)
            ax.scatter(self.layer1.s[:, 0], self.layer1.s[:, 1], c=c2, s=s, zorder=-2)

            ax.scatter(self.layer2.mo[:, 0], self.layer2.mo[:, 1], c=c1, s=2*s)
            ax.scatter(self.layer2.s[:, 0], self.layer2.s[:, 1], c=c2, s=s)
