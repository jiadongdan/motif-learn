import numbers
import numpy as np
import numpy.ma as ma
from sklearn.neighbors import KDTree

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
        pass

    @property
    def s(self):
        pass

    @property
    def s1(self):
        pass

    @property
    def s2(self):
        pass


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

def mos2(n, m=None, z=0, theta=0, center='Mo', clip=False, elements=('Mo', 'S1', 'S2')):
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

    return Atoms(xyz, lbs_num)

