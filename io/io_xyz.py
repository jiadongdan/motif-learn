import copy
import numbers
import numpy as np

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

class Atom(object):
    def __init__(self, representation='X', position=None):
        if isinstance(representation, str):
            self.symbol = representation
            self.Z = atomic_numbers[representation]
        elif isinstance(representation, numbers.Number):
            self.symbol = chemical_symbols[representation]
            self.Z = representation
        self.position = position

    def set_position(self, position):
        self.position = position

    def set_atomtype(self, symbol):
        self.symbol = symbol

    def offset(self, dx, dy, dz):
        x, y, z = self.position
        self.position = (x+dx, y+dy, z+dz)

    def __repr__(self):
        s = "Atom('{}') at position {}.".format(self.symbol, self.position)
        return s

class Atoms(object):
    def __init__(self, atoms):
        self.num_atoms = len(atoms)
        self.atoms = np.array(atoms)
        self.atom_types = set([atom.symbol for atom in atoms])
        self.positions = np.array([atom.position for atom in atoms])

    @classmethod
    def from_file(cls, xyz_filename):
        with open(xyz_filename) as f:
            content = f.readlines()
        num_atoms = int(content[0].strip())
        num_lines = len(content)
        atoms = []
        for i in np.arange(num_lines - num_atoms, num_lines):
            line = content[i].split()
            symbol = line[0]
            position = (float(line[1]), float(line[2]), float(line[3]))
            atoms.append(Atom(symbol, position))
        return cls(atoms)

    def copy(self):
        return copy.deepcopy(self)

    def append(self, symbol, position):
        self.num_atoms += 1
        self.atoms = np.append(self.atoms, Atom(symbol, position))
        self.atom_types = set([atom.symbol for atom in self.atoms])
        self.positions = np.array([atom.position for atom in self.atoms])

    def remove(self, inds):
        self.atoms = np.delete(self.atoms, inds, axis=0)
        self.num_atoms = len(self.atoms)
        self.atom_types = set([atom.symbol for atom in self.atoms])
        self.positions = np.array([atom.position for atom in self.atoms])

    def nbrs(self, position):
        kdt = KDTree(self.positions, leaf_size=30, metric='euclidean')
        return kdt.query(position, k=1, return_distance=False).flatten()

def load_xyz(file_name):
    with open(file_name) as f:
        content = f.readlines()
    num_atoms = int(content[0].strip())
    num_lines = len(content)
    atoms = []
    for i in np.arange(num_lines - num_atoms, num_lines):
        line = content[i].split()
        symbol = line[0]
        position = (float(line[1]), float(line[2]), float(line[3]))
        atoms.append(Atom(symbol, position))
    return Atoms(atoms)

def save_xyz(atoms, filename):
    num_atoms = len(atoms)
    with open(filename, 'w') as f:
        f.writelines(str(num_atoms)+'\n\n')
        for atom in atoms:
            symbol = atom.symbol
            x, y, z = atom.position
            line = symbol+' '+str(x)+' '+str(y)+' '+str(z)+'\n'
            f.writelines(line)