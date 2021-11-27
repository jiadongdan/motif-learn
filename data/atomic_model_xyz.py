import re
import os
import numpy as np

from ..io import load_pickle

__all__ = ['atoms_mos2',
           'coeffs_Lobato']


def read_atoms(file_name):
    elements = np.array(['H', 'He', 'Li', 'Be', 'B', 'C', 'N', 'O', 'F', 'Ne', 'Na', 'Mg', 'Al', 'Si', 'P', 'S', 'Cl',
                         'Ar', 'K', 'Ca', 'Sc', 'Ti', 'V', 'Cr', 'Mn', 'Fe', 'Co', 'Ni', 'Cu', 'Zn', 'Ga', 'Ge', 'As',
                         'Se', 'Br', 'Kr', 'Rb', 'Sr', 'Y', 'Zr', 'Nb', 'Mo', 'Tc', 'Ru', 'Rh', 'Pd', 'Ag', 'Cd', 'In',
                         'Sn', 'Sb', 'Te', 'I', 'Xe', 'Cs', 'Ba', 'La', 'Ce', 'Pr', 'Nd', 'Pm', 'Sm', 'Eu', 'Gd', 'Tb',
                         'Dy', 'Ho', 'Er', 'Tm', 'Yb', 'Lu', 'Hf', 'Ta', 'W', 'Re', 'Os', 'Ir', 'Pt', 'Au', 'Hg', 'Tl',
                         'Pb', 'Bi', 'Po', 'At', 'Rn', 'Fr', 'Ra', 'Ac', 'Th', 'Pa', 'U', 'Np', 'Pu', 'Am', 'Cm', 'Bk',
                         'Cf', 'Es', 'Fm', 'Md', 'No', 'Lr', 'Rf', 'Db', 'Sg', 'Bh', 'Hs', 'Mt', 'Ds', 'Rg', 'Cn', 'Nh',
                         'Fl', 'Mc', 'Lv', 'Ts', 'Og'])

    numeric_const_pattern = '[-+]? (?: (?: \d* \. \d+ ) | (?: \d+ \.? ) )(?: [Ee] [+-]? \d+ ) ?'
    rx = re.compile(numeric_const_pattern, re.VERBOSE)
    l = []
    with open(file_name, 'r') as f:
        lines = f.readlines()
        for line in lines[2:]:
            symbol = line[0:2].strip()
            atomic_numbers = np.where(elements == symbol)[0][0] + 1
            x, y, z = rx.findall(line)
            l.append([atomic_numbers, x, y, z])
    return np.array(l).astype(np.float)

data_dir = os.path.dirname(__file__) + '/atomic_models/'


atoms_mos2 = read_atoms(data_dir+'monolayer_MoS2_20.xyz')


coeffs_Lobato = load_pickle(data_dir+'coeffs_Lobato.pkl')

