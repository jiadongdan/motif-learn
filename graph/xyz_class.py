import numpy as np


def read_xyz(file_name):
    with open(file_name) as f:
        content = f.readlines()
        num_atoms = int(content[0].strip())
        num_lines = len(content)
        atoms = []
        for i in np.arange(num_lines - num_atoms, num_lines):
            line = content[i].split()
            symbol = line[0]
            x, y, z = (float(line[1]), float(line[2]), float(line[3]))
            atoms.append([symbol, x, y, z])
    atoms = np.array(atoms, dtype=object)

    elements = np.unique(atoms[:, 0])
    data = {e: atoms[:, 1:][atoms[:, 0] == e] for e in elements}
    return data


def extract_elements(data):
    return [e for e in data]

def get_num_of_atoms(data):
    return sum([data[e].shape[0] for e in data])


class XYZ:

    def __init__(self, data):

        self.data = data
        self.num_of_atoms = get_num_of_atoms(self.data)
        self.elements = extract_elements(self.data)

    @classmethod
    def from_file(cls, filename):
        data = read_xyz(filename)
        return cls(data)





