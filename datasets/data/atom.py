import numpy as np
class Atom():
    def __init__(self, symbol, x, y, z):
        self.symbol = symbol
        self.x = x
        self.y = y
        self.z = z
        self.position = np.array((x, y, z))

    def set_x(self, x):
        self.x = x
        self.position[0] = x

    def set_y(self, y):
        self.y = y
        self.position[1] = y

    def set_z(self, z):
        self.z = z
        self.position[2] = z

    def set_position(self, position):
        self.x, self.y, self.z = position
        self.position = np.array(position)

    def set_element(self, symbol):
        self.symbol = symbol

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
    return np.array(atoms, dtype=object)

class Atoms():
    def __init__(self, file_name, atoms=None):
        if atoms is None:
            self.data = read_xyz(file_name)
        else:
            self.data = atoms
        self.xyz = self.data[:, 1:]
        self.elements = self.data[:, 0]


