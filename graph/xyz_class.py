import numpy as np


def read_xyz(filename):
    pass


def extract_elements(data):
    pass

def get_num_of_atoms(data):
    pass



class XYZ:

    def __init__(self, filename):

        self.data = read_xyz(filename)

        self.num_of_atoms = get_num_of_atoms(self.data)
        self.elements = extract_elements(self.data)


    @classmethod
    def from_dict(cls, dictionary):
        pass




