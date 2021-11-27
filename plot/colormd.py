import numpy as np
from matplotlib.colors import to_hex


def color2md(c):
    if c=='':
        # 72 is the length of color block
        block = ' '*72
    else:
        hexcolor = to_hex(c)
        md_image = r'![{}](https://via.placeholder.com/15/{}/000000?text=+)'.format(hexcolor, hexcolor[1:])
        md_color = r'`{}`'.format(hexcolor)
        block = md_image +  md_color
    return block

class mdcolortable:

    def __init__(self, colordict):
        self.ncols = len(colordict.keys())
        self.nrows = np.max([len(e) for e in colordict.values()])

        colors = []
        for e in list(colordict.keys()):
            if len(colordict[e]) < self.nrows:
                c = [to_hex(i) for i in colordict[e]] + [''] * (self.nrows - len(colordict[e]))
            else:
                c = [to_hex(i) for i in colordict[e]]
            colors.append(c)
        self.colors = np.array(colors).T

        line = '|' + '|'.join(['-' * 72 for e in list(colordict.keys())]) + '|\n'

        self.content = '|' + '|'.join([e + ' ' * (72 - len(e)) for e in list(colordict.keys())]) + '|\n' + line

        for row in self.colors:
            ss = '|' + '|'.join([color2md(c) for c in row]) + '|\n'
            self.content += ss