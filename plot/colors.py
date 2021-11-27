import numpy as np
import matplotlib.colors as mc
import matplotlib.pyplot as plt
from  colorsys import rgb_to_hls, hls_to_rgb
from markdown import markdown

from ._color_data import mpl_20


def color2md(c):
    if c=='':
        # 72 is the length of color block
        block = ' '*72
    else:
        hexcolor = mc.to_hex(c)
        md_image = r'![{}](https://via.placeholder.com/15/{}/000000?text=+)'.format(hexcolor, hexcolor[1:])
        md_color = r'`{}`'.format(hexcolor)
        block = md_image +  md_color
    return block

def md2html(mdcolor):
    return markdown(mdcolor)

#=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
# Color, Colors and ColorTable class, markdown --> html
#=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=

class Color:

    def __init__(self, c):
        self.hexcolor = to_hex(c)
        self.mdcolor = color2md(c)

    def _repr_html_(self):
        return md2html(self.mdcolor)

class Colors:

    def __init__(self, colors, name=None):
        if name is None:
            self.name = 'colors'
        else:
            self.name = name

        hexcolors = [mc.to_hex(c) for c in colors]
        if len(hexcolors) < 10:
            self.hexcolors = hexcolors
            self.mdcolors = [color2md(c) for c in self.hexcolors]
        else:
            n = int(np.ceil(len(hexcolors)/10)*10)
            self.hexcolors = hexcolors + ['']*(n-len(hexcolors))
            self.mdcolors = [color2md(c) for c in self.hexcolors]
            self.mdcolors = np.array(self.mdcolors).reshape(-1, 10).T.ravel()
        # insert \n
        ncols = int(np.ceil(len(hexcolors)/10))
        cs = []
        for i, c in enumerate(self.mdcolors):
            cs.append(c)
            if (i+1)%ncols == 0:
                cs.append('\n')
        # join by "|"
        content = '|'+'|'.join(cs)

        line = '|'+'|'.join(['-'*72 for i in range(ncols)])+ '|\n'
        head = '|'+'|'.join([self.name+' '*(72-len(self.name))] + [' '*72 for i in range(ncols-1)])+ '|\n'
        self.md = head + line + content

    def _repr_html_(self):
        return markdown(self.md, extensions=['markdown.extensions.tables'])


class ColorTable:

    def __init__(self, colordict):
        self.ncols = len(colordict.keys())
        self.nrows = np.max([len(e) for e in colordict.values()])

        colors = []
        for e in list(colordict.keys()):
            if len(colordict[e]) < self.nrows:
                c = [mc.to_hex(i) for i in colordict[e]] + [''] * (self.nrows - len(colordict[e]))
            else:
                c = [mc.to_hex(i) for i in colordict[e]]
            colors.append(c)
        self.colors = np.array(colors).T

        line = '|' + '|'.join(['-' * 72 for e in list(colordict.keys())]) + '|\n'

        self.content = '|' + '|'.join([e + ' ' * (72 - len(e)) for e in list(colordict.keys())]) + '|\n' + line

        for row in self.colors:
            ss = '|' + '|'.join([color2md(c) for c in row]) + '|\n'
            self.content += ss

    def _repr_html_(self):
        return markdown(self.content, extensions=['markdown.extensions.tables'])




# this function is frequnetly used !!!
def generate_colors_from_lbs(lbs, colors=None):
    if colors is None:
        colors = np.array(mpl_20)
    else:
        colors = np.array(colors)
    lbs = np.array(lbs) % len(colors)
    return colors[lbs]


# this function is frequnetly used !!!
def colors_from_lbs(lbs, colors=None, xy=None, alpha_min=0.5):
    if colors is None:
        colors = np.array(mpl_20)
    else:
        colors = np.array(colors)
    lbs = np.array(lbs) % len(colors)

    c = to_rgba(colors[lbs])

    if xy is not None:
        xy_ = np.vstack([xy[lbs == e] - xy[lbs == e].mean(axis=0) for e in np.unique(lbs)])
        r = np.hypot(xy_[:, 0], xy_[:, 1])
        r = r / r.max()
        r[r < alpha_min] = alpha_min
        c[:, 3] = r
    return c


def get_pca_colors(xy, cmap=None, lmin=0.2, lmax=0.8):
    if cmap is None:
        cmap = plt.cm.coolwarm
    norm1 = plt.Normalize(xy[:, 0].min(), xy[:, 0].max())
    rgb = cmap(norm1(xy[:, 0]))[:, 0:3]
    norm2 = plt.Normalize(xy[:, 1].min(), xy[:, 1].max())
    ls= (norm2(xy[:, 1]))*(lmax-lmin)+lmin
    hls = np.array([(rgb_to_hls(*e)[0], l, rgb_to_hls(*e)[2])  for e, l in zip(rgb, ls)])
    rgb = np.array([hls_to_rgb(*e) for e in hls])
    return rgb


#=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
# color conversion
#=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=

def to_rgba(colors, alpha=None):
    rgba = np.array([mc.to_rgba(c, alpha) for c in colors if mc.is_color_like(c)])
    return rgba

def to_rgb(colors):
    rgba = np.array([mc.to_rgba(c) for c in colors if mc.is_color_like(c)])
    return rgba[:, 0:3]

def to_hex(colors, keep_alpha=False):
    hex = np.array([mc.to_hex(c, keep_alpha=keep_alpha) for c in colors if mc.is_color_like(c)])
    return hex
