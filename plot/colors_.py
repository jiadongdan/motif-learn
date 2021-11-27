import numpy as np
import matplotlib.pyplot as plt
from collections import OrderedDict

from matplotlib.colors import LinearSegmentedColormap
from matplotlib.cm import register_cmap

import matplotlib.colors as mc
import colorsys


class AttrODict(OrderedDict):
    """Ordered dictionary with attribute access (e.g. for tab completion)"""
    def __dir__(self): return self.keys()
    def __delattr__(self, name): del self[name]
    def __getattr__(self, name):
        return self[name] if not name.startswith('_') else super(AttrODict, self).__getattr__(name)
    def __setattr__(self, name, value):
        if (name.startswith('_')): return super(AttrODict, self).__setattr__(name, value)
        self[name] = value


def mpl_cm(name,colorlist):
    """
    Register colormap
    :param name: str
        colormap name
    :param colorlist: array_like
        colormap data
    :return: matplotlib.colors.LinearSegmentedColormap
        matplotlib colormap
    """
    cm[name] = LinearSegmentedColormap.from_list(name, colorlist, N=len(colorlist))
    register_cmap("cet_"+name, cmap=cm[name])
    return cm[name]

def color_palette_(color):
    try:
        hex_color = mc.cnames[color]
    except:
        hex_color = color
    rgb = mc.to_rgb(hex_color)
    hls = colorsys.rgb_to_hls(*rgb)
    palette = np.array([colorsys.hls_to_rgb(*(hls[0], i, hls[2])) for i in np.linspace(0, 1, 256)])
    cm = mc.ListedColormap(palette)
    return cm

def color_palette(name, low=0, high=1):
    if mc.is_color_like(name):
        rgb = mc.to_rgb(name)
        hls = colorsys.rgb_to_hls(*rgb)
        palette = np.array([colorsys.hls_to_rgb(*(hls[0], i, hls[2])) for i in np.linspace(low, high, 256)])
        cmap = mc.ListedColormap(palette)
    else:
        cmap = cm.get_cmap(name)
    return cmap

#=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
# color cycles
#=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=

matplotlib_cc = plt.rcParams['axes.prop_cycle'].by_key()['color']

mathematica97_cc = ['#5e81b5', '#e19c24', '#8fb032', '#eb6235', '#8778b3',
                    '#c56e1a', '#5d9ec7', '#ffbf00', '#a5609d', '#929600',
                    '#e95536', '#6685d9', '#f89f13', '#bc5b80', '#47b66d']

mathematica98_cc = ['#4a969c', '#e28617', '#9d6095', '#85a818', '#d15739',
                    '#6f7bb8', '#e9ac03', '#af5b71', '#38a77e', '#dd6f22',
                    '#8468b8', '#c2aa00', '#b8575c', '#48909f', '#dd8516']

cc = AttrODict()
cc['matplotlib'] = matplotlib_cc
cc['mathematica97'] = mathematica97_cc
cc['mathematica98'] = mathematica98_cc

#=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
# colormaps
#=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
cm = AttrODict()

# colormap-blue
g = np.arange(0, 256)
b = np.arange(0, 256*2,2)
b[128:] = 255
r = np.arange(-256, 256, 2)
r[0:128] = 0
blue = (np.vstack([r, g, b]).T)/255

m_blue = mpl_cm('blue', blue)
m_blue_r = mpl_cm('blue_r',list(reversed(blue)))

# colormap-purple
purple = np.array([[0.        , 0.        , 0.        ],
       [0.00393701, 0.00393701, 0.00787402],
       [0.00787402, 0.00787402, 0.01181102],
       [0.00787402, 0.00787402, 0.01968504],
       [0.01181102, 0.01181102, 0.02362205],
       [0.01574803, 0.01574803, 0.03149606],
       [0.01968504, 0.01968504, 0.03937008],
       [0.02362205, 0.02362205, 0.04330709],
       [0.02362205, 0.02362205, 0.0511811 ],
       [0.02755906, 0.02755906, 0.05511811],
       [0.03149606, 0.03149606, 0.06299213],
       [0.03543307, 0.03543307, 0.07086614],
       [0.03937008, 0.03937008, 0.07480315],
       [0.03937008, 0.03937008, 0.08267717],
       [0.04330709, 0.04330709, 0.08661417],
       [0.04724409, 0.04724409, 0.09448819],
       [0.0511811 , 0.0511811 , 0.1023622 ],
       [0.05511811, 0.05511811, 0.10629921],
       [0.05511811, 0.05511811, 0.11417323],
       [0.05905512, 0.05905512, 0.11811024],
       [0.06299213, 0.06299213, 0.12598425],
       [0.06692913, 0.06692913, 0.13385827],
       [0.07086614, 0.07086614, 0.13779528],
       [0.07480315, 0.07480315, 0.14566929],
       [0.07480315, 0.07480315, 0.1496063 ],
       [0.07874016, 0.07874016, 0.15748031],
       [0.08267717, 0.08267717, 0.16535433],
       [0.08661417, 0.08661417, 0.16929134],
       [0.09055118, 0.09055118, 0.17716535],
       [0.09055118, 0.09055118, 0.18110236],
       [0.09448819, 0.09448819, 0.18897638],
       [0.0984252 , 0.0984252 , 0.19685039],
       [0.1023622 , 0.1023622 , 0.2007874 ],
       [0.10629921, 0.10629921, 0.20866142],
       [0.10629921, 0.10629921, 0.21259843],
       [0.11023622, 0.11023622, 0.22047244],
       [0.11417323, 0.11417323, 0.22834646],
       [0.11811024, 0.11811024, 0.23228346],
       [0.12204724, 0.12204724, 0.24015748],
       [0.12204724, 0.12204724, 0.24409449],
       [0.12598425, 0.12598425, 0.2519685 ],
       [0.12992126, 0.12992126, 0.25590551],
       [0.13385827, 0.13385827, 0.26377953],
       [0.13779528, 0.13779528, 0.27165354],
       [0.13779528, 0.13779528, 0.27559055],
       [0.14173228, 0.14173228, 0.28346457],
       [0.14566929, 0.14566929, 0.28740157],
       [0.1496063 , 0.1496063 , 0.29527559],
       [0.15354331, 0.15354331, 0.30314961],
       [0.15354331, 0.15354331, 0.30708661],
       [0.15748031, 0.15748031, 0.31496063],
       [0.16141732, 0.16141732, 0.31889764],
       [0.16535433, 0.16535433, 0.32677165],
       [0.16929134, 0.16929134, 0.33464567],
       [0.16929134, 0.16929134, 0.33858268],
       [0.17322835, 0.17322835, 0.34645669],
       [0.17716535, 0.17716535, 0.3503937 ],
       [0.18110236, 0.18110236, 0.35826772],
       [0.18503937, 0.18503937, 0.36614173],
       [0.18503937, 0.18503937, 0.37007874],
       [0.18897638, 0.18897638, 0.37795276],
       [0.19291339, 0.19291339, 0.38188976],
       [0.19685039, 0.19685039, 0.38976378],
       [0.2007874 , 0.2007874 , 0.3976378 ],
       [0.20472441, 0.20472441, 0.4015748 ],
       [0.20472441, 0.20472441, 0.40944882],
       [0.20866142, 0.20866142, 0.41338583],
       [0.21259843, 0.21259843, 0.42125984],
       [0.21653543, 0.21653543, 0.42913386],
       [0.22047244, 0.22047244, 0.43307087],
       [0.22047244, 0.22047244, 0.44094488],
       [0.22440945, 0.22440945, 0.44488189],
       [0.22834646, 0.22834646, 0.45275591],
       [0.23228346, 0.23228346, 0.46062992],
       [0.23622047, 0.23622047, 0.46456693],
       [0.23622047, 0.23622047, 0.47244094],
       [0.24015748, 0.24015748, 0.47637795],
       [0.24409449, 0.24409449, 0.48425197],
       [0.2480315 , 0.2480315 , 0.49212598],
       [0.2519685 , 0.2519685 , 0.49606299],
       [0.2519685 , 0.2519685 , 0.50393701],
       [0.25590551, 0.25590551, 0.50787402],
       [0.25984252, 0.25984252, 0.51574803],
       [0.26377953, 0.26377953, 0.52362205],
       [0.26771654, 0.26771654, 0.52755906],
       [0.26771654, 0.26771654, 0.53543307],
       [0.27165354, 0.27165354, 0.53937008],
       [0.27559055, 0.27559055, 0.54724409],
       [0.27952756, 0.27952756, 0.55511811],
       [0.28346457, 0.28346457, 0.55905512],
       [0.28346457, 0.28346457, 0.56692913],
       [0.28740157, 0.28740157, 0.57086614],
       [0.29133858, 0.29133858, 0.57874016],
       [0.29527559, 0.29527559, 0.58661417],
       [0.2992126 , 0.2992126 , 0.59055118],
       [0.2992126 , 0.2992126 , 0.5984252 ],
       [0.30314961, 0.30314961, 0.6023622 ],
       [0.30708661, 0.30708661, 0.61023622],
       [0.31102362, 0.31102362, 0.61811024],
       [0.31496063, 0.31496063, 0.62204724],
       [0.31889764, 0.31889764, 0.62992126],
       [0.31889764, 0.31889764, 0.63385827],
       [0.32283465, 0.32283465, 0.64173228],
       [0.32677165, 0.32677165, 0.6496063 ],
       [0.33070866, 0.33070866, 0.65354331],
       [0.33464567, 0.33464567, 0.66141732],
       [0.33464567, 0.33464567, 0.66535433],
       [0.33858268, 0.33858268, 0.67322835],
       [0.34251969, 0.34251969, 0.68110236],
       [0.34645669, 0.34645669, 0.68503937],
       [0.3503937 , 0.3503937 , 0.69291339],
       [0.3503937 , 0.3503937 , 0.69685039],
       [0.35433071, 0.35433071, 0.70472441],
       [0.35826772, 0.35826772, 0.71259843],
       [0.36220472, 0.36220472, 0.71653543],
       [0.36614173, 0.36614173, 0.72440945],
       [0.36614173, 0.36614173, 0.72834646],
       [0.37007874, 0.37007874, 0.73622047],
       [0.37401575, 0.37401575, 0.74409449],
       [0.37795276, 0.37795276, 0.7480315 ],
       [0.38188976, 0.38188976, 0.75590551],
       [0.38188976, 0.38188976, 0.75984252],
       [0.38582677, 0.38582677, 0.76771654],
       [0.38976378, 0.38976378, 0.77165354],
       [0.39370079, 0.39370079, 0.77952756],
       [0.3976378 , 0.3976378 , 0.78740157],
       [0.3976378 , 0.3976378 , 0.79133858],
       [0.4015748 , 0.4015748 , 0.7992126 ],
       [0.40551181, 0.40551181, 0.80314961],
       [0.40944882, 0.40944882, 0.81102362],
       [0.41338583, 0.41338583, 0.81889764],
       [0.41338583, 0.41338583, 0.82283465],
       [0.41732283, 0.41732283, 0.83070866],
       [0.42125984, 0.42125984, 0.83464567],
       [0.42519685, 0.42519685, 0.84251969],
       [0.42913386, 0.42913386, 0.8503937 ],
       [0.42913386, 0.42913386, 0.85433071],
       [0.43307087, 0.43307087, 0.86220472],
       [0.43700787, 0.43700787, 0.86614173],
       [0.44094488, 0.44094488, 0.87401575],
       [0.44488189, 0.44488189, 0.88188976],
       [0.4488189 , 0.4488189 , 0.88582677],
       [0.4488189 , 0.4488189 , 0.89370079],
       [0.45275591, 0.45275591, 0.8976378 ],
       [0.45669291, 0.45669291, 0.90551181],
       [0.46062992, 0.46062992, 0.91338583],
       [0.46456693, 0.46456693, 0.91732283],
       [0.46456693, 0.46456693, 0.92519685],
       [0.46850394, 0.46850394, 0.92913386],
       [0.47244094, 0.47244094, 0.93700787],
       [0.47637795, 0.47637795, 0.94488189],
       [0.48031496, 0.48031496, 0.9488189 ],
       [0.48031496, 0.48031496, 0.95669291],
       [0.48425197, 0.48425197, 0.96062992],
       [0.48818898, 0.48818898, 0.96850394],
       [0.49212598, 0.49212598, 0.97637795],
       [0.49606299, 0.49606299, 0.98031496],
       [0.49606299, 0.49606299, 0.98818898],
       [0.5       , 0.5       , 0.99212598],
       [0.50393701, 0.50393701, 0.99606299],
       [0.50787402, 0.50787402, 0.99606299],
       [0.51574803, 0.51574803, 0.99606299],
       [0.51968504, 0.51968504, 0.99606299],
       [0.52362205, 0.52362205, 0.99606299],
       [0.53149606, 0.53149606, 0.99606299],
       [0.53543307, 0.53543307, 0.99606299],
       [0.53937008, 0.53937008, 0.99606299],
       [0.54724409, 0.54724409, 0.99606299],
       [0.5511811 , 0.5511811 , 0.99606299],
       [0.55511811, 0.55511811, 0.99606299],
       [0.55905512, 0.55905512, 0.99606299],
       [0.56692913, 0.56692913, 0.99606299],
       [0.57086614, 0.57086614, 0.99606299],
       [0.57480315, 0.57480315, 0.99606299],
       [0.58267717, 0.58267717, 0.99606299],
       [0.58661417, 0.58661417, 0.99606299],
       [0.59055118, 0.59055118, 0.99606299],
       [0.5984252 , 0.5984252 , 0.99606299],
       [0.6023622 , 0.6023622 , 0.99606299],
       [0.60629921, 0.60629921, 0.99606299],
       [0.61417323, 0.61417323, 0.99606299],
       [0.61811024, 0.61811024, 0.99606299],
       [0.62204724, 0.62204724, 0.99606299],
       [0.62992126, 0.62992126, 0.99606299],
       [0.63385827, 0.63385827, 0.99606299],
       [0.63779528, 0.63779528, 0.99606299],
       [0.64173228, 0.64173228, 0.99606299],
       [0.6496063 , 0.6496063 , 0.99606299],
       [0.65354331, 0.65354331, 0.99606299],
       [0.65748031, 0.65748031, 0.99606299],
       [0.66535433, 0.66535433, 0.99606299],
       [0.66929134, 0.66929134, 0.99606299],
       [0.67322835, 0.67322835, 0.99606299],
       [0.68110236, 0.68110236, 0.99606299],
       [0.68503937, 0.68503937, 0.99606299],
       [0.68897638, 0.68897638, 0.99606299],
       [0.69685039, 0.69685039, 0.99606299],
       [0.7007874 , 0.7007874 , 0.99606299],
       [0.70472441, 0.70472441, 0.99606299],
       [0.71259843, 0.71259843, 0.99606299],
       [0.71653543, 0.71653543, 0.99606299],
       [0.72047244, 0.72047244, 0.99606299],
       [0.72440945, 0.72440945, 0.99606299],
       [0.73228346, 0.73228346, 0.99606299],
       [0.73622047, 0.73622047, 0.99606299],
       [0.74015748, 0.74015748, 0.99606299],
       [0.7480315 , 0.7480315 , 0.99606299],
       [0.7519685 , 0.7519685 , 1.        ],
       [0.75590551, 0.75590551, 1.        ],
       [0.76377953, 0.76377953, 1.        ],
       [0.76771654, 0.76771654, 1.        ],
       [0.77165354, 0.77165354, 1.        ],
       [0.77952756, 0.77952756, 1.        ],
       [0.78346457, 0.78346457, 1.        ],
       [0.78740157, 0.78740157, 1.        ],
       [0.79527559, 0.79527559, 1.        ],
       [0.7992126 , 0.7992126 , 1.        ],
       [0.80314961, 0.80314961, 1.        ],
       [0.80708661, 0.80708661, 1.        ],
       [0.81496063, 0.81496063, 1.        ],
       [0.81889764, 0.81889764, 1.        ],
       [0.82283465, 0.82283465, 1.        ],
       [0.83070866, 0.83070866, 1.        ],
       [0.83464567, 0.83464567, 1.        ],
       [0.83858268, 0.83858268, 1.        ],
       [0.84645669, 0.84645669, 1.        ],
       [0.8503937 , 0.8503937 , 1.        ],
       [0.85433071, 0.85433071, 1.        ],
       [0.86220472, 0.86220472, 1.        ],
       [0.86614173, 0.86614173, 1.        ],
       [0.87007874, 0.87007874, 1.        ],
       [0.87795276, 0.87795276, 1.        ],
       [0.88188976, 0.88188976, 1.        ],
       [0.88582677, 0.88582677, 1.        ],
       [0.88976378, 0.88976378, 1.        ],
       [0.8976378 , 0.8976378 , 1.        ],
       [0.9015748 , 0.9015748 , 1.        ],
       [0.90551181, 0.90551181, 1.        ],
       [0.91338583, 0.91338583, 1.        ],
       [0.91732283, 0.91732283, 1.        ],
       [0.92125984, 0.92125984, 1.        ],
       [0.92913386, 0.92913386, 1.        ],
       [0.93307087, 0.93307087, 1.        ],
       [0.93700787, 0.93700787, 1.        ],
       [0.94488189, 0.94488189, 1.        ],
       [0.9488189 , 0.9488189 , 1.        ],
       [0.95275591, 0.95275591, 1.        ],
       [0.96062992, 0.96062992, 1.        ],
       [0.96456693, 0.96456693, 1.        ],
       [0.96850394, 0.96850394, 1.        ],
       [0.97244094, 0.97244094, 1.        ],
       [0.98031496, 0.98031496, 1.        ],
       [0.98425197, 0.98425197, 1.        ],
       [0.98818898, 0.98818898, 1.        ],
       [0.99606299, 0.99606299, 1.        ],
       [1.        , 1.        , 1.        ]])

m_purple = mpl_cm('purple', purple)
m_purple_r = mpl_cm('purple_r',list(reversed(purple)))

#=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
# color conversion
#=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
def rgb2hex():
    pass

def hex2rgb(color):
    try:
        hex_color = mc.cnames[color]
    except:
        hex_color = color
    rgb = mc.to_rgb(hex_color)
    return rgb

def hex2rgba(color):
    try:
        hex_color = mc.cnames[color]
    except:
        hex_color = color
    rgb = list(mc.to_rgb(hex_color))
    rgba = rgb+[1]
    return rgba

def lighter(color, f=0.5):
    try:
        hex_color = mc.cnames[color]
    except:
        hex_color = color
    rgb = mc.to_rgb(hex_color)
    h, l, s = colorsys.rgb_to_hls(*rgb)
    l = (1 - l)*f+l
    hls = (h, l, s)
    rgb = colorsys.hls_to_rgb(*hls)
    return rgb

def darker(color, f=0.5):
    try:
        hex_color = mc.cnames[color]
    except:
        hex_color = color
    rgb = mc.to_rgb(hex_color)
    h, l, s = colorsys.rgb_to_hls(*rgb)
    l = l*(1-f)
    hls = (h, l, s)
    rgb = colorsys.hls_to_rgb(*hls)
    return rgb


default_colors = plt.rcParams['axes.prop_cycle'].by_key()['color']

def generate_colors_from_lbs(lbs, colors=None):
    if colors is None:
        colors = np.array(default_colors)
    else:
        colors = np.array(colors)
    lbs = np.array(lbs) % len(colors)
    return colors[lbs]