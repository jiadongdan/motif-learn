import numpy as np
from sklearn.neighbors import NearestNeighbors
from matplotlib.collections import LineCollection


def generate_lattice(u, v, z=0., n=10, dx=0., dy=0., element='X'):
    uv = np.stack([u, v])
    ijd = np.array([[i, j, np.sqrt(i ** 2 + j ** 2 + i * j)] for i in np.arange(-n, n) for j in np.arange(-n, n)])
    d_selected = np.unique(ijd[:, 2])[0:n]
    ind = np.where(np.in1d(ijd[:, 2], d_selected))[0]
    ij = ijd[ind, 0:2]
    xy = ij.dot(uv)
    xyz = np.array([(element, x + dx, y + dy, z) for (x, y) in xy], dtype=object)
    return xyz


def rot_matrix(angle=30):
    theta = np.radians(angle)
    c, s = np.cos(theta), np.sin(theta)
    R = np.array(((c, -s), (s, c)))
    return R


def generate_mos2(z=0.0, n=20, theta=0, center='Mo'):
    a = 3.12
    c = 12.3
    u = np.array([a, 0])
    v = np.array([a / 2, np.sqrt(3) / 2 * a])
    R = rot_matrix(theta)
    u = R.dot(u)
    v = R.dot(v)

    dx, dy = (u + v) / 3
    if center == 'Mo':
        Mo = generate_lattice(u, v, z=z, n=n, element='Mo')
        S1 = generate_lattice(u, v, z=-a / 2 + z, n=n, dx=dx, dy=dy, element='S')
        S2 = generate_lattice(u, v, z=a / 2 + z, n=n, dx=dx, dy=dy, element='S')
    elif center == 'S':
        S1 = generate_lattice(u, v, z=a / 2 + z, n=n, element='S')
        S2 = generate_lattice(u, v, z=-a / 2 + z, n=n, element='S')
        Mo = generate_lattice(u, v, z=z, n=n, dx=dx, dy=dy, element='Mo')

    mos2 = np.vstack([Mo, S1, S2])
    return mos2


def bilayer_mos2(n, stacking='2H'):
    if stacking == '2H':
        layer1 = generate_mos2(z=6.1 / 2, n=n, center='Mo')
        layer2 = generate_mos2(z=-6.1 / 2, n=n, center='S')
    elif stacking == '2H1':
        layer1 = generate_mos2(z=6.1 / 2, n=n, center='Mo')
        layer2 = generate_mos2(z=-6.1 / 2, n=n, center='S', theta=60)
    elif stacking == 'AA':
        layer1 = generate_mos2(z=6.1 / 2, n=n, center='Mo')
        layer2 = generate_mos2(z=-6.1 / 2, n=n, center='Mo')
    elif stacking == 'AA1':
        layer1 = generate_mos2(z=6.1 / 2, n=n, center='Mo')
        layer2 = generate_mos2(z=-6.1 / 2, n=n, center='Mo', theta=60)
    elif stacking == 'AA2':
        layer1 = generate_mos2(z=6.1 / 2, n=n, center='S')
        layer2 = generate_mos2(z=-6.1 / 2, n=n, center='S', theta=60)

    return [layer1, layer2]


def plot_bonds(ax, layer, direction='xy', color='gray', lw=3.):
    if direction == 'xy':
        ind1, ind2 = 0, 1
    elif direction == 'xz':
        ind1, ind2 = 0, 2
    elif direction == 'yz':
        ind1, ind2 = 1, 2

    e1, e2 = np.unique(layer[:, 0])
    mask = layer[:, 0] == e1
    pts1 = layer[mask, 1:]
    pts2 = layer[~mask, 1:]
    nbrs = NearestNeighbors(n_neighbors=7, algorithm='ball_tree').fit(pts2)
    d, inds = nbrs.radius_neighbors(pts1, 3.12)
    segs = [[pts1[i][[ind1, ind2]], pts2[ind][[ind1, ind2]]] for i, row in enumerate(inds) for ind in row]

    line_segments = LineCollection(segs, zorder=-2, color=color, lw=lw)
    ax.add_collection(line_segments)

def get_size_inches(ax):
    fig_w, fig_h = ax.figure.get_size_inches()
    w, h = ax.get_position().bounds[2:]
    return (fig_w*w, fig_h*h)

def plot_bilayer(ax, model, colors=('C0', 'C1'), s=None, ratio=None, direction='xy'):
    if direction == 'xy':
        ind1, ind2 = 1, 2
    elif direction == 'xz':
        ind1, ind2 = 1, 3
    elif direction == 'yz':
        ind1, ind2 = 2, 3

    if len(colors) == 2:
        colors1 = colors
        colors2 = colors
    elif len(colors) == 4:
        colors1 = colors[0:2]
        colors2 = colors[2:4]

    if s is None:
        w, h = get_size_inches(ax)
        d = np.minimum(w, h)
        s = 10 * np.sqrt(d / 3.6)
    if ratio is None:
        ratio = 0.65
    ss = [s ** 2, (s * ratio) ** 2]

    layer1, layer2 = model
    elements = np.unique(layer1[:, 0])
    for i, e in enumerate(elements):
        mask = layer1[:, 0] == e
        ax.scatter(layer1[mask, ind1], layer1[mask, ind2], s=ss[i], label=e, c=colors1[i], zorder=-1)

    elements = np.unique(layer2[:, 0])
    for i, e in enumerate(elements):
        mask = layer2[:, 0] == e
        ax.scatter(layer2[mask, ind1], layer2[mask, ind2], s=ss[i], label=e, c=colors2[i], zorder=0)

    plot_bonds(ax, layer1, color='darkgray', direction=direction, lw=s * 0.3)
    plot_bonds(ax, layer2, color='darkgray', direction=direction, lw=s * 0.3)

    ax.axis('equal')








