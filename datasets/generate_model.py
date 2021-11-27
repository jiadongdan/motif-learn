import numpy as np

def string2symbols(s):
    """Convert string to list of chemical symbols."""
    n = len(s)

    if n == 0:
        return []

    c = s[0]

    if c.isdigit():
        i = 1
        while i < n and s[i].isdigit():
            i += 1
        return int(s[:i]) * string2symbols(s[i:])

    if c == '(':
        p = 0
        for i, c in enumerate(s):
            if c == '(':
                p += 1
            elif c == ')':
                p -= 1
                if p == 0:
                    break
        j = i + 1
        while j < n and s[j].isdigit():
            j += 1
        if j > i + 1:
            m = int(s[i + 1:j])
        else:
            m = 1
        return m * string2symbols(s[1:i]) + string2symbols(s[j:])

    if c.isupper():
        i = 1
        if 1 < n and s[1].islower():
            i += 1
        j = i
        while j < n and s[j].isdigit():
            j += 1
        if j > i:
            m = int(s[i:j])
        else:
            m = 1
        symbol = s[:i]
        return m * [symbol] + string2symbols(s[j:])
    else:
        raise ValueError


def vector_decompose(v, a, b):
    x1, y1 = np.array(a)
    x2, y2 = np.array(b)
    x, y = np.array(v)
    n = (x * y1 - y * x1) / (x2 * y1 - x1 * y2)
    m = (x * y2 - y * x2) / (x1 * y2 - x2 * y1)
    return m, n


def box_contain(p, box):
    x, y = p
    x0, y0, w, h = box
    if x0 <= x <= x0 + w and y0 <= y <= y0 + h:
        return True
    else:
        return False

def fill_box(p, a, b, box, z=0, element='X'):
    # Convert to array
    p = np.array(p)
    a = np.array(a)
    b = np.array(b)
    # Box params
    x0, y0, w, h = box
    pts = np.array([(x0, y0), (x0 + w, y0), (x0 + w, y0 + h), (x0, y0 + h)])
    box_center = np.array((x0 + w / 2, y0 + h / 2))
    dx, dy = box_center - p
    m, n = vector_decompose((dx, dy), a, b)
    m, n = int(m), int(n)
    # move p to somewhere near box center
    p = p + m * a + n * b

    d2 = [(e[0] - p[0]) ** 2 + (e[1] - p[1]) ** 2 for e in pts]
    # get vertice which is furthest from p
    ind = np.argmax(d2)
    dx, dy = pts[ind] - p
    m, n = vector_decompose((dx, dy), a, b)
    N = max(int(abs(m)), int(abs(n))) + 3
    # fill the box
    pp = np.array([p + i * a + j * b for i in range(-N, N + 1) for j in range(-N, N + 1)])

    err = 1e-4
    mask = (np.abs(pp[:, 0] - x0) < err)
    pp[mask, 0] = x0
    mask = (np.abs(pp[:, 0] - x0 - w) < err)
    pp[mask, 0] = x0 + w

    mask = (np.abs(pp[:, 1] - y0) < err)
    pp[mask, 1] = y0
    mask = (np.abs(pp[:, 1] - y0 - h) < err)
    pp[mask, 1] = y0 + h

    return np.array([(element, e[0], e[1], z) for e in pp if box_contain(e, box)], dtype=object)

def rot_matrix(angle=30):
    theta = np.radians(angle)
    c, s = np.cos(theta), np.sin(theta)
    R = np.array(((c, -s), (s, c)))
    return R

def get_a_and_c(symbol):
    l = np.array(['MoS2', 'MoSe2', 'WS2', 'WSe2'])
    a_list = [3.16, 3.288, 3.16, 3.297]
    c_list = [12.32, 12.931, 12.358, 12.2982]
    ind = np.where(l == symbol)[0][0]
    a = a_list[ind]
    c = c_list[ind]
    return a, c

def get_elements(symbol):
    e1, e2 = string2symbols(symbol)[0:2]
    return e1, e2

def mx2(symbol='MoS2', w=30, ref='m', angle=0, t=None):
    a, c = get_a_and_c(symbol)
    e1, e2 = get_elements(symbol)

    d = a / 2
    tt = np.sqrt(3) / 2
    # rotation matrix
    R = rot_matrix(angle)
    # unit cell vectors
    u = np.array([tt * a, -0.5 * a])
    v = np.array([0, a])
    # shift vector
    shift = np.array([a / np.sqrt(3), 0])
    # rotated version of u, v, and shift
    u = R.dot(u)
    v = R.dot(v)
    shift = R.dot(shift)
    if t is None:
        t = np.array([0, 0])

    box = [-w / 2, -w / 2, w, w]
    if ref == 'm':
        m_atoms = fill_box(t, u, v, box, 0.0, e1)
        x1_atoms = fill_box(t + shift, u, v, box, d, e2)
        x2_atoms = fill_box(t + shift, u, v, box, -d, e2)
    elif ref == 'x':
        x1_atoms = fill_box(t, u, v, box, d, e2)
        x2_atoms = fill_box(t, u, v, box, -d, e2)
        m_atoms = fill_box(t + shift, u, v, box, 0.0, e1)
    MX2 = np.vstack([m_atoms, x1_atoms, x2_atoms])

    return MX2


def bilayer_mx2(symbol='MoS2', w=30, stack='2H'):
    a, c = get_a_and_c(symbol)

    if stack == '2H':
        stack = 1
    elif stack == '3R':
        stack = 4

    tt = a / np.sqrt(3)
    stacking_params = np.array([(0, 0, 60), (tt, 0, 60), (2 * tt, 0, 60), (0, 0, 0), (tt, 0, 0)])[stack]
    t, theta = stacking_params[0:2], stacking_params[2]
    layer1 = mx2(symbol, w=w, angle=0, t=[0, 0])
    layer2 = mx2(symbol, w=w, angle=theta, t=t)
    # z direction information
    layer1[:, 3] = layer1[:, 3] + c / 4
    layer2[:, 3] = layer2[:, 3] - c / 4
    bilayer = np.vstack([layer1, layer2])
    return bilayer


def mtb_mx2(symbol='MoS2', w=30):
    a, c = get_a_and_c(symbol)
    e1, e2 = get_elements(symbol)
    h = w
    box1 = [0, -h / 2, w / 2, h]

    d = a / 2
    tt = np.sqrt(3) / 2
    u1 = np.array([tt * a, -0.5 * a])
    v1 = np.array([tt * a, 0.5 * a])
    # v1 = np.array([0, a])
    shift1 = np.array([a / np.sqrt(3) / 2, 0.5 * a])

    p = np.array([0, 0])
    x1 = fill_box(p, u1, v1, box1, d, e2)
    x2 = fill_box(p, u1, v1, box1, -d, e2)
    m1 = fill_box(p + shift1, u1, v1, box1, 0.0, e1)
    layer1 = np.vstack((m1, x1, x2))
    # mirror symmetry
    layer2 = layer1.copy()
    layer2[:, 1] = -layer2[:, 1]
    mask = (layer2[:, 1]!=0.0)

    gb = np.vstack((layer1, layer2[mask]))

    return gb


def mx2_metalic_dopant(symbol='MoS2', dopant='Fe', w=30):
    atoms = mx2(symbol=symbol, w=w, ref='m', angle=0, t=None)
    a=atoms[:, 1:3]
    b = np.array([ 0.,  0.])
    ind = np.where(np.all(a==b,axis=1))[0][0]
    atoms[ind, 0] = dopant
    return atoms


def write_xyz(atoms, file_name):
    num_atoms = len(atoms)

    with open(file_name, 'w') as fout:
        fout.truncate(0)
        fout.write(str(num_atoms) + '\n')
        fout.write(file_name[:-4] + '\n')
        for e in atoms:
            ss = '{:6}{:12.4}{:12.4}{:12.4}'.format(e[0], e[1], e[2], e[3])
            fout.write(ss + '\n')


def generate_2d_mat_models(w=30):
    l = ['MoS2', 'MoSe2', 'WS2', 'WSe2']
    for symbol in l:
        atoms = mx2(symbol, w)
        file_name = 'monolayer_'+symbol+'.xyz'
        write_xyz(atoms, file_name)


#=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
# functions to generate MTB
#=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=

def generate_translations(a, num, move='up'):
    ty = a/2*np.sqrt(3)
    if move == 'up':
        sign = 1
    elif move == 'down':
        sign = -1
    t1 = np.array([a/2, sign*ty, 0])
    t2 = np.array([-a/2, sign*ty, 0])
    t = [(0, 0, 0)]
    t += [(i//2+1)*t1+np.ceil(i/2)*t2 for i in range(num)]
    return t

def mtb_generate_cells_from_x(x, d=None, move='up'):
    atom1, atom2 = x[0], x[1]
    # Caculate lattice constant a from atom1 and atom2
    a = np.abs(atom2[0] - atom1[0])
    if d is None:
        d = a/np.sqrt(3)
    d1 = a/np.sqrt(3)
    if move == 'up':
        sign = 1
    elif move == 'down':
        sign = -1
    l = []
    for (x0, y0, z0) in x:
        l.append([x0-a/2, y0+sign*d/2, 0.])
        l.append([x0-a/2, y0+sign*(d/2+d1), a/2])
        l.append([x0-a/2, y0+sign*(d/2+d1), -a/2])
    x0, y0, z0 = x[-1]
    l.append([x0+a/2, y0+sign*d/2, 0.])
    l.append([x0+a/2, y0+sign*(d/2+d1), a/2])
    l.append([x0+a/2, y0+sign*(d/2+d1), -a/2])
    return np.array(l)

def translate(pts, t):
    dim = pts.shape[1]
    data_copy = pts.copy()
    for i in range(dim):
        data_copy[:, i] +=  t[i]
    return data_copy

def prepend_elements(pts, elements):
    N = pts.shape[0]
    n = len(elements)
    num = int(N/n)
    elements = elements*num
    return np.array([[e, x, y, z] for e, (x, y, z) in zip(elements, pts)], dtype=object)

def mtb_new(symbol, width, up, down, d=None):
    a, c = get_a_and_c(symbol)
    e1, e2 = get_elements(symbol)
    if d is None:
        d = a/np.sqrt(3)
    n = np.floor(width / 2 / a).astype(np.int)
    x1 = np.array([(i * a, 0., a / 2) for i in np.arange(-n, n + 1)])
    x2 = np.array([(i * a, 0., -a / 2) for i in np.arange(-n, n + 1)])
    # Generate lower cells
    l1 = mtb_generate_cells_from_x(x1, d, move='down')
    # Generate upper cells
    l2 = mtb_generate_cells_from_x(x1, d, move='up')
    # Generate lower region
    t1 = generate_translations(a, down, move='down')
    atoms1 = np.vstack([translate(l1, t) for t in t1])
    # Generate upper region
    t2 = generate_translations(a, up, move='up')
    atoms2 = np.vstack([translate(l2, t) for t in t2])
    # prepend elements
    atoms1 = prepend_elements(atoms1, [e1, e2, e2])
    atoms2 = prepend_elements(atoms2, [e1, e2, e2])
    x1 = prepend_elements(x1, [e2])
    x2 = prepend_elements(x2, [e2])
    return np.vstack((atoms1, atoms2, x1, x2))



