import numpy as np

def _generate_lattice(shape, u, v):
    '''Helper function to generate lattice points within a rectangular
    box given special basis vectors u and v.
    Parameters
    ----------
    shape: tuple (y, x)
        tuple shape to give the range of the box
    u: array_like
        base vector u, u must lie on the positive direction of x -axis
    v: array_like
        base vector v, v cannot be out of the box regeions
    '''
    # Calculate height and width of the box
    rows, cols = shape
    m_max = np.floor(cols/u[0])
    n_max = np.floor(rows/v[1])
    dx = cols - m_max*u[0]
    l=[]
    # Calculate all points lies on the x axis
    pts = np.array([(m*u[0], 0) for m in np.arange(0, m_max+1)])
    l.append(pts)
    for n in np.arange(1, n_max+1):
        temp = pts.copy()
        # Calculate the number of points shifted out due to translation
        num_out = int(np.ceil((-dx + n*v[0])/u[0]))
        temp[:,0] = temp[:,0] + n*v[0] - num_out*u[0]
        temp[:,1] = temp[:,1] + n*v[1]
        total_shift = n*v[0] - num_out*u[0]
        if total_shift < 0:
            temp = np.delete(temp, 0, axis=0)
        l.append(temp)
    # n is negative
    for n in np.arange(-n_max, 0):
        temp = pts.copy()
        num_out = np.ceil(-n*v[0]/u[0])
        temp[:,0] = temp[:,0] + n*v[0]+ num_out*u[0]
        temp[:,1] = temp[:,1] + n*v[1]
        total_shift = n*v[0] + num_out*u[0]
        if total_shift > dx:
            temp = np.delete(temp, -1, axis=0)
        l.append(temp)
    l = np.vstack(l)
    l = np.vstack((l,-l))
    return np.unique(l, axis=0)

def generate_lattice(shape, p,  u, v):
    p = np.array(p)
    u = np.array(u)
    v = np.array(v)
    rows, cols = shape
    rows_ = int(2 * np.ceil(np.sqrt(2) * rows))
    cols_ = int(2 * np.ceil(np.sqrt(2) * rows))
    # Defining error to treat really small numbers after rotation
    eps = 1e-6
    # Calculate rotation angle in radians
    theta = np.arctan2(u[1], u[0])
    rot_matrix = np.array([[np.cos(theta), np.sin(theta)], [-np.sin(theta), np.cos(theta)]])
    u = u.dot(rot_matrix.T)
    v = v.dot(rot_matrix.T)
    # Round small values
    if np.abs(u[1]) < eps:
        u[1] = 0.
    if np.abs(v[0]) < eps:
        v[0] = 0.
    # Make v all non-negative
    if (v[0] < 0 and v[1] > 0):
        v[0] = v[0] + u[0]
        v[1] = v[1] + u[1]
    if (v[0] < 0 and v[1] < 0):
        v[0] = -v[0]
        v[1] = -v[1]
    if (v[0] > 0 and v[1] < 0):
        v[0] = -v[0] + u[0]
        v[1] = -v[1] + u[1]
    lattice_ = _generate_lattice((rows_, cols_), u, v)
    rot_matrix = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
    lattice_ = lattice_.dot(rot_matrix.T)
    lattice_[:, 0] = lattice_[:, 0] + p[0]
    lattice_[:, 1] = lattice_[:, 1] + p[1]
    # check box boundaries
    lattice_[:, 0][np.abs(lattice_[:, 0]) < eps] = 0.0
    lattice_[:, 1][np.abs(lattice_[:, 1]) < eps] = 0.0
    lattice_[:, 0][np.abs(lattice_[:, 0] - cols) < eps] = cols
    lattice_[:, 1][np.abs(lattice_[:, 1] - rows) < eps] = rows
    lattice = np.array([(p[0], p[1]) for p in lattice_ if (p[0] >= 0 and p[0] <= cols and p[1] >= 0 and p[1] <= rows)])
    return lattice