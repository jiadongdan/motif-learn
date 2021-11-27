import numpy as np


def rot_matrix(angle=30):
    theta = np.radians(angle)
    c, s = np.cos(theta), np.sin(theta)
    R = np.array(((c, -s), (s, c)))
    return process_zeros(R)


def process_zeros(data, eps=1e-9):
    data[np.abs(data) < eps] = 0.0
    return data


def _generate_lattice(u, v, n, m=None, z=0, dx=0, dy=0, element='X', clip=False):
    if m is None:
        m = n
    uv = np.stack([u, v])
    X, Y = np.meshgrid(range(-n, n), range(-m, m))
    xy = np.dstack([X, Y])
    xy = xy.reshape(-1, 2).dot(uv)
    xy = xy + [dx, dy]
    xyz = np.column_stack([xy, [z] * len(xy)])

    d2 = X ** 2 + Y ** 2 + X * Y
    mask = d2 <= np.unique(d2)[n]
    xyz = xyz[mask.ravel()]

    l = 2 * np.sqrt(np.unique(d2)[n])
    if clip:
        mask1 = np.logical_and(np.abs(xyz[:, 0]) <= l, np.abs(xyz[:, 1]) <= l)
        xyz = xyz[mask1]

    lbs = [element] * len(xyz)
    return xyz, lbs


class mos2(np.ndarray):

    def __new__(cls, n, m=None, z=0, theta=0, center='Mo', clip=False):
        a = 3.12
        u = np.array([a, 0])
        v = np.array([a / 2, np.sqrt(3) / 2 * a])
        R = rot_matrix(theta)
        u = R.dot(u)
        v = R.dot(v)

        dx, dy = (u + v) / 3
        if center == 'Mo':
            Mo, lbs1 = _generate_lattice(u, v, n, m, z=z, element='Mo', clip=clip)
            S1, lbs2 = _generate_lattice(u, v, n, m, z=-a / 2 + z, dx=dx, dy=dy, element='S2', clip=clip)
            S2, lbs3 = _generate_lattice(u, v, n, m, z=a / 2 + z, dx=dx, dy=dy, element='S1', clip=clip)
        elif center == 'S':
            S1, lbs2 = _generate_lattice(u, v, n, m, z=a / 2 + z, element='S1', clip=clip)
            S2, lbs3 = _generate_lattice(u, v, n, m, z=-a / 2 + z, element='S2', clip=clip)
            Mo, lbs1 = _generate_lattice(u, v, n, m, z=z, dx=dx, dy=dy, element='Mo', clip=clip)
        pts = np.vstack([Mo, S1, S2])

        obj = np.asarray(pts).view(cls)
        obj.lbs = np.array(lbs1 + lbs2 + lbs3)

        return obj

    @property
    def mo(self):
        return self[self.lbs=='Mo']

    @property
    def s1(self):
        return self[self.lbs == 'S1']

    @property
    def s2(self):
        return self[self.lbs == 'S2']

    def __array_ufunc__(self, ufunc, method, *inputs, out=None, **kwargs):
        args = []
        in_no = []
        for i, input_ in enumerate(inputs):
            if isinstance(input_, mos2):
                in_no.append(i)
                args.append(input_.view(np.ndarray))
            else:
                args.append(input_)

        outputs = out
        if outputs:
            out_args = []
            for j, output in enumerate(outputs):
                if isinstance(output, mos2):
                    out_args.append(output.view(np.ndarray))
                else:
                    out_args.append(output)
            kwargs['out'] = tuple(out_args)
        else:
            outputs = (None,) * ufunc.nout

        results = super(mos2, self).__array_ufunc__(ufunc, method, *args, **kwargs)
        if results is NotImplemented:
            return NotImplemented

        if ufunc.nout == 1:
            results = (results,)

        # this part is different from the doc
        results_list = []
        for result, output in zip(results, outputs):
            if output is None:
                if np.asarray(result).shape[-1] == 3:
                    results_list.append(np.asarray(result).view(mos2))
                else:
                    results_list.append(np.asarray(result))
            else:
                results_list.append(output)

        return results_list[0] if len(results_list) == 1 else results_list

    def __array_finalize__(self, obj):
        if obj is None: return
        self.lbs = getattr(obj, 'lbs', None)


def generate_angle_matrix(n):
    p = np.arange(1, n+1)
    q = np.arange(1, n+1)
    P, Q = np.meshgrid(p, q)
    i1 = 2*P*Q
    i2 = 3*Q**2-P**2
    i3 = 3*Q**2+P**2
    angle = np.rad2deg(np.arccos(i2/i3))
    mask1 = Q > P
    mask2 = (np.gcd(P, Q) == 1)
    angle[~mask1] = 0
    angle[~mask2] = 0
    return angle

def generate_N(n):
    p = np.arange(1, n+1)
    q = np.arange(1, n+1)
    P, Q = np.meshgrid(p, q)
    mask1 = Q > P
    mask2 = (np.gcd(P, Q) == 1)
    gamma = np.gcd(3*Q+P, 3*Q-P)
    sigma = 3/np.gcd(P, 3).astype(np.int)
    N = 3*(3*Q**2+P**2)/(sigma*gamma**2).astype(np.int)
    mask1 = Q > P
    mask2 = (np.gcd(P, Q) == 1)
    N[~mask1] = 0
    N[~mask2] = 0
    return N






