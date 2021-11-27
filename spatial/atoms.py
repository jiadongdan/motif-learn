import numpy as np

class atoms(np.ndarray):

    def __new__(cls, xyz, lbs):
        obj = np.asarray(xyz).view(cls)
        obj.lbs = np.array(lbs)

        return obj

    @property
    def x(self):
        if len(self.shape) == 1:
            return self[0]
        else:
            return self[:, 0].view(np.ndarray)

    @property
    def y(self):
        if len(self.shape) == 1:
            return self[1]
        else:
            return self[:, 1].view(np.ndarray)

    @property
    def z(self):
        if len(self.shape) == 1:
            return self[2]
        else:
            return self[:, 2].view(np.ndarray)

    def select(self, ind):
        xyz = self[ind]
        lbs = self.lbs[ind]
        return atoms(xyz, lbs)

    def add(self, a):
        xyz = np.vstack([self, a])
        lbs = np.hstack([self.lbs, a.lbs])
        return atoms(xyz, lbs)

    def __array_ufunc__(self, ufunc, method, *inputs, out=None, **kwargs):
        args = []
        in_no = []
        for i, input_ in enumerate(inputs):
            if isinstance(input_, atoms):
                in_no.append(i)
                args.append(input_.view(np.ndarray))
            else:
                args.append(input_)

        outputs = out
        if outputs:
            out_args = []
            for j, output in enumerate(outputs):
                if isinstance(output, atoms):
                    out_args.append(output.view(np.ndarray))
                else:
                    out_args.append(output)
            kwargs['out'] = tuple(out_args)
        else:
            outputs = (None,) * ufunc.nout

        results = super(atoms, self).__array_ufunc__(ufunc, method, *args, **kwargs)
        if results is NotImplemented:
            return NotImplemented

        if ufunc.nout == 1:
            results = (results,)

        # this part is different from the doc
        results_list = []
        for result, output in zip(results, outputs):
            if output is None:
                result_ = np.asarray(result)
                if len(result_.shape)!=0 and  result_.shape[-1]== 3:
                    results_list.append(result_.view(atoms))
                else:
                    results_list.append(result)
            else:
                results_list.append(output)

        return results_list[0] if len(results_list) == 1 else results_list

    def __array_finalize__(self, obj):
        if obj is None: return
        self.lbs = getattr(obj, 'lbs', None)


def rot_matrix(angle=30):
    theta = np.radians(angle)
    c, s = np.cos(theta), np.sin(theta)
    R = np.array(((c, -s), (s, c)))
    return process_zeros(R)


def process_zeros(data, eps=1e-9):
    data[np.abs(data) < eps] = 0.0
    return data


def _generate_lattice(u, v, n, m=None, z=0., dx=0, dy=0, element='X', clip=False):
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

class mos2(atoms):

    def __new__(cls, n, m=None, z=0, theta=0, center='Mo', clip=False, elements=('Mo', 'S1', 'S2')):
        a = 3.12
        u = np.array([a, 0])
        v = np.array([a / 2, np.sqrt(3) / 2 * a])
        R = rot_matrix(theta)
        u = R.dot(u)
        v = R.dot(v)

        dx, dy = (u + v) / 3
        mo, s1, s2 = elements
        if center == 'Mo':
            Mo, lbs1 = _generate_lattice(u, v, n, m, z=z, element=mo, clip=clip)
            S1, lbs2 = _generate_lattice(u, v, n, m, z=-a / 2 + z, dx=dx, dy=dy, element=s2, clip=clip)
            S2, lbs3 = _generate_lattice(u, v, n, m, z=a / 2 + z, dx=dx, dy=dy, element=s1, clip=clip)
        elif center == 'S':
            S1, lbs2 = _generate_lattice(u, v, n, m, z=a / 2 + z, element=s1, clip=clip)
            S2, lbs3 = _generate_lattice(u, v, n, m, z=-a / 2 + z, element=s2, clip=clip)
            Mo, lbs1 = _generate_lattice(u, v, n, m, z=z, dx=dx, dy=dy, element=mo, clip=clip)
        pts = np.vstack([Mo, S1, S2])

        obj = np.asarray(pts).view(cls)
        obj.lbs = np.array(lbs1 + lbs2 + lbs3)

        return obj

    @property
    def mo(self):
        mo = np.unique(self.lbs)[0]
        return atoms(self[self.lbs==mo], self.lbs[self.lbs==mo])

    @property
    def s1(self):
        s1 = np.unique(self.lbs)[1]
        return atoms(self[self.lbs==s1], self.lbs[self.lbs==s1])

    @property
    def s2(self):
        s2 = np.unique(self.lbs)[2]
        return atoms(self[self.lbs==s2], self.lbs[self.lbs==s2])