import numpy as np
from scipy.spatial import cKDTree


def rotation_matrix(angle):
    angle = np.radians(angle)
    s = np.sin(angle)
    c = np.cos(angle)
    R = np.array([(c, -s), (s, c)])
    return process_zeros(R)


def process_zeros(data, eps=1e-9):
    data[np.abs(data) < eps] = 0.0
    return data


class ptsarray(np.ndarray):
    def __new__(cls, input_array, lbs=None):
        # Input array is an already formed ndarray instance
        # We first cast to be our class type
        obj = np.asarray(input_array).view(cls)
        obj.lbs = lbs
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
    def extents(self):
        return tuple([self.x.min(), self.x.max(), self.y.min(), self.y.max()])

    def clear_border(self, size=10, shape=None):
        if shape is None:
            xmin, xmax, ymin, ymax = self.extents
        else:
            xmin, xmax, ymin, ymax = 0, shape[1], 0, shape[0]
        mask1 = np.logical_and(self.x > xmin + size, self.x < xmax - size)
        mask2 = np.logical_and(self.y > ymin + size, self.y < ymax - size)
        mask = np.logical_and(mask1, mask2)

        if self.lbs is None:
            pts = ptsarray(self[mask], lbs=self.lbs)
        else:
            pts = ptsarray(self[mask], lbs=self.lbs[mask])
        return pts

    def rotate(self, angles):
        if np.isscalar(angles):
            R = rotation_matrix(angles)
            pts = np.dot(self, R)
        else:
            pts = [e.dot(rotation_matrix(a)) for e, a in zip(self, angles)]
            pts = ptsarray(pts, lbs=self.lbs)
        return pts

    def merge(self, r=4):
        tree = cKDTree(self)
        rows_to_fuse = tree.query_pairs(r=r, output_type='ndarray')
        pts_mean = np.asarray(self)[rows_to_fuse].mean(axis=1)
        ind = np.unique(rows_to_fuse)
        pts = np.delete(self, ind, axis=0)
        pts = np.vstack([pts, pts_mean])
        return ptsarray(pts)

    def add(self, pts):
        pts = np.unique(np.vstack([self, pts]), axis=1)
        return ptsarray(pts)

    def remove(self, inds):
        return np.delete(self, inds, axis=0)

    def extract_patches(self, img, size, flat=False):
        pts = self.clear_border(size=size, shape=img.shape)
        if flat == True:
            patches = np.array([img[y - size:y + size + 1, x - size:x + size + 1].flatten() for (x, y) in pts])
        else:
            patches = np.array([img[y - size:y + size + 1, x - size:x + size + 1] for (x, y) in pts])
        return ptsarray(pts, pts.lbs), patches

    def __array_ufunc__(self, ufunc, method, *inputs, out=None, **kwargs):
        args = []
        in_no = []
        for i, input_ in enumerate(inputs):
            if isinstance(input_, ptsarray):
                in_no.append(i)
                args.append(input_.view(np.ndarray))
            else:
                args.append(input_)

        outputs = out
        if outputs:
            out_args = []
            for j, output in enumerate(outputs):
                if isinstance(output, ptsarray):
                    out_args.append(output.view(np.ndarray))
                else:
                    out_args.append(output)
            kwargs['out'] = tuple(out_args)
        else:
            outputs = (None,) * ufunc.nout

        results = super(ptsarray, self).__array_ufunc__(ufunc, method, *args, **kwargs)
        if results is NotImplemented:
            return NotImplemented

        if ufunc.nout == 1:
            results = (results,)

        # this part is different from the doc
        results_list = []
        for result, output in zip(results, outputs):
            if output is None:
                if np.asarray(result).shape[-1] == 2:
                    results_list.append(np.asarray(result).view(ptsarray))
                else:
                    results_list.append(np.asarray(result))
            else:
                results_list.append(output)

        return results_list[0] if len(results_list) == 1 else results_list

    def __array_finalize__(self, obj):
        if obj is None: return
        self.lbs = getattr(obj, 'lbs', None)