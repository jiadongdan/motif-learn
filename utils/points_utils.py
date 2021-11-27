import numpy as np
from scipy.spatial.transform import Rotation as R


# rotation matrix
def rotation_matrix(angle, mode='clockwise'):
    if mode is 'clockwise':
        return R.from_euler('z', -angle, degrees=True).as_matrix()[0:2, 0:2]
    else:
        return R.from_euler('z', angle, degrees=True).as_matrix()[0:2, 0:2]

def rotate_pts(pts, angle, mode='clockwise'):
    pts = np.array(pts)
    R = rotation_matrix(angle, mode=mode)
    return np.dot(pts, R)