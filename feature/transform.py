import numpy as np
#from skimage.feature import register_translation
from skimage.registration import phase_cross_correlation
from skimage.morphology import disk
from skimage.transform import rotate, warp_polar
from sklearn.utils import check_random_state


def register_rotation(img1, img2, upsample_factor=20):
    mask = disk((img1.shape[0]-1)/2)
    img1_polar = warp_polar(img1 * mask)
    img2_polar = warp_polar(img2 * mask)
    shifts, error, phasediff = phase_cross_correlation(img1_polar, img2_polar, upsample_factor=upsample_factor)
    return shifts[0]

def _register_imgs(imgs, upsample_factor=20, return_angles=False):
    ref = imgs[0]
    angles = []
    for i, img in enumerate(imgs):
        a = register_rotation(img, ref, upsample_factor)
        angles.append(a)
        img_rot = rotate(img, angle=a, order=1)
        ref = (ref * (i + 1) + img_rot) / (i + 2)
    if return_angles:
        return ref, np.array(angles)
    else:
        return ref

def register_imgs(imgs, max_samples=15, seed=48):
    if len(imgs) > max_samples:
        rng = check_random_state(seed=seed)
        mask = rng.choice(len(imgs), max_samples, replace=False)
        data = imgs[mask]
    else:
        data = imgs
    return _register_imgs(data)

def rotation_matrix(angle):
    angle = np.radians(angle)
    s = np.sin(angle)
    c = np.cos(angle)
    R = np.array([(c, -s), (s, c)])
    return process_zeros(R)


def process_zeros(data, eps=1e-9):
    data[np.abs(data) < eps] = 0.0
    return data


def rotate_pts(pts, angle):
    pts = np.array(pts)
    R = rotation_matrix(angle)
    return np.dot(pts, R)


def many_gaussians(pts, sigma, s):
    data = np.zeros((s, s))
    Y, X = np.ogrid[-s // 2:s // 2:1j * s, -s // 2:s // 2:1j * s]
    for (x, y) in pts:
        data = data + np.exp((-(X - x) ** 2 - (Y - y) ** 2) / (2 * sigma * sigma))
    return data

def gaussians(n_fold, sigma, l, theta, size):
    p0 = np.array([0, l])
    p0 = rotate_pts(p0, theta)
    pts = np.array([rotate_pts(p0, 360*i/n_fold) for i in range(n_fold)])
    return many_gaussians(pts, sigma, size)



