import numpy as np

from skimage.transform import warp
from skimage.transform import SimilarityTransform
# from skimage.feature import register_translation
from skimage.registration import phase_cross_correlation

def get_roi(img, roi):
    x, y, h, w = roi
    return img[y:y+h,x:x+w]

def get_translation(img1, img2, roi=None, n_iter=10, upsample_factor=100):
    h, w = img1.shape
    if roi == None:
        roi = (10, 10, h-20, w-20)
    ij = np.array([0, 0])
    for i in np.arange(0, n_iter):
        shift, error, diffphase = phase_cross_correlation(img1, img2, upsample_factor = upsample_factor)
        j, i = -shift
        ij = ij + np.array([i, j])
        if i == 0.0 and j == 0.0:
            break
        tform = SimilarityTransform(translation=(i, j))
        warped = warp(img2, tform)
        # Update img1 and img2
        img1 = get_roi(img1, roi)
        img2 = get_roi(warped, roi)
    return ij

def calculate_ij(data, roi=None, upsample_factor=100):
    img_stack = data
    n = data.shape[0]
    ij = [(0, 0)]
    for ind in np.arange(0, n-1):
        img1 = img_stack[ind]
        img2 = img_stack[ind+1]
        shift = get_translation(img1, img2, roi=roi, n_iter=10, upsample_factor=upsample_factor)
        print('{}, '.format(ind), end='')
        i, j = shift
        i = i + ij[ind][0]
        j = j + ij[ind][1]
        ij.append((i, j))
    ij = np.array(ij)
    return ij

def align_imgs_ij(data, ij):
    l = []
    num_images = data.shape[0]
    for ind in np.arange(0, num_images):
        i, j = ij[ind]
        tform = SimilarityTransform(translation=(i, j))
        warped = warp(data[ind], tform)
        l.append(warped)
    return np.array(l)

def align_images(data, roi=None, upsample_factor=100):
    ij = calculate_ij(data, roi=roi, upsample_factor=upsample_factor)
    data_align = align_imgs_ij(data, ij)
    return data_align, ij


# def _high_pass(img):
#     rows = np.cos(np.linspace(-np.pi/2, np.pi/2, img.shape[0]))
#     cols = np.cos(np.linspace(-np.pi/2, np.pi/2, img.shape[1]))
#     x = np.outer( rows, cols)
#     return x*(3-x)/2
#
# def _centroid(data):
#     Y, X = np.mgrid[0:data.shape[0],0:data.shape[1]]
#     y = (Y*data).sum()/data.sum()
#     x = (X*data).sum()/data.sum()
#     return np.array((y,x))
#
# def phase_correlate_ij(img1, img2, refine = 'centroid', high_pass = False):
#     if high_pass == True:
#         # High pass filter reduce edge efect
#         img1 = img1 * _high_pass(img1)
#         img2 = img2 * _high_pass(img2)
#
#     fft1 = np.fft.fftshift(np.fft.fft2(img1))
#     fft2 = np.fft.fftshift(np.fft.fft2(img2))
#     t = fft1*np.conjugate(fft2)
#     m = t / np.abs(t)
#     correlate_map = np.abs(np.fft.fftshift(np.fft.ifft2(m)))
#     # Find the indices of the max value in correlation map
#     i,j =  np.unravel_index( np.argmax(correlate_map), correlate_map.shape)
#     if refine == "centroid":
#         p = np.array([np.array(_centroid(correlate_map[i-size:i+size+1,j-size:j+size+1]))-size for size in np.arange(2,6)])
#         di,dj = p[:,0].mean(), p[:,1].mean()
#         i,j = i+di, j+dj
#     return i, j
#
# def image_align(img_stack):
#     empty = []
#     ij = []
#     temp = img_stack[0].copy()
#     for img in img_stack:
#         if np.array_equal(img, temp):
#             i, j = phase_correlate_ij(temp, img)
#         else:
#             i, j = phase_correlate_ij(temp, img, refine="centroid")
#         tform = SimilarityTransform(translation=(-j + img.shape[1] // 2, -i + img.shape[0] // 2))
#         warped = warp(img, tform)
#         empty.append(warped)
#         temp = warped
#         ij.append([i, j])
#     ij = np.array(ij)
#     std_i, std_j = np.std(ij[:, 0]), np.std(ij[:, 1])
#     print("std_i:{: >20} std_j:{: >20}".format(std_i, std_j))
#     return np.array(empty)


def image_align_deprecated(data, roi = None):
    empty = []
    if roi == None:
        x, y = (0, 0)
        h, w = data.shape[1:]
    else:
        x, y, h, w = roi
    img_stack = data[:, y:y+h, x:x+w]
    ref = img_stack[0].copy()
    print('Starting...')
    for e, img in enumerate(img_stack):
        shift, error, diffphase = phase_cross_correlation(ref, img, upsample_factor = 100)
        i, j = shift
        tform = SimilarityTransform(translation=(-j, -i))
        warped = warp(data[e], tform)
        empty.append(warped)
        print('{}, '.format(e), end='')
    return np.array(empty)


def image_align_ij_deprecated(data, roi=None):
    empty = []
    if roi == None:
        x, y = (0, 0)
        h, w = data.shape[1:]
    else:
        x, y, h, w = roi
    img_stack = data[:, y:y + h, x:x + w]
    ref = img_stack[0].copy()
    print('Starting...')
    for e, img in enumerate(img_stack):
        shift, error, diffphase = phase_cross_correlation(ref, img, upsample_factor=100)
        i, j = shift
        empty.append([i, j])
    return np.array(empty)