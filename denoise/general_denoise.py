import numpy as np
from skimage.morphology import white_tophat
from skimage.morphology import disk
from skimage.transform import warp_polar


def denoise_fft_(img, fraction=0.1, band_width=0.5):
    """Multi-dimensional Gaussian filter.
    Parameters
    ----------
    img : array-like
        Input image or images to filter.
    fraction : scalar, optional
        Fraction of components in Fourier space selected for
        denoising the image. Default is 0.1.
    band_width : scalar, optional
        Band with for low frequency to pass. Default is 0.5
    Returns
    -------
    denoised_image : ndarray
        the denoised array
    Notes
    -----
    This function is a wrapper around :func:`scipy.ndi.gaussian_filter`.
    Integer arrays are converted to float.
        """
    N = np.prod(img.shape)
    fft_complex = np.fft.fft2(img)
    a = np.sort(np.abs(fft_complex).flatten())[::-1]
    t = a[int(N*fraction)]
    mask = (np.abs(fft_complex) >= t) * 1
    mask = np.fft.fftshift(mask)
    width = int(np.sqrt(N)*band_width/2)
    # pad the mask with zero
    for i in range(mask.ndim):
        mask = mask.swapaxes(0, i)
        mask[:width] = mask[-width:] = 0
        mask = mask.swapaxes(0, i)
    mask = np.fft.fftshift(mask)
    imgf = np.fft.ifft2(mask*fft_complex)
    return np.abs(imgf)


def remove_bg(img, disk_size):
    selem = disk(disk_size)
    return white_tophat(img, selem)


def denoise_fft(img, alpha=1):
    img_fft = np.fft.fft2(img)
    fft_real, fft_imag = img_fft.real, img_fft.imag
    fft_abs = np.abs(np.fft.fftshift(img_fft))
    # warp image around center
    b = warp_polar(fft_abs)

    # get radial intesnity
    l = b.mean(axis=0)
    s1 = int(len(l) // np.sqrt(2))
    s2 = s1 // 2
    t = l[s2: s1].mean()
    n1 = np.ceil(np.log10(t))
    n2 = np.floor(np.log10(fft_abs.max()))
    n = int(min(n1, n2))
    # this is estimated threshold
    t = alpha*10 ** n
    mask = np.fft.fftshift(fft_abs > t)
    imgf = np.abs(np.fft.ifft2((img_fft * mask)))

    return imgf


