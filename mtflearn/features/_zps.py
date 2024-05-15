import numpy as np
from scipy.special import factorial
from scipy.signal import fftconvolve
from sklearn.base import BaseEstimator, TransformerMixin
from typing import Union, Optional, Tuple


class ZPs(BaseEstimator, TransformerMixin):

    def __init__(self, n_max: int, size: int):
        self.n_max = n_max
        self.size = size
        self.n, self.m, self.polynomials = self._generate_polynomials()

    def _radial_polynomial(self, n: int, m: int, rho: np.ndarray) -> np.ndarray:
        """Calculate the radial part of Zernike polynomial."""
        z = np.zeros_like(rho)
        for k in range((n - abs(m)) // 2 + 1):
            num = (-1) ** k * factorial(n - k)
            denom = (
                    factorial(k)
                    * factorial((n + abs(m)) // 2 - k)
                    * factorial((n - abs(m)) // 2 - k)
            )
            coefficient = num / denom
            z += coefficient * rho ** (n - 2 * k)
        return z

    def _generate_polynomials(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Generate Zernike polynomials and corresponding n and m values."""
        x = np.linspace(-1, 1, self.size)
        y = np.linspace(-1, 1, self.size)
        xv, yv = np.meshgrid(x, y)
        rho = np.sqrt(xv ** 2 + yv ** 2)
        theta = np.arctan2(yv, xv)

        n_values = []
        m_values = []
        zernike = []
        for n in range(self.n_max + 1):
            for m in range(-n, n + 1, 2):
                if abs(m) <= n:
                    n_values.append(n)
                    m_values.append(m)
                    radial = self._radial_polynomial(n, m, rho)
                    normalization_factor = np.sqrt(2 * (n + 1) / (1 + (m == 0)))
                    zernike_polynomial = np.where(rho <= 1, radial * normalization_factor, 0)
                    if m < 0:
                        zernike.append(zernike_polynomial * np.sin(-m * theta))
                    else:
                        zernike.append(zernike_polynomial * np.cos(m * theta))

        return np.array(n_values), np.array(m_values), np.array(zernike)

    def get_polynomials(self) -> np.ndarray:
        return self.polynomials

    def fit(self, X: np.ndarray, y: Optional[Union[np.ndarray, None]] = None):
        return self

    def transform(self, images: np.ndarray) -> np.ndarray:
        if images.ndim == 2:
            return self._transform_fft_convolve(images)
        elif images.ndim == 3:
            return self._transform_dot_product(images)

    def _transform_dot_product(self, images: np.ndarray) -> np.ndarray:
        if images.ndim != 3:
            raise ValueError("Input images should be a 3D numpy array")

        num_images, height, width = images.shape
        if height != self.size or width != self.size:
            raise ValueError("Each image size must match the initialized size")

        reshaped_polynomials = np.array(self.polynomials).reshape(-1, self.size * self.size)
        reshaped_images = images.reshape(num_images, self.size * self.size)

        area = (self.size * self.size) / 4 * np.pi
        zernike_moments = np.dot(reshaped_images, reshaped_polynomials.T) / area

        return zernike_moments

    def _transform_pseudo_inverse(self, images: np.ndarray) -> np.ndarray:
        if images.ndim != 3:
            raise ValueError("Input images should be a 3D numpy array")

        num_images, height, width = images.shape
        if height != self.size or width != self.size:
            raise ValueError("Each image size must match the initialized size")

        reshaped_polynomials = np.array(self.polynomials).reshape(-1, self.size * self.size)
        reshaped_images = images.reshape(num_images, self.size * self.size)

        pseudo_inv_polynomials = np.linalg.pinv(reshaped_polynomials)
        zernike_moments = reshaped_images.dot(pseudo_inv_polynomials)

        return zernike_moments

    def _transform_fft_convolve(self, image: np.ndarray):
        shape = (len(self.n), image.shape[0], image.shape[1])
        image = np.broadcast_to(image, shape)
        zernike_moments = fftconvolve(image, self.polynomials, mode='same', axes=[1, 2])
        f = 1 - self.n % 2
        f[f == 0] = -1
        f = f[:, np.newaxis, np.newaxis]
        area = np.pi * (self.size) ** 2 / 4
        zernike_moments = f * zernike_moments / area
        return zernike_moments

    def fit_transform(self, X: np.ndarray, y=None, **kwargs) -> np.ndarray:
        return self.fit(X).transform(X)
