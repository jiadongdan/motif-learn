import numpy as np
from scipy.special import factorial
from scipy.signal import fftconvolve
from sklearn.base import BaseEstimator, TransformerMixin
from typing import Optional, Tuple

from ._zmoments import zmoments


class ZPs(BaseEstimator, TransformerMixin):
    """
    Zernike Polynomials transformer for computing Zernike moments.

    Parameters
    ----------
    n_max : int
        Maximum radial order.
    size : int
        Size of the polynomial grid (size x size).
    """

    def __init__(self, n_max: int, size: int):
        if n_max < 0:
            raise ValueError("n_max must be non-negative.")
        if size <= 0:
            raise ValueError("size must be positive.")

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
        """Return the generated Zernike polynomials."""
        return self.polynomials

    def fit(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> 'ZPs':
        """Fit method (no-op for compatibility with sklearn)."""
        return self

    def transform(self, images: np.ndarray) -> zmoments:
        """
        Transform images to Zernike moments.

        Parameters
        ----------
        images : np.ndarray
            2D image or 3D array of images.

        Returns
        -------
        zmoments
            Zernike moments object.
        """
        if images.ndim == 2:
            return self._transform_fft_convolve(images)
        elif images.ndim == 3:
            return self._transform_dot_product(images)
        else:
            raise ValueError("Images must be 2D or 3D array.")

    def _validate_size(self, images: np.ndarray) -> None:
        """Validate that image size matches polynomial size."""
        if images.ndim == 2:
            height, width = images.shape
        elif images.ndim == 3:
            _, height, width = images.shape
        else:
            raise ValueError("Images must be 2D or 3D array.")

        if height != self.size or width != self.size:
            raise ValueError(
                f"Image size ({height}x{width}) must match polynomial size "
                f"({self.size}x{self.size})"
            )

    def _transform_dot_product(self, images: np.ndarray) -> zmoments:
        """Transform using dot product method."""
        self._validate_size(images)
        num_images = images.shape[0]

        reshaped_polynomials = self.polynomials.reshape(-1, self.size * self.size)
        reshaped_images = images.reshape(num_images, self.size * self.size)

        area = np.pi * self.size ** 2 / 4
        zernike_moments = np.dot(reshaped_images, reshaped_polynomials.T) / area

        return zmoments(data=zernike_moments, n=self.n, m=self.m)

    def _transform_fft_convolve(self, image: np.ndarray) -> zmoments:
        """Transform using FFT convolution method."""
        self._validate_size(image)

        shape = (len(self.n), image.shape[0], image.shape[1])
        image_broadcast = np.broadcast_to(image, shape)
        zernike_moments = fftconvolve(image_broadcast, self.polynomials, mode='same', axes=[1, 2])

        # Parity-based sign correction: f = (-1)^n
        # FFT convolution computes ∫∫ f(x,y) * V_nm(-x,-y) dx dy (note the sign flip)
        # but Zernike moments require ∫∫ f(x,y) * V_nm(x,y) dx dy
        # Due to Zernike symmetry: V_nm(-x,-y) = (-1)^n * V_nm(x,y)
        # Therefore, we must multiply by (-1)^n to correct for this coordinate inversion
        f = 1 - self.n % 2
        f[f == 0] = -1
        f = f[:, np.newaxis, np.newaxis]

        area = np.pi * self.size ** 2 / 4
        zernike_moments = f * zernike_moments / area

        return zmoments(data=zernike_moments, n=self.n, m=self.m)

    def fit_transform(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> zmoments:
        """Fit and transform (fit is no-op)."""
        return self.fit(X).transform(X)