import numpy as np
from typing import Tuple, Optional


class HoneyCombLattice:
    """
    Graphene-like 2D honeycomb lattice in a square simulation box.

    Parameters
    ----------
    size : int
        Width and height of the simulation box (pixels).
    l : float, optional
        Nearest-neighbor bond length (default 12.0 in pixel units).
    a : float, optional
        Lattice constant of the 2D hexagonal Bravais lattice.
        For ideal graphene: a = sqrt(3) * l.
        If provided together with l, it is checked for consistency.
    angle : float
        Rotation angle of the lattice in degrees (counter-clockwise).
    """

    def __init__(
            self,
            size: int = 512,
            l: float = 12.0,
            a: Optional[float] = None,
            angle: float = 0.0,
    ):
        self.size = int(size)

        # Bond length (nearest-neighbor distance)
        self.l = float(l)

        # Lattice constant derived from bond length
        inferred_a = self.l * np.sqrt(3.0)

        if a is not None:
            a = float(a)
            if not np.isclose(a, inferred_a, rtol=1e-5, atol=1e-6):
                raise ValueError(
                    f"Inconsistent 'a' and 'l': got a={a}, l={self.l}, "
                    f"but for ideal graphene expect a≈sqrt(3)*l≈{inferred_a:.6f}."
                )
            self.a = a
        else:
            self.a = inferred_a

        # Rotation
        self.angle_deg = float(angle)
        self.angle = np.deg2rad(self.angle_deg)

        # Primitive lattice vectors (triangular Bravais lattice built from bond length)
        # This matches your original, working honeycomb construction:
        #   a1 = (3/2 l,  +√3/2 l)
        #   a2 = (3/2 l,  -√3/2 l)
        self.a1 = np.array([1.5 * self.l,  np.sqrt(3.0) * self.l / 2.0], dtype=np.float64)
        self.a2 = np.array([1.5 * self.l, -np.sqrt(3.0) * self.l / 2.0], dtype=np.float64)

        # Basis vectors (A and B sublattices)
        self.dA = np.array([0.0, 0.0], dtype=np.float64)
        self.dB = np.array([self.l, 0.0], dtype=np.float64)  # exactly one bond length away

        # Rotation matrix
        self._update_rotation()

        # Over-generate in lattice index space and then clip
        self.N = int(np.ceil(self.size / self.l)) + 3
        self.center = np.array([self.size / 2.0, self.size / 2.0], dtype=np.float64)

        # Cached coordinates
        self.pts_A: Optional[np.ndarray] = None
        self.pts_B: Optional[np.ndarray] = None

    # -----------------------------
    # Rotation and angle control
    # -----------------------------
    def _update_rotation(self) -> None:
        c, s = np.cos(self.angle), np.sin(self.angle)
        self.Rmat = np.array([[c, -s], [s, c]], dtype=np.float64)

    def set_angle(self, angle: float) -> None:
        """Set a new rotation angle in degrees and clear cached points."""
        self.angle_deg = float(angle)
        self.angle = np.deg2rad(self.angle_deg)
        self._update_rotation()
        self.pts_A = None
        self.pts_B = None

    # -----------------------------
    # Lattice generation
    # -----------------------------
    def generate(self) -> Tuple[np.ndarray, np.ndarray]:
        """Generate A and B sublattice coordinates inside the simulation box."""
        coords_A = []
        coords_B = []

        for n1 in range(-self.N, self.N + 1):
            for n2 in range(-self.N, self.N + 1):
                R = n1 * self.a1 + n2 * self.a2
                coords_A.append(R + self.dA)
                coords_B.append(R + self.dB)

        coords_A = np.asarray(coords_A, dtype=np.float64)
        coords_B = np.asarray(coords_B, dtype=np.float64)

        # Rotate
        coords_A = coords_A @ self.Rmat.T
        coords_B = coords_B @ self.Rmat.T

        # Center in box
        coords_A += self.center
        coords_B += self.center

        # Clip to [0, size)
        L = self.size
        mask_A = (
                (coords_A[:, 0] >= 0.0) & (coords_A[:, 0] < L) &
                (coords_A[:, 1] >= 0.0) & (coords_A[:, 1] < L)
        )
        mask_B = (
                (coords_B[:, 0] >= 0.0) & (coords_B[:, 0] < L) &
                (coords_B[:, 1] >= 0.0) & (coords_B[:, 1] < L)
        )

        self.pts_A = coords_A[mask_A]
        self.pts_B = coords_B[mask_B]

        return self.pts_A, self.pts_B

    def get_points(self) -> Tuple[np.ndarray, np.ndarray]:
        """Return A and B sublattice coordinates, generating them if needed."""
        if self.pts_A is None or self.pts_B is None:
            return self.generate()
        return self.pts_A, self.pts_B

    # -----------------------------
    # Image rasterization
    # -----------------------------
    def to_image(
            self,
            sigma: Optional[float] = None,
            intensity_A: float = 1.0,
            intensity_B: float = 0.5,
            normalize: bool = False,
    ) -> np.ndarray:
        """
        Rasterize the lattice into a 2D image with Gaussian blobs.

        Parameters
        ----------
        sigma : float or None
            Gaussian std. dev. in pixels. If None, defaults to l/4.
        intensity_A : float
            Amplitude for A-sublattice.
        intensity_B : float
            Amplitude for B-sublattice.
        normalize : bool
            If True, divide image by its max value.

        Returns
        -------
        img : (size, size) ndarray
            Rendered image.
        """
        pts_A, pts_B = self.get_points()
        size = self.size

        if sigma is None:
            sigma = self.l / 4.0

        img = np.zeros((size, size), dtype=np.float32)

        # Precompute Gaussian kernel
        radius = int(np.ceil(3 * sigma))
        y, x = np.mgrid[-radius:radius + 1, -radius:radius + 1]
        kernel = np.exp(-(x**2 + y**2) / (2.0 * sigma**2)).astype(np.float32)

        def add_gaussians(points: np.ndarray, amplitude: float) -> None:
            for px, py in points:
                cx = int(round(px))
                cy = int(round(py))

                if cx < 0 or cx >= size or cy < 0 or cy >= size:
                    continue

                x0, x1 = cx - radius, cx + radius + 1
                y0, y1 = cy - radius, cy + radius + 1

                ix0, ix1 = max(x0, 0), min(x1, size)
                iy0, iy1 = max(y0, 0), min(y1, size)
                if ix0 >= ix1 or iy0 >= iy1:
                    continue

                kx0, kx1 = ix0 - x0, ix1 - x0
                ky0, ky1 = iy0 - y0, iy1 - y0

                img[iy0:iy1, ix0:ix1] += amplitude * kernel[ky0:ky1, kx0:kx1]

        add_gaussians(pts_A, amplitude=intensity_A)
        add_gaussians(pts_B, amplitude=intensity_B)

        if normalize:
            vmax = float(img.max())
            if vmax > 0.0:
                img /= vmax

        return img
