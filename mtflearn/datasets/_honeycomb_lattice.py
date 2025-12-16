import numpy as np
from typing import Tuple, Optional
from ._tapered_gaussian import add_tapered_gaussian


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
    random_shift : bool
        If True, shift the lattice origin by a random vector within one
        primitive unit cell (spanned by a1 and a2).
    seed : int or None
        Random seed for reproducible random shift and jitter.
    jitter : float
        Standard deviation (in pixels) of Gaussian random noise applied
        independently to each A/B site position (0.0 means no jitter).
    """

    def __init__(
            self,
            size: int = 512,
            l: float = 12.0,
            a: Optional[float] = None,
            angle: float = 0.0,
            random_shift: bool = True,
            seed: Optional[int] = None,
            jitter: float = 0.0,
    ):
        self.size = int(size)
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

        # Primitive lattice vectors (unrotated)
        self.a1 = np.array([1.5 * self.l, np.sqrt(3.0) * self.l / 2.0], dtype=np.float64)
        self.a2 = np.array([1.5 * self.l, -np.sqrt(3.0) * self.l / 2.0], dtype=np.float64)

        # Basis vectors (A and B sublattices, unrotated)
        self.dA = np.array([0.0, 0.0], dtype=np.float64)
        self.dB = np.array([self.l, 0.0], dtype=np.float64)

        # RNG for all randomness
        self.rng = np.random.default_rng(seed)

        # Random shift parameters
        self.random_shift = random_shift
        if random_shift:
            self.shift_u1, self.shift_u2 = self.rng.random(2)
        else:
            self.shift_u1 = 0.0
            self.shift_u2 = 0.0

        # Jitter strength
        self.jitter = float(jitter)

        # Over-generate in lattice index space
        self.N = int(np.ceil(self.size / self.l)) + 3

        # Cached coordinates (all in final rotated + shifted + perturbed positions)
        self._coords_A: Optional[np.ndarray] = None
        self._coords_B: Optional[np.ndarray] = None

    def _update_rotation(self) -> None:
        """Update rotation matrix from current angle."""
        c, s = np.cos(self.angle), np.sin(self.angle)
        self.Rmat = np.array([[c, -s], [s, c]], dtype=np.float64)

    def set_angle(self, angle: float) -> None:
        """Set a new rotation angle in degrees and invalidate cache."""
        self.angle_deg = float(angle)
        self.angle = np.deg2rad(self.angle_deg)
        self._coords_A = None
        self._coords_B = None

    def _generate_coordinates(self) -> None:
        """Generate all lattice coordinates (rotation + shift + perturbation applied)."""
        # Update rotation matrix
        self._update_rotation()

        # Compute random shift in unrotated frame
        offset = self.shift_u1 * self.a1 + self.shift_u2 * self.a2

        # Generate all lattice points in unrotated frame
        coords_A = []
        coords_B = []
        for n1 in range(-self.N, self.N + 1):
            for n2 in range(-self.N, self.N + 1):
                R = n1 * self.a1 + n2 * self.a2
                coords_A.append(R + self.dA + offset)
                coords_B.append(R + self.dB + offset)

        coords_A = np.asarray(coords_A, dtype=np.float64)
        coords_B = np.asarray(coords_B, dtype=np.float64)

        # Apply rotation
        coords_A = coords_A @ self.Rmat.T
        coords_B = coords_B @ self.Rmat.T

        # Center in box
        box_center = np.array([self.size / 2.0, self.size / 2.0], dtype=np.float64)
        coords_A += box_center
        coords_B += box_center

        # Add random jitter
        if self.jitter > 0.0:
            coords_A += self.rng.normal(0.0, self.jitter, coords_A.shape)
            coords_B += self.rng.normal(0.0, self.jitter, coords_B.shape)

        # Cache the results
        self._coords_A = coords_A
        self._coords_B = coords_B

    def get_points(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Return A and B sublattice coordinates clipped to [0, size).

        Returns
        -------
        pts_A : (N_A, 2) ndarray
            A sublattice coordinates inside the box.
        pts_B : (N_B, 2) ndarray
            B sublattice coordinates inside the box.
        """
        # Ensure coordinates are generated
        if self._coords_A is None or self._coords_B is None:
            self._generate_coordinates()

        # Clip to box
        L = self.size
        mask_A = (
                (self._coords_A[:, 0] >= 0.0) & (self._coords_A[:, 0] < L) &
                (self._coords_A[:, 1] >= 0.0) & (self._coords_A[:, 1] < L)
        )
        mask_B = (
                (self._coords_B[:, 0] >= 0.0) & (self._coords_B[:, 0] < L) &
                (self._coords_B[:, 1] >= 0.0) & (self._coords_B[:, 1] < L)
        )

        return self._coords_A[mask_A], self._coords_B[mask_B]

    def to_image(
            self,
            sigma: Optional[float] = None,
            intensity_A: float = 1.0,
            intensity_B: float = 0.5,
            normalize: bool = False,
    ) -> np.ndarray:
        """
        Rasterize the lattice into a 2D image with Gaussian blobs.
        Includes contributions from points outside the box whose Gaussians overlap.

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
        # Ensure coordinates are generated
        if self._coords_A is None or self._coords_B is None:
            self._generate_coordinates()

        if sigma is None:
            sigma = self.l / 4.0

        img = np.zeros((self.size, self.size), dtype=np.float32)
        radius = int(np.ceil(3 * sigma))

        # Filter to points whose Gaussian support intersects the box
        def in_range(coords: np.ndarray) -> np.ndarray:
            if coords.size == 0:
                return coords
            mask = (
                    (coords[:, 0] >= -radius) & (coords[:, 0] <= self.size - 1 + radius) &
                    (coords[:, 1] >= -radius) & (coords[:, 1] <= self.size - 1 + radius)
            )
            return coords[mask]

        coords_A_render = in_range(self._coords_A)
        coords_B_render = in_range(self._coords_B)

        add_tapered_gaussian(img, coords_A_render, sigma, amplitude=intensity_A, r_factor=3.0)
        add_tapered_gaussian(img, coords_B_render, sigma, amplitude=intensity_B, r_factor=3.0)

        if normalize:
            vmax = float(img.max())
            if vmax > 0.0:
                img /= vmax

        return img