import numpy as np
from ._honeycomb_lattice import HoneyCombLattice

def get_zps_test_patches(size=64, n_fold=3, num_patches=10, include_center=True, relative_center_intensity=1):
    """
    Generate multiple Gaussian blob patches, each with different rotational orientations.

    Args:
        size: Patch size in pixels
        n_fold: N-fold rotational symmetry for blob arrangement
        num_patches: Number of different patch orientations to generate
        include_center: Whether to include a center Gaussian blob
        relative_center_intensity: Intensity of center blob relative to others

    Returns:
        patches: (num_patches, size, size) numpy array with Gaussian blob patches
    """
    import numpy as np

    patches = []

    # Create coordinate grid
    y, x = np.ogrid[:size, :size]
    center = size / 2

    # Gaussian parameters
    sigma = size / 10  # Width of Gaussian blobs

    # Generate patches with different rotational angles
    angles = np.linspace(0, 2 * np.pi, num_patches, endpoint=False)

    for rotation_angle in angles:
        # Initialize empty patch
        patch = np.zeros((size, size), dtype=np.float32)

        # Add center blob if requested
        if include_center:
            center_gaussian = np.exp(-((x - center)**2 + (y - center)**2) / (2 * sigma**2))
            patch += relative_center_intensity * center_gaussian

        # Calculate angular sectors based on n-fold symmetry
        sector_angle = 2 * np.pi / n_fold
        radius = size / 3

        # Place blobs at symmetry-related positions
        for fold in range(n_fold):
            angle = fold * sector_angle + rotation_angle

            blob_x = center + radius * np.cos(angle)
            blob_y = center + radius * np.sin(angle)

            # Add Gaussian blob at this position
            gaussian = np.exp(-((x - blob_x)**2 + (y - blob_y)**2) / (2 * sigma**2))
            patch += gaussian

        # Normalize
        patch = patch / patch.max()
        patches.append(patch)

    return np.array(patches)

def get_zps_test_image():
    lattice = HoneyCombLattice(size=512, l=12)
    img = lattice.to_image()
    return img