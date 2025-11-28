import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import fftconvolve
from scipy.ndimage import gaussian_filter


class TMDImageSimulator:
    def __init__(self, size=(512, 512), a=30, theta=0.0,
                 species_params=None, basis=None, use_fft=True):
        self.size = size
        self.a = a
        self.theta = theta
        self.use_fft = use_fft

        self.species_params = species_params or {
            'TM': {'sigma': 2.0, 'A': 1.0},
            'X': {'sigma': 1.5, 'A': 0.6}
        }

        self.basis = basis or [
            (0.0, 0.0, 'TM'),
            (1/3, 1/3, 'X'),
            #(2/3, 2/3, 'X')
        ]

        self.vacancies = []
        self.dopants = []
        self._cached_coords = None
        self.pts_all = None
        self.pts = None
        self.labels = None
        self.lbs = None
        self.filtered_indices = {}

    def rotation_matrix(self):
        theta = np.deg2rad(self.theta)
        return np.array([
            [np.cos(theta), -np.sin(theta)],
            [np.sin(theta),  np.cos(theta)]
        ])

    def generate_lattice(self):
        self.defect_label_map = {}
        a1 = self.a * np.array([1, 0])
        a2 = self.a * np.array([0.5, np.sqrt(3)/2])
        R = self.rotation_matrix()

        height, width = self.size
        nx = int(width // self.a * 2) + 4
        ny = int(height // (self.a * np.sqrt(3)/2) * 2) + 4

        coords = {label: [] for label in set(b[2] for b in self.basis)}
        all_pts = []
        all_labels = []

        label_map = {}
        global_index = 0
        label_indices = {label: [] for label in coords}

        for i in range(-nx, nx):
            for j in range(-ny, ny):
                base = i * a1 + j * a2
                for dx, dy, label in self.basis:
                    pos = base + dx * a1 + dy * a2
                    pos = R @ pos
                    coords[label].append(pos)
                    all_pts.append(pos)
                    all_labels.append(label)
                    label_map[global_index] = label
                    label_indices[label].append(global_index)
                    global_index += 1

        for key in coords:
            coords[key] = np.array(coords[key])

        self._cached_coords = coords
        self.pts_all = np.array(all_pts)

        mask = (
                (self.pts_all[:, 0] >= 0) & (self.pts_all[:, 0] < width) &
                (self.pts_all[:, 1] >= 0) & (self.pts_all[:, 1] < height)
        )

        self.pts = self.pts_all[mask]
        raw_labels = np.array(all_labels)

        defect_labels = raw_labels.copy()
        for vac_label, vac_idx, scale in self.vacancies:
            if scale == 0.5:
                defect_labels[vac_idx] = 'v1'
            elif scale == 0.0:
                defect_labels[vac_idx] = 'v2'
        for dop_label, dop_idx, dop_scale in self.dopants:
            defect_labels[dop_idx] = 'D'

        self.labels = defect_labels[mask]
        label_to_int = {lbl: i for i, lbl in enumerate(sorted(set(defect_labels)))}
        self.lbs = np.array([label_to_int[lbl] for lbl in self.labels], dtype=int)

        visible_indices = np.where(mask)[0]
        for label in coords:
            self.filtered_indices[label] = [i for i in label_indices[label] if i in visible_indices]

        return coords

    def add_single_vacancy(self, label='X', index=None):
        if index is not None:
            self.vacancies.append((label, index, 0.5))

    def add_double_vacancy(self, label='X', indices=None):
        if indices is not None and len(indices) >= 1:
            self.vacancies.append((label, indices[0], 0.0))

    def add_random_vacancies(self, label='X', num_single=0, num_double=0, seed=None):
        if self._cached_coords is None:
            self.generate_lattice()
        indices = self.filtered_indices.get(label, [])
        if len(indices) == 0:
            return

        rng = np.random.default_rng(seed)
        rng.shuffle(indices)

        selected = 0
        for _ in range(num_single):
            if selected >= len(indices): break
            self.add_single_vacancy(label, indices[selected])
            selected += 1

        for _ in range(num_double):
            if selected >= len(indices): break
            self.add_double_vacancy(label, [indices[selected]])
            selected += 1

    def add_random_dopants(self, label='TM', num_dopants=0, dopant_intensity=0.8, seed=None):
        if self._cached_coords is None:
            self.generate_lattice()
        indices = self.filtered_indices.get(label, [])
        if len(indices) == 0:
            return

        rng = np.random.default_rng(seed)
        dopant_indices = rng.choice(indices, size=min(num_dopants, len(indices)), replace=False)
        for idx in dopant_indices:
            self.dopants.append((label, idx, dopant_intensity))

    def place_atoms_delta(self, positions, scales):
        img = np.zeros(self.size, dtype=np.float32)
        h, w = self.size
        xi = np.round(positions[:, 0]).astype(int)
        yi = np.round(positions[:, 1]).astype(int)
        mask = (xi >= 0) & (xi < w) & (yi >= 0) & (yi < h)
        np.add.at(img, (yi[mask], xi[mask]), scales[mask])
        return img

    def gaussian_kernel(self, size, sigma, A):
        r = np.arange(-size // 2 + 1, size // 2 + 1)
        X, Y = np.meshgrid(r, r)
        return A * np.exp(-(X**2 + Y**2) / (2 * sigma**2))

    def simulate(self, return_masks=False):
        coords = self.generate_lattice()
        total_img = np.zeros(self.size, dtype=np.float32)
        masks = {}

        for label, atoms in coords.items():
            scales = np.ones(len(atoms), dtype=np.float32)

            for vac_label, vac_idx, scale in self.vacancies:
                if vac_label == label:
                    if 0 <= vac_idx < len(atoms):
                        scales[vac_idx] = scale

            for dop_label, dop_idx, dop_scale in self.dopants:
                if dop_label == label and 0 <= dop_idx < len(atoms):
                    scales[dop_idx] = dop_scale

            delta = self.place_atoms_delta(atoms, scales)
            p = self.species_params.get(label, {'sigma': 2.0, 'A': 1.0})

            if self.use_fft:
                ksize = int(6 * p['sigma']) | 1
                kernel = self.gaussian_kernel(ksize, p['sigma'], p['A'])
                blurred = fftconvolve(delta, kernel, mode='same')
            else:
                blurred = p['A'] * gaussian_filter(delta, sigma=p['sigma'])

            total_img += blurred
            if return_masks:
                masks[label] = delta

        if return_masks:
            return total_img, masks
        return total_img

    def plot(self, img):
        plt.figure(figsize=(6, 6))
        plt.imshow(img, cmap='gray')
        plt.title("Simulated Atomic-Resolution TMD Image")
        plt.axis('off')
        plt.show()

    def get_defect_counts(self):
        if self.labels is None:
            raise ValueError("Run simulate() first to generate labels.")
        unique, counts = np.unique(self.labels, return_counts=True)
        return dict(zip(unique, counts))

    def save(self, img, path):
        np.save(path, img)
