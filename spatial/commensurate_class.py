import numpy as np
import matplotlib.pyplot as plt


# =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
# from this paper: Continuum model of the twisted graphene bilayer
# =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=

def generate_theta(n):
    p = np.arange(1, n + 1)
    q = np.arange(1, n + 1)
    P, Q = np.meshgrid(p, q)
    theta = np.rad2deg(np.arccos((3 * P ** 2 + 3 * P * Q + Q ** 2 / 2) / (3 * P ** 2 + 3 * P * Q + Q ** 2)))
    # mask1 = Q > P
    # theta[~mask1] = 0
    mask2 = (np.gcd(P, Q) == 1)
    theta[~mask2] = 0
    theta[theta == 0] = 0
    return theta


def generate_N(n):
    p = np.arange(1, n + 1)
    q = np.arange(1, n + 1)
    P, Q = np.meshgrid(p, q)
    mask1 = np.gcd(Q, 3) == 1
    mask2 = np.gcd(Q, 3) == 3
    N = np.zeros_like(P, dtype=int)
    N1 = 3 * P ** 2 + 3 * P * Q + Q ** 2
    N2 = P ** 2 + P * Q + Q ** 2 // 3
    N[mask1] = N1[mask1]
    N[mask2] = N2[mask2]
    return N


# =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
# from this paper: Electronic structure of turbostratic graphene
# =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=

def generate_theta_matrix(n):
    p = np.arange(1, n + 1)
    q = np.arange(1, n + 1)
    P, Q = np.meshgrid(p, q)
    i1 = 2 * P * Q
    i2 = 3 * Q ** 2 - P ** 2
    i3 = 3 * Q ** 2 + P ** 2
    angle = np.rad2deg(np.arccos(i2 / i3))
    mask1 = Q > P
    mask2 = (np.gcd(P, Q) == 1)
    angle[~mask1] = 0
    angle[~mask2] = 0
    return angle


def generate_N_matrix(n):
    p = np.arange(1, n + 1)
    q = np.arange(1, n + 1)
    P, Q = np.meshgrid(p, q)
    mask1 = Q > P
    mask2 = (np.gcd(P, Q) == 1)
    gamma = np.gcd(3 * Q + P, 3 * Q - P)
    sigma = 3 / np.gcd(P, 3).astype(int)
    N = 3 * (3 * Q ** 2 + P ** 2) / (sigma * gamma ** 2).astype(int)
    mask1 = Q > P
    mask2 = (np.gcd(P, Q) == 1)
    N[~mask1] = 0
    N[~mask2] = 0
    return N


def plot_lower_bound(ax, **kwargs):
    theta = np.linspace(0.001, 30, 512)
    t = np.deg2rad(theta)
    ax.plot(theta, 1 / (4*np.sin(t/2) ** 2), ls='--', **kwargs)


def get_lower_bound_thetas(n):
    p = 1
    q = np.arange(3, n + 1, 2)
    i2 = 3 * q ** 2 - p ** 2
    i3 = 3 * q ** 2 + p ** 2
    theta = np.rad2deg(np.arccos(i2 / i3))
    return theta

def get_N_by_theta(theta):
    t = np.deg2rad(theta)
    return 1 / (4*np.sin(t/2) ** 2)


class CommensurateLattice:

    def __init__(self, n=80):
        self.n = n
        self.theta_matrix = generate_theta_matrix(self.n)
        self.N_matrix = generate_N_matrix(self.n)
        self.ind = np.nonzero(self.theta_matrix)
        self.lower_bound_thetas = get_lower_bound_thetas(self.n)

    def plot(self, ax=None, ymax=1000):
        if ax is None:
            fig, ax = plt.subplots(1, 1, figsize=(7.2, 7.2))
        theta = self.theta_matrix[self.ind]
        N = self.N_matrix[self.ind]
        ax.scatter(theta, N, color='C0', s=8)
        ax.set_xlim(0, 60)
        ax.set_ylim(0, ymax)
        plot_lower_bound(ax, c='C1')
        ax.set_ylabel(r'$N$', fontsize=14)
        ax.set_xlabel(r'$\theta$', fontsize=14)

    def plot_polar(self, ax=None):
        if ax is None:
            fig = plt.figure(figsize=(7.2, 7.2))
            ax = fig.add_subplot(111, projection='polar')
            ax.set_thetamin(0)
            ax.set_thetamax(60)
            ax.set_theta_offset(np.deg2rad(-180 + 60))
            # ax.set_ylim(0, 1)
            ax.set_rscale('symlog')

        t = np.deg2rad(self.lower_bound_thetas)
        l = np.sqrt(1 / (np.sin(t)) ** 2)
        ax.scatter(t, l, s=10)
        ax.set_ylim(0, l.max())

        for angle in self.lower_bound_thetas:
            ax.plot([np.deg2rad(angle)] * 10, np.linspace(0, l.max(), 10), ls='--', c='C1', lw=0.5, zorder=-1)

    def show_lattice(self):
        pass